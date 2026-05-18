"""
FastAPI backend for the affordance-system service.

Run:
    python backend.py --password 123456 --host 0.0.0.0 --port 8800

Architecture
------------
* Single FastAPI app with **13 logical APIs**, all exposed over **one WebSocket
  endpoint** at ``/ws`` using a small JSON message envelope:

      {"type": "<api_name>", "request_id": "<uuid>", "payload": {...}}

  The backend replies asynchronously with messages of the same shape plus a
  ``"ok": true/false`` and ``"data"`` / ``"error"`` field. This satisfies the
  spec's requirement of "协议异步实时通信".

* Three connection roles share the same endpoint but identify themselves on
  connect via ``payload.role`` ∈ {"robot", "user", "admin"}.

* Admin operations (APIs 4, 5, 12, 13) require the ``--password`` CLI arg.

The 13 APIs
-----------
  1.  register_robot       robot → backend     register robot_id
  2.  register_user        user  → backend     register uuid
  3.  watch_robot          user  → backend     subscribe to robot_id
  4.  register_admin       user  → backend     password auth, grant admin
  5.  poweron              admin → backend     load model + memory + word map
  6.  infer_from_robot     robot → backend     full inference, push back
  7.  infer_from_user_img  user  → backend     inference with user image
  8.  infer_user_pref      user  → backend     fuse user preference directly
  9.  feedback             robot → backend     persist user feedback
 10.  annotate             user  → backend     run annotation model
 11.  dataset_sample       robot/user → backend  random PIAD sample
 12.  train_main           admin → backend     train main model
 13.  train_annotation     admin → backend     train annotation model

The watchers of a robot are notified for every inference result the robot
receives, so they can mirror the robot's view in real time.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ---------------------------------------------------------------------------
# Project imports (lazy where possible to keep startup fast)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from memory_system import ImageMemoryManager, MemoryManager  # noqa: E402

# Heavy imports (MyNet / annotation / dataset) are deferred to poweron / use.

# ---------------------------------------------------------------------------
# Server-wide state
# ---------------------------------------------------------------------------

AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab',
]

# Object labels from annotation/config_annotation.yaml (23 classes)
OBJECT_LABELS = [
    'bag', 'bed', 'bottle', 'bowl', 'chair', 'clock', 'DishWasher',
    'display', 'door', 'EarPhone', 'faucet', 'hat', 'keyboard', 'knife',
    'laptop', 'microwave', 'mug', 'refrigeator', 'scissors',
    'StorageFurniture', 'table', 'TrashCan', 'vase',
]

MEMORY_CACHE_DIR = os.path.join(PROJECT_ROOT, "Memory_cache")
os.makedirs(MEMORY_CACHE_DIR, exist_ok=True)

MEMORY_CACHE_PUSH_DIR = os.path.join(PROJECT_ROOT, "memory_cache_push")
os.makedirs(MEMORY_CACHE_PUSH_DIR, exist_ok=True)

GLOVE_PATH = os.path.join(PROJECT_ROOT, "glove/glove.6B.300d.txt")

@dataclass
class ServerState:
    password: str
    map_json_path: str

    robot_ids: List[str] = field(default_factory=list)
    user_uuids: List[str] = field(default_factory=list)
    admin_uuids: List[str] = field(default_factory=list)

    # robot_id -> [user_uuid, ...]
    watchers: Dict[str, List[str]] = field(default_factory=dict)

    # robot_id -> latest in-flight data dict (used by API 7 to enrich)
    robot_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # uuid / robot_id  ->  WebSocket
    sockets: Dict[str, WebSocket] = field(default_factory=dict)
    socket_role: Dict[str, str] = field(default_factory=dict)

    # Loaded resources after API 5 "poweron"
    main_model: Any = None
    annotation_models: Dict[str, Any] = field(default_factory=dict)
    image_memory: Optional[ImageMemoryManager] = None
    pref_memory: Optional[MemoryManager] = None
    word_yaml: Optional[Dict[str, Any]] = None
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    poweron_payload: Optional[Dict[str, Any]] = None

    # Cached PIAD datasets for API 11
    datasets: Dict[str, Any] = field(default_factory=dict)

    # Lightweight lock around model inference (serialise GPU access)
    model_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # create emb func to embedding the word
    emb_func: Optional[callable[any]] =None
    emb_dict: Optional[Dict[str, np.ndarray]] = None

    is_temb:bool = False

    # Cache: image_path -> base64 string to avoid repeated np.load + encode
    img_b64_cache: Dict[str, str] = field(default_factory=dict)

    # Whether to use preference memory for enhanced localization
    use_pref_memory: bool = True



STATE: ServerState  # populated in main()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"ok": True, "data": payload or {}}


def _err(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}


def _require_admin(uuid_: str) -> Optional[Dict[str, Any]]:
    if uuid_ not in STATE.admin_uuids:
        return _err("permission denied: admin required")
    return None


def _decode_b64_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.split(',', 1)[-1])
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    return np.array(img)


def _encode_image_b64(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1 \
            else arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def _send(socket_id: str, message: Dict[str, Any]) -> bool:
    ws = STATE.sockets.get(socket_id)
    if ws is None:
        return False
    try:
        await ws.send_text(json.dumps(message))
        return True
    except Exception:
        return False


async def _notify_watchers(robot_id: str, message: Dict[str, Any]):
    for uuid_ in STATE.watchers.get(robot_id, []):
        await _send(uuid_, message)


# ---------------------------------------------------------------------------
# API 1 — register_robot
# ---------------------------------------------------------------------------

async def api_register_robot(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    robot_id = payload.get("robot_id")
    if not robot_id:
        return _err("robot_id required")
    if robot_id in STATE.robot_ids:
        return _err(f"robot_id '{robot_id}' already registered (duplicate)")
    STATE.robot_ids.append(robot_id)
    STATE.watchers.setdefault(robot_id, [])
    # Bind socket → robot_id (the robot's socket key becomes its robot_id)
    STATE.sockets[robot_id] = STATE.sockets.pop(conn_id, STATE.sockets.get(conn_id))
    STATE.socket_role[robot_id] = "robot"
    return _ok({"robot_id": robot_id})


# ---------------------------------------------------------------------------
# API 2 — register_user
# ---------------------------------------------------------------------------

async def api_register_user(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    uuid_ = payload.get("uuid")
    if not uuid_:
        return _err("uuid required")
    if uuid_ not in STATE.user_uuids:
        STATE.user_uuids.append(uuid_)
    STATE.sockets[uuid_] = STATE.sockets.pop(conn_id, STATE.sockets.get(conn_id))
    STATE.socket_role[uuid_] = "user"
    return _ok({"uuid": uuid_})


# ---------------------------------------------------------------------------
# API 3 — watch_robot
# ---------------------------------------------------------------------------

async def api_watch_robot(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    uuid_ = payload.get("uuid")
    robot_id = payload.get("robot_id")
    if not uuid_ or not robot_id:
        return _err("uuid and robot_id required")
    if robot_id not in STATE.robot_ids:
        return _err(f"robot '{robot_id}' not registered")
    watchers = STATE.watchers.setdefault(robot_id, [])
    if uuid_ not in watchers:
        watchers.append(uuid_)
    return _ok({"robot_id": robot_id, "watchers": watchers})


# ---------------------------------------------------------------------------
# API 4 — register_admin
# ---------------------------------------------------------------------------

async def api_register_admin(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    uuid_ = payload.get("uuid")
    password = payload.get("password")
    if not uuid_ or password is None:
        return _err("uuid and password required")
    if password != STATE.password:
        return _err("invalid password")
    if uuid_ not in STATE.admin_uuids:
        STATE.admin_uuids.append(uuid_)
    # Hand back the full map json so the admin frontend can list / select
    # memories, annotation models, etc.
    mapping = {}
    if os.path.exists(STATE.map_json_path):
        try:
            with open(STATE.map_json_path, 'r') as f:
                mapping = json.load(f)
        except Exception as e:
            mapping = {"_load_error": str(e)}
    return _ok({"uuid": uuid_, "is_admin": True, "mapping": mapping,
                "map_path": STATE.map_json_path})


# ---------------------------------------------------------------------------
# API 5 — poweron (admin)
# ---------------------------------------------------------------------------

def _load_main_model(model_path: str):
    """Build a MyNet/IAG_TextEmb and load weights from a .pt/.pth file."""
    from model.MyNet import MyNet, IAG_TextEmb

    is_textemb = "textemb" in os.path.basename(model_path).lower()
    model = IAG_TextEmb(pre_train=False) if is_textemb else MyNet(pre_train=False)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        cleaned = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
    model.eval().to(STATE.device)
    return model, is_textemb


def _load_word_yaml(word_path: str) -> Dict[str, Any]:
    import yaml
    if not os.path.exists(word_path):
        return {}
    with open(word_path, 'r') as f:
        return yaml.safe_load(f) or {}

def _load_emb(path: str=GLOVE_PATH,word_yaml: Dict[str, Any]=None) -> Dict[str, np.ndarray]:
    print(f"[GloVe] Loading embeddings from: {path}")
    embeddings = {}
    af_list=word_yaml.get("affordance_labels", []) if word_yaml else []
    obj_list=word_yaml.get("object_labels", []) if word_yaml else []
    sem_dic=word_yaml.get("word_map", []) if word_yaml else []
    print(af_list,obj_list,sem_dic)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]

            # Skip words we don't need
            if word not in af_list and word not in obj_list and [word] not in sem_dic.values():
                continue
            if [word] in sem_dic.values():
                vector = np.array([float(x) for x in parts[1:]])
                for i in sem_dic.keys():
                    if sem_dic[i]==[word]:
                        embeddings[i]=vector
            elif word in af_list or word in obj_list:
                # print(f"Found affordance word in GloVe: {word}")
                vector = np.array([float(x) for x in parts[1:]])
                embeddings[word] = vector

    print("[GloVe] key:",list(embeddings.keys()))
    print(f"[GloVe] Loaded {len(embeddings)} word vectors")
    STATE.emb_dict = embeddings
    return embeddings


def _load_emb_func(affordance:str):
    if STATE.is_temb == False:
        aff_idx = AFFORDANCE_LABELS.index(affordance) if affordance in AFFORDANCE_LABELS else 0
        return torch.tensor([aff_idx], device=STATE.device,dtype=torch.float32)
    else :
        return STATE.emb_dict.get(affordance, torch.zeros(300, dtype=torch.float32))


async def api_poweron(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    uuid_ = payload.get("uuid")
    deny = _require_admin(uuid_)
    if deny:
        return deny

    # Persist the incoming spec to the map json
    spec = {k: payload[k] for k in
            ("model_path", "model_name", "memory_path",
             "word_path", "annotation_path") if k in payload}
    if not spec:
        # Fallback: load the existing json file
        if os.path.exists(STATE.map_json_path):
            with open(STATE.map_json_path, 'r') as f:
                spec = json.load(f)
        else:
            return _err("no spec provided and map json does not exist")
    else:
        with open(STATE.map_json_path, 'w') as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)

    try:
        # 1. Main model -------------------------------------------------------
        model_path = spec.get("model_path", "")
        STATE.main_model, is_textemb = _load_main_model(model_path)

        # 2. Memory stores ----------------------------------------------------
        # memory_path is a list of [db_path, description]. We treat the first
        # one ending in image_memories.db as the image store, others as pref.
        image_dir = None
        pref_dir = None
        for entry in spec.get("memory_path", []):
            if not entry:
                continue
            db_path = entry[0] if isinstance(entry, (list, tuple)) else entry
            store_dir = os.path.dirname(db_path) or "."
            if "image" in os.path.basename(db_path).lower() and image_dir is None:
                image_dir = store_dir
            elif pref_dir is None:
                pref_dir = store_dir

        if image_dir:
            STATE.image_memory = ImageMemoryManager(
                store_dir=image_dir,
                feature_dim=512,
                use_faiss=True,
                max_images_per_key=4,
                max_memory_images=4,
            )
        if pref_dir:
            STATE.pref_memory = MemoryManager(store_dir=pref_dir)
            STATE.pref_memory.indexer.eval()
            STATE.pref_memory.aligner.eval()

            # Optional: load learned indexer / aligner checkpoints.
            # Spec may include:
            #   "indexer_ckpt": "path/to/indexer.pt"
            #   "aligner_ckpt": "path/to/aligner.pt"
            # Falls back to <pref_dir>/indexer.pt and <pref_dir>/aligner.pt
            # if those files exist.
            def _resolve_ckpt(spec_key: str, default_name: str) -> Optional[str]:
                p = spec.get(spec_key, "")
                if p and os.path.exists(p):
                    return p
                cand = os.path.join(pref_dir, default_name)
                return cand if os.path.exists(cand) else None

            idx_ckpt = _resolve_ckpt("indexer_ckpt", "indexer.pt")
            if idx_ckpt:
                try:
                    sd = torch.load(idx_ckpt, map_location="cpu", weights_only=False)
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    STATE.pref_memory.indexer.load_state_dict(
                        {k.replace("module.", ""): v for k, v in sd.items()},
                        strict=False,
                    )
                    print(f"[poweron] Loaded MemoryIndexer ckpt: {idx_ckpt}")
                except Exception as e:
                    print(f"[poweron] WARN: failed to load indexer ckpt: {e}")

            aln_ckpt = _resolve_ckpt("aligner_ckpt", "aligner.pt")
            if aln_ckpt:
                try:
                    sd = torch.load(aln_ckpt, map_location="cpu", weights_only=False)
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    STATE.pref_memory.aligner.load_state_dict(
                        {k.replace("module.", ""): v for k, v in sd.items()},
                        strict=False,
                    )
                    print(f"[poweron] Loaded MemoryAligner ckpt: {aln_ckpt}")
                except Exception as e:
                    print(f"[poweron] WARN: failed to load aligner ckpt: {e}")

        # 3. Word / yaml -----------------------------------------------------
        STATE.word_yaml = _load_word_yaml(spec.get("word_path", ""))
        if is_textemb:
            STATE.is_temb = True
            STATE.emb_dict = _load_emb(path=GLOVE_PATH, word_yaml=STATE.word_yaml)
            STATE.emb_func = _load_emb_func

        # 4. Annotation models ----------------------------------------------
        from annotation.annotation_model import (
            AnnotationModelScheme1, AnnotationModelScheme2,
        )
        anno_paths = spec.get("annotation_path", [])
        if len(anno_paths) >= 1 and os.path.exists(anno_paths[0]):
            m1 = AnnotationModelScheme1()
            st = torch.load(anno_paths[0], map_location='cpu', weights_only=False)
            if isinstance(st, dict) and 'state_dict' in st:
                st = st['state_dict']
            m1.load_state_dict({k.replace('module.', ''): v for k, v in st.items()},
                               strict=False)
            STATE.annotation_models["scheme1"] = m1.eval().to(STATE.device)
        if len(anno_paths) >= 2 and os.path.exists(anno_paths[1]):
            m2 = AnnotationModelScheme2()
            st = torch.load(anno_paths[1], map_location='cpu', weights_only=False)
            if isinstance(st, dict) and 'state_dict' in st:
                st = st['state_dict']
            m2.load_state_dict({k.replace('module.', ''): v for k, v in st.items()},
                               strict=False)
            STATE.annotation_models["scheme2"] = m2.eval().to(STATE.device)

        STATE.poweron_payload = spec

        return _ok({
            "model_loaded": True,
            "is_textemb": is_textemb,
            "image_memory": bool(STATE.image_memory),
            "pref_memory": bool(STATE.pref_memory),
            "annotation_models": list(STATE.annotation_models.keys()),
            "device": str(STATE.device),
        })
    except Exception as e:
        traceback.print_exc()
        return _err(f"poweron failed: {e}")


# ---------------------------------------------------------------------------
# API 6 / 7 / 8 — inference variants
# ---------------------------------------------------------------------------

def _decode_point_cloud(b64_or_list) -> np.ndarray:
    if isinstance(b64_or_list, str):
        raw = base64.b64decode(b64_or_list)
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        return arr.reshape(-1, 3)
    return np.array(b64_or_list, dtype=np.float32)


def _pc_normalize(pc: np.ndarray) -> np.ndarray:
    centroid = pc.mean(0)
    pc = pc - centroid
    m = np.max(np.sqrt((pc ** 2).sum(axis=1))) + 1e-8
    return pc / m


def _knn_interpolate_pref(
    pref_np: np.ndarray,   # [N_p]  preference at abstract points
    src_xyz: np.ndarray,   # [N_p, 3]  abstract point positions
    tgt_xyz: np.ndarray,   # [N_raw, 3]  raw point positions
    k: int = 3,
) -> np.ndarray:
    """Up-sample an N_p-dim preference vector to N_raw via inverse-distance KNN.

    Uses the same weighted-nearest-neighbour interpolation as PointNet++
    FeaturePropagation but implemented with pure NumPy / torch for ease of
    use outside the model graph.

    Returns
    -------
    np.ndarray shape [N_raw]
    """
    # Squared distances [N_raw, N_p]
    diff = tgt_xyz[:, None, :] - src_xyz[None, :, :]   # [N_raw, N_p, 3]
    dist2 = (diff ** 2).sum(-1)                         # [N_raw, N_p]

    k = min(k, src_xyz.shape[0])
    nn_idx = np.argpartition(dist2, k, axis=1)[:, :k]   # [N_raw, k]
    nn_dist2 = np.take_along_axis(dist2, nn_idx, axis=1) # [N_raw, k]

    # Inverse-distance weights; guard against exact overlap
    weights = 1.0 / (nn_dist2 + 1e-8)
    weights = weights / weights.sum(axis=1, keepdims=True)

    nn_pref = pref_np[nn_idx]                            # [N_raw, k]
    return (weights * nn_pref).sum(axis=1)               # [N_raw]


async def _capture_pref_entry(
    *,
    img_np: np.ndarray,
    points: np.ndarray,
    sub_box: np.ndarray,
    obj_box: np.ndarray,
    object_category: str,
    affordance: str,
    preference: np.ndarray,
    outcome: str = "unknown",
) -> Optional[str]:
    """Extract ARM module features from main model and save a pref cache entry.

    Calls get_F_affordance_and_others() to obtain the ARM feature (memory index),
    PointNet features, and preference heatmap.  Packages them as a .npz file in
    memory_cache_push/ for later admin review and upload.

    Returns the saved file path, or None if the main model is not loaded.
    """
    if STATE.main_model is None:
        return None

    from data_utils.dataset import img_normalize_val

    img_pil = Image.fromarray(img_np).resize((224, 224))
    img_tensor = img_normalize_val(img_pil).unsqueeze(0).to(STATE.device)

    def _norm_box(b):
        arr = np.asarray(b, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 224.0
        return arr.clip(0.0, 1.0)

    sub_t = torch.from_numpy(_norm_box(sub_box)).unsqueeze(0).to(STATE.device)
    obj_t = torch.from_numpy(_norm_box(obj_box)).unsqueeze(0).to(STATE.device)

    is_textemb = STATE.emb_func is not None
    if is_textemb:
        aff_vec = STATE.emb_func(affordance)
        aff_tensor = torch.as_tensor(aff_vec, dtype=torch.float32, device=STATE.device)
        if aff_tensor.dim() == 1:
            aff_tensor = aff_tensor.unsqueeze(0)
    else:
        aff_tensor = None

    pts = _pc_normalize(points).T
    pts_tensor = torch.from_numpy(pts).float().unsqueeze(0).to(STATE.device)

    loop = asyncio.get_event_loop()
    async with STATE.model_lock:
        def _feat_forward():
            STATE.main_model.eval()
            with torch.no_grad():
                if is_textemb:
                    return STATE.main_model.get_F_affordance_and_others(
                        img_tensor, pts_tensor, sub_t, obj_t, aff_tensor)
                return STATE.main_model.get_F_affordance_and_others(
                    img_tensor, pts_tensor, sub_t, obj_t)
        try:
            feat_out = await loop.run_in_executor(None, _feat_forward)
        except Exception:
            return None

    if feat_out is None:
        return None

    # feat_out = (arm_feat [B,N_p+N_i,C], F_j, F_p_wise)
    arm_feat, F_j, F_p_wise = feat_out

    # F_p_wise[-1]: [l3_xyz [B,3,N_p], l3_pts [B,C,N_p]]
    # Store in [N_p, *] layout for the aligner
    l3_xyz = F_p_wise[-1][0]  # [B, 3, N_p]
    l3_pts = F_p_wise[-1][1]  # [B, C, N_p]
    np_xyz = l3_xyz[0].T.detach().cpu().numpy()    # [N_p, 3]
    np_feat = l3_pts[0].T.detach().cpu().numpy()   # [N_p, C]

    # Down-sample preference to N_p for storage (will be up-sampled at retrieval)
    pts_norm = _pc_normalize(points)               # [N_raw, 3]
    pref_np_abs = _knn_interpolate_pref(preference, pts_norm, np_xyz, k=3)
    # Keep N_p-dim version; also store raw N_raw preference for fast residual path
    ts = int(time.time() * 1000)
    fname = f"pref_{ts}.npz"
    save_path = os.path.join(MEMORY_CACHE_PUSH_DIR, fname)

    np.savez(save_path,
             arm_feature=arm_feat.detach().cpu().numpy(),   # [1, N_p+N_i, C]
             l3_xyz=np_xyz,       # [N_p, 3]  abstract point coords
             l3_features=np_feat, # [N_p, C]  abstract point features
             pref_at_np=pref_np_abs,  # [N_p]   preference downsampled to N_p
             preference=preference,   # [N_raw]  original full-res preference
             points=points,           # [N_raw, 3]
             outcome=np.array(outcome),
             object=np.array(object_category),
             affordance=np.array(affordance))
    return save_path


async def _capture_img_entry(
    *,
    img_np: np.ndarray,
    sub_box: np.ndarray,
    obj_box: np.ndarray,
    object_category: str,
    affordance: str,
) -> Optional[str]:
    """Extract image features (F_i, F_s, F_e) and save an image cache entry.

    Calls get_img_and_feature() to obtain F_i / F_s / F_e from the main model.
    Packages them together with the image and annotation boxes as a .npz file
    in memory_cache_push/ for later admin review and upload to image memory.

    Returns the saved file path, or None if the main model is not loaded.
    """
    if STATE.main_model is None:
        return None

    from data_utils.dataset import img_normalize_val

    img_pil = Image.fromarray(img_np).resize((224, 224))
    img_tensor = img_normalize_val(img_pil).unsqueeze(0).to(STATE.device)
    img_224 = np.array(img_pil)  # for storage

    def _norm_box(b):
        arr = np.asarray(b, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 224.0
        return arr.clip(0.0, 1.0)

    sub_t = torch.from_numpy(_norm_box(sub_box)).unsqueeze(0).to(STATE.device)
    obj_t = torch.from_numpy(_norm_box(obj_box)).unsqueeze(0).to(STATE.device)

    loop = asyncio.get_event_loop()
    async with STATE.model_lock:
        def _img_feat_forward():
            STATE.main_model.eval()
            with torch.no_grad():
                return STATE.main_model.get_img_and_feature(img_tensor, sub_t, obj_t)
        try:
            feat_out = await loop.run_in_executor(None, _img_feat_forward)
        except Exception:
            return None

    if feat_out is None:
        return None

    F_i, F_s, F_e = feat_out

    ts = int(time.time() * 1000)
    fname = f"img_{ts}.npz"
    save_path = os.path.join(MEMORY_CACHE_PUSH_DIR, fname)

    # Encode image as PNG bytes for compact storage
    buf = io.BytesIO()
    Image.fromarray(img_224).save(buf, format='PNG')
    img_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)

    np.savez(save_path,
             img_bytes=img_bytes,
             sub_box=_norm_box(sub_box),
             obj_box=_norm_box(obj_box),
             object=np.array(object_category),
             affordance=np.array(affordance),
             F_i=F_i.detach().cpu().numpy(),
             F_s=F_s.detach().cpu().numpy(),
             F_e=F_e.detach().cpu().numpy())
    return save_path


async def _run_inference(*, img_np: np.ndarray, points: np.ndarray,
                         sub_box: np.ndarray, obj_box: np.ndarray,
                         object_category: str, affordance: str) -> Dict[str, Any]:
    """Run the main IAG_TextEmb model.

    图像记忆库的作用：
      1. 提供演示图像（返回给前端展示）
      2. 提供标注数据（sub_box / obj_box），替代机器人上报的框，用于模型 forward
    若记忆库中无匹配条目，则回退到调用方传入的 sub_box / obj_box。
    """
    if STATE.main_model is None:
        raise RuntimeError("main model not loaded; admin must call poweron first")

    from data_utils.dataset import img_normalize_val

    img_pil = Image.fromarray(img_np).resize((224, 224))
    img_tensor = img_normalize_val(img_pil).unsqueeze(0).to(STATE.device)

    pts = _pc_normalize(points).T  # [3, N]
    pts_tensor = torch.from_numpy(pts).float().unsqueeze(0).to(STATE.device)

    # 默认使用调用方提供的标注框，归一化到 [0,1]（机器人上报的是像素坐标）
    def _norm_box(b):
        arr = np.asarray(b, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 224.0
        return arr.clip(0.0, 1.0)

    sub_t = torch.from_numpy(_norm_box(sub_box)).unsqueeze(0).to(STATE.device)
    obj_t = torch.from_numpy(_norm_box(obj_box)).unsqueeze(0).to(STATE.device)

    info: Dict[str, Any] = {"object": object_category, "affordance": affordance}

    # ── 构建 text embedding ───────────────────────────────────────────────
    is_textemb = STATE.emb_func is not None
    if is_textemb:
        aff_vec = STATE.emb_func(affordance)
        aff_tensor = torch.as_tensor(aff_vec, dtype=torch.float32, device=STATE.device)
        if aff_tensor.dim() == 1:
            aff_tensor = aff_tensor.unsqueeze(0)
    else:
        aff_tensor = None

    # ── 图像记忆库查询 ────────────────────────────────────────────────────
    # 作用一：获取演示图像（前端展示用）
    # 作用二：获取该 (object, affordance) 对应的标注框，用于模型推理
    retrieved_images: List[str] = []
    if STATE.image_memory is not None:
        entries = STATE.image_memory.store.retrieve_by_key(
            object_category, affordance, top_k=4)
        info["image_memory_hits"] = len(entries)

        for e in entries:
            # 演示图像
            p = e.get("image_path", "")
            if p and os.path.exists(p):
                try:
                    if p not in STATE.img_b64_cache:
                        STATE.img_b64_cache[p] = _encode_image_b64(np.load(p))
                    retrieved_images.append(STATE.img_b64_cache[p])
                except Exception:
                    pass

        # 用记忆库中最新条目的标注框替换当前推理所用的框
        # 记忆库中的框可能是像素坐标（0-224），需要归一化到 [0,1] 以匹配模型期望
        if entries:
            best = entries[0]
            mem_sub = best.get("sub_box_decoded")
            mem_obj = best.get("obj_box_decoded")
            if mem_sub is not None and mem_obj is not None:
                mem_sub = np.asarray(mem_sub, dtype=np.float32)
                mem_obj = np.asarray(mem_obj, dtype=np.float32)
                # Normalise: if values > 1, assume pixel coords in 224×224 space
                if mem_sub.max() > 1.0:
                    mem_sub = mem_sub / 224.0
                if mem_obj.max() > 1.0:
                    mem_obj = mem_obj / 224.0
                sub_t = torch.from_numpy(mem_sub).unsqueeze(0).to(STATE.device)
                obj_t = torch.from_numpy(mem_obj).unsqueeze(0).to(STATE.device)
                info["annotation_from_memory"] = True
                info["inference_mode"] = "memory_annotation"
            else:
                info["inference_mode"] = "fallback_provided_box"
    else:
        info["inference_mode"] = "no_image_memory"

    # ── 模型推理（拆分 forward，以便接入点云偏好记忆） ────────────────────
    loop = asyncio.get_event_loop()
    async with STATE.model_lock:
        def _forward():
            STATE.main_model.eval()
            with torch.no_grad():
                has_split = hasattr(STATE.main_model, 'get_F_affordance_and_others')
                if not has_split:
                    # MyNet (non-TextEmb): no split method, do full forward
                    out_raw = STATE.main_model(img_tensor, pts_tensor, sub_t, obj_t)
                    _3daff = out_raw[0] if isinstance(out_raw, (tuple, list)) else out_raw
                    return _3daff, None, None

                # Step 1: 图像编码 + PointNet + JRA + ARM，获取 arm_feat 和点特征
                if is_textemb:
                    arm_feat, F_j, F_p_wise = STATE.main_model.get_F_affordance_and_others(
                        img_tensor, pts_tensor, sub_t, obj_t, aff_tensor)
                else:
                    arm_feat, F_j, F_p_wise = STATE.main_model.get_F_affordance_and_others(
                        img_tensor, pts_tensor, sub_t, obj_t)
                # Step 2: Decoder → sigmoid-ed per-point affordance
                if is_textemb:
                    _3daff, logits, to_KL = STATE.main_model.decoder(
                        F_j, arm_feat, F_p_wise, aff_tensor)
                else:
                    _3daff, logits, to_KL = STATE.main_model.decoder(
                        F_j, arm_feat, F_p_wise)
                return _3daff, arm_feat, F_p_wise
        try:
            out = await loop.run_in_executor(None, _forward)
        except Exception as e:
            raise RuntimeError(f"main model forward failed: {e}")

    _3daff, arm_feat, F_p_wise = out
    pred_pref = _3daff.squeeze().detach().cpu().numpy()  # [N_raw]

    # ── 点云偏好记忆检索与融合 ────────────────────────────────────────────
    fused_pref = pred_pref
    info["pref_memory_applied"] = False
    info["use_pref_memory"] = STATE.use_pref_memory
    print("will search pm")
    if (STATE.use_pref_memory
            and STATE.pref_memory is not None
            and F_p_wise is not None
            and STATE.pref_memory.store.count() > 0):
        print("searching preference memory...")
        try:
            # l3_xyz [B,3,N_p]  l3_pts [B,C,N_p]
            l3_xyz = F_p_wise[-1][0][0].T.detach().cpu().numpy()  # [N_p, 3]
            l3_feat = F_p_wise[-1][1][0].T.detach().cpu().numpy() # [N_p, C]
            print("l3_xyz shape:", l3_xyz.shape, "l3_feat shape:", l3_feat.shape)
            pts_norm = _pc_normalize(points)                        # [N_raw, 3]

            # retrieve_and_fuse 在 N_p 空间操作：
            #   - l3_xyz 是归一化坐标，与存储时一致
            #   - l3_feat [N_p, C] 作为 current_point_features
            print("pm_lib stores total:",STATE.pref_memory.store.count())
            
            pref_fused_np = STATE.pref_memory.retrieve_and_fuse(
                arm_feature=arm_feat,
                current_point_cloud=l3_xyz,
                current_point_features=l3_feat,
                affordance_label=affordance,
                object_category=object_category,
            )  # [N_p]
            '''
            loop = asyncio.get_event_loop()
            pref_fused_np = await loop.run_in_executor(
                None,
                lambda: STATE.pref_memory.retrieve_and_fuse(
                    arm_feature=arm_feat,
                    current_point_cloud=l3_xyz,
                    current_point_features=l3_feat,
                    affordance_label=affordance,
                    object_category=object_category,
                )
            )'''
            print("[main]:search end")
            if pref_fused_np is not None and pref_fused_np.size == l3_xyz.shape[0]:
                # 上采样 N_p → N_raw
                print("search complete, fusing preference from memory...")
                pref_fused_raw = _knn_interpolate_pref(
                    pref_fused_np, l3_xyz, pts_norm, k=3)  # [N_raw]

                # 残差叠加（sigmoid 后直接加，再 clamp）
                alpha = 0.9
                fused_pref = np.clip(pred_pref + alpha * pref_fused_raw, 0.0, 1.0)
                info["pref_memory_applied"] = True
                info["pref_memory_hits"] = int(STATE.pref_memory.store.count())
                print("search complete")
        except Exception as e:
            info["pref_memory_error"] = str(e)
    print("[run_inference] done")
    return {
        "preference": fused_pref.tolist(),
        "preference_raw": pred_pref.tolist(),
        "points": points.tolist(),
        "retrieved_images": retrieved_images,
        "info": info,
    }


async def api_infer_from_robot(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    robot_id = payload.get("robot_id")
    if robot_id not in STATE.robot_ids:
        return _err("robot not registered")
    try:
        img_np = _decode_b64_image(payload["image"])
        points = _decode_point_cloud(payload["points"])
        sub_box = payload.get("sub_box", [0, 0, 0, 0])
        obj_box = payload.get("obj_box", [0, 0, 0, 0])
        object_cat = payload["object"]
        affordance = payload["affordance"]
    except KeyError as e:
        return _err(f"missing field: {e}")

    # Cache current robot data so API 7 / feedback can reuse it.
    STATE.robot_state[robot_id] = {
        "image": img_np, "points": points,
        "sub_box": sub_box, "obj_box": obj_box,
        "object": object_cat, "affordance": affordance,
        "gt": payload.get("gt"),
    }

    result = await _run_inference(
        img_np=img_np, points=points,
        sub_box=np.asarray(sub_box, dtype=np.float32),
        obj_box=np.asarray(obj_box, dtype=np.float32),
        object_category=object_cat,
        affordance=affordance,
    )
    if payload.get("gt") is not None:
        result.setdefault("info", {})["gt"] = payload["gt"]

    # Push to watchers
    await _notify_watchers(robot_id, {
        "type": "robot_inference_update",
        "payload": {"robot_id": robot_id, **result},
    })
    return _ok(result)


async def api_infer_from_user_img(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """API 7 — user uploads an annotated 2D image with a robot_id.

    Optional ``annotation`` field: if present (user accepted annotation in
    watch.html), triggers ARM feature capture + pref cache entry saving to
    memory_cache_push/, and if the annotation includes sub/obj boxes, also
    saves an image memory cache entry.
    """
    robot_id = payload.get("robot_id")
    if robot_id not in STATE.robot_ids:
        return _err("robot not registered / unknown")
    cur = STATE.robot_state.get(robot_id)
    if cur is None:
        return _err("robot has no in-flight data; wait for the robot to push first")
    try:
        extra_img = _decode_b64_image(payload["image"])
    except KeyError:
        return _err("image required")

    annotation = payload.get("annotation")  # may be None

    # Use annotation boxes if provided, otherwise fall back to robot's boxes
    if annotation and annotation.get("sub_box") and annotation.get("obj_box"):
        use_sub = np.asarray(annotation["sub_box"], dtype=np.float32)
        use_obj = np.asarray(annotation["obj_box"], dtype=np.float32)
        use_obj_cat = annotation.get("object_label") or cur["object"]
        use_affordance = annotation.get("action_label") or cur["affordance"]
    else:
        use_sub = np.asarray(cur["sub_box"], dtype=np.float32)
        use_obj = np.asarray(cur["obj_box"], dtype=np.float32)
        use_obj_cat = cur["object"]
        use_affordance = cur["affordance"]

    result = await _run_inference(
        img_np=cur["image"], points=cur["points"],
        sub_box=use_sub,
        obj_box=use_obj,
        object_category=use_obj_cat,
        affordance=use_affordance,
    )
    # Append the user's uploaded image to the retrieved set so the frontend
    # can show it as part of the image memory used in this turn.
    result["retrieved_images"].append(_encode_image_b64(extra_img))
    result["info"]["user_image_appended"] = True

    # If the user accepted the annotation, capture ARM features and image features
    img_cache_path: Optional[str] = None
    pref_cache_path: Optional[str] = None
    if annotation is not None and STATE.main_model is not None:
        pts = cur.get("points")
        if isinstance(pts, np.ndarray) and pts.size > 0:
            try:
                pref_cache_path = await _capture_pref_entry(
                    img_np=cur["image"],
                    points=pts,
                    sub_box=use_sub,
                    obj_box=use_obj,
                    object_category=use_obj_cat,
                    affordance=use_affordance,
                    preference=np.asarray(result["preference"], dtype=np.float32),
                    outcome="成功",
                )
            except Exception:
                pass
        try:
            img_cache_path = await _capture_img_entry(
                img_np=extra_img,
                sub_box=use_sub,
                obj_box=use_obj,
                object_category=use_obj_cat,
                affordance=use_affordance,
            )
        except Exception:
            pass
        if img_cache_path:
            result["info"]["img_cache_path"] = img_cache_path
        if pref_cache_path:
            result["info"]["pref_cache_path"] = pref_cache_path

    await _notify_watchers(robot_id, {
        "type": "robot_inference_update",
        "payload": {"robot_id": robot_id, **result},
    })
    return _ok(result)


async def api_infer_user_pref(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """API 8 — user uploads a preference heat-map directly (no model call)."""
    robot_id = payload.get("robot_id")
    if robot_id not in STATE.robot_ids:
        return _err("robot not registered")
    pref = payload.get("preference")
    if pref is None:
        return _err("preference (list of per-point scores) required")
    pref_np = np.asarray(pref, dtype=np.float32)

    cur = STATE.robot_state.get(robot_id, {})
    result = {
        "preference": pref_np.tolist(),
        "points": cur.get("points", np.zeros((0, 3))).tolist()
        if isinstance(cur.get("points"), np.ndarray) else cur.get("points", []),
        "retrieved_images": [],
        "info": {"source": "user_supplied", "object": cur.get("object"),
                 "affordance": cur.get("affordance")},
    }

    # Also capture ARM features and save pref cache entry in memory_cache_push/
    pts = cur.get("points")
    pts_np = pts if isinstance(pts, np.ndarray) else np.zeros((0, 3), dtype=np.float32)
    pref_cache_path: Optional[str] = None
    if STATE.main_model is not None and pts_np.size > 0:
        cur_img = cur.get("image")
        if cur_img is not None:
            try:
                pref_cache_path = await _capture_pref_entry(
                    img_np=cur_img,
                    points=pts_np,
                    sub_box=np.asarray(cur.get("sub_box", [0, 0, 0, 0]), dtype=np.float32),
                    obj_box=np.asarray(cur.get("obj_box", [0, 0, 0, 0]), dtype=np.float32),
                    object_category=cur.get("object", ""),
                    affordance=cur.get("affordance", ""),
                    preference=pref_np,
                    outcome="成功",
                )
            except Exception:
                pass

    await _notify_watchers(robot_id, {
        "type": "robot_inference_update",
        "payload": {"robot_id": robot_id, **result, "source": "user_preference"},
    })
    return _ok({**result, "pref_cache_path": pref_cache_path})


# ---------------------------------------------------------------------------
# API 9 — feedback
# ---------------------------------------------------------------------------

async def api_feedback(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    robot_id = payload.get("robot_id")
    pref = payload.get("preference")
    outcome = payload.get("outcome", "unknown")  # 优秀/成功/失败
    if robot_id not in STATE.robot_ids or pref is None:
        return _err("robot_id and preference required")
    
    pref_np = np.asarray(pref, dtype=np.float32)
    cur = STATE.robot_state.get(robot_id, {})
    pts = cur.get("points")
    
    if not isinstance(pts, np.ndarray):
        pts = np.zeros((0, 3), dtype=np.float32)
    
    # 1. 保存到本地缓存（原有逻辑保留）
    cache_path = os.path.join(MEMORY_CACHE_DIR,
                              f"feedback_{robot_id}_{int(time.time()*1000)}.npz")
    np.savez(cache_path,
             preference=pref_np, points=pts,
             outcome=np.array(outcome),
             object=np.array(cur.get("object", "")),
             affordance=np.array(cur.get("affordance", "")))
    
    # 2. 捕获ARM特征并保存到记忆库推送缓存（原有逻辑保留）
    pref_cache_path: Optional[str] = None
    if STATE.main_model is not None and pts.size > 0:
        try:
            cur_img = cur.get("image")
            if cur_img is not None:
                pref_cache_path = await _capture_pref_entry(
                    img_np=cur_img,
                    points=pts,
                    sub_box=np.asarray(cur.get("sub_box", [0, 0, 0, 0]), dtype=np.float32),
                    obj_box=np.asarray(cur.get("obj_box", [0, 0, 0, 0]), dtype=np.float32),
                    object_category=cur.get("object", ""),
                    affordance=cur.get("affordance", ""),
                    preference=pref_np,
                    outcome=outcome,
                )
        except Exception:
            pass
    
    # ============ 新增关键部分：立即更新并推送 ============
    # 3. 用用户提交的preference立即更新当前机器人的状态
    if robot_id in STATE.robot_state:
        # 更新状态中的偏好值，确保后续操作基于最新反馈
        STATE.robot_state[robot_id]["last_feedback"] = pref_np
    
    # 4. 构造即时更新的消息，推送给所有监视者
    update_payload = {
        "robot_id": robot_id,
        "preference": pref_np.tolist(),  # 使用用户提交的偏好值
        "points": pts.tolist() if isinstance(pts, np.ndarray) else [],
        "retrieved_images": [],  # 反馈更新不包含图像记忆
        "info": {
            "source": "immediate_feedback",
            "object": cur.get("object", ""),
            "affordance": cur.get("affordance", ""),
            "outcome": outcome,
            "timestamp": time.time()
        }
    }
    
    # 5. 立即推送更新给所有watcher
    await _notify_watchers(robot_id, {
        "type": "robot_inference_update",  # 使用与推理结果相同的消息类型
        "payload": update_payload
    })
    
    # 6. 立即推送反馈给机器人本身（实现用户实时指导）
    if robot_id in STATE.sockets:
        await _send(robot_id, {
            "type": "user_feedback_received",  # 新的消息类型，专门用于通知机器人
            "payload": {
                "robot_id": robot_id,
                "preference": pref_np.tolist(),
                "points": pts.tolist() if isinstance(pts, np.ndarray) else [],
                "outcome": outcome,
                "timestamp": time.time(),
                "source": "monitor_user"
            }
        })
    
    # 6. 立即推送反馈给机器人本身（实现用户实时指导）
    if robot_id in STATE.sockets:
        await _send(robot_id, {
            "type": "user_feedback_received",  # 新的消息类型，专门用于通知机器人
            "payload": {
                "robot_id": robot_id,
                "preference": pref_np.tolist(),
                "points": pts.tolist() if isinstance(pts, np.ndarray) else [],
                "outcome": outcome,
                "timestamp": time.time(),
                "source": "monitor_user"
            }
        })
    
    # ============ 新增部分结束 ============
    
    return _ok({
        "cache_path": cache_path, 
        "pref_cache_path": pref_cache_path,
        "immediate_update_sent": True,  # 新增返回字段，确认已推送
        "watchers_notified": len(STATE.watchers.get(robot_id, []))
    })

# ---------------------------------------------------------------------------
# API 10 — annotate
# ---------------------------------------------------------------------------

async def api_annotate(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    if "scheme2" not in STATE.annotation_models and "scheme1" not in STATE.annotation_models:
        return _err("annotation model not loaded (call poweron)")
    try:
        img_np = _decode_b64_image(payload["image"])
    except KeyError:
        return _err("image required")
    scheme = payload.get("scheme", "scheme2")
    if scheme not in STATE.annotation_models:
        scheme = next(iter(STATE.annotation_models))
    model = STATE.annotation_models[scheme]

    from data_utils.dataset import img_normalize_val
    img_pil = Image.fromarray(img_np).resize((224, 224))
    img_t = img_normalize_val(img_pil).unsqueeze(0).to(STATE.device)

    loop = asyncio.get_event_loop()
    async with STATE.model_lock:
        def _anno_forward():
            with torch.no_grad():
                print("STATE.robot_state:", STATE.robot_state,"STATE.robot_ids:", STATE.robot_ids)
                return model(img_t,object_wv=torch.from_numpy(STATE.emb_dict.get(STATE.robot_state.get(STATE.robot_ids[0]).get("object", "").lower(), None)).float().squeeze(0).to(STATE.device) if STATE.emb_dict else None)
        out = await loop.run_in_executor(None, _anno_forward)

    if not isinstance(out, dict):
        return _err("annotation model returned unexpected output type")

    def _box_to_list(v):
        """Convert normalised [0,1] box tensor to pixel coords [0, 224].
        BoxRegressionHead now applies sigmoid internally, so output is already [0,1]."""
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            return (arr * 224.0).tolist()
        return v

    def _to_list(v):
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            return arr.tolist()
        return v

    # Remap model output keys to frontend-expected keys:
    #   subject_box   -> sub_box   (green, label "sub")
    #   object_box    -> obj_box   (yellow, label "obj")
    #   action_logits -> action_label (top-1 affordance name)
    #   object_logits -> object_label (top-1 object name)
    sub_box = _box_to_list(out.get("subject_box"))
    obj_box = _box_to_list(out.get("object_box"))

    action_label = ""
    action_logits = out.get("action_logits")
    if action_logits is not None and isinstance(action_logits, torch.Tensor):
        idx = int(action_logits.squeeze(0).argmax().item())
        action_label = AFFORDANCE_LABELS[idx] if idx < len(AFFORDANCE_LABELS) else ""

    object_label = ""
    object_logits = out.get("object_logits")
    if object_logits is not None and isinstance(object_logits, torch.Tensor):
        idx = int(object_logits.squeeze(0).argmax().item())
        object_label = OBJECT_LABELS[idx] if idx < len(OBJECT_LABELS) else str(idx)

    return _ok({
        "scheme": scheme,
        "sub_box": sub_box,
        "obj_box": obj_box,
        "action_label": action_label,
        "object_label": object_label,
        # keep raw outputs for debugging
        "action_embed": _to_list(out.get("action_embed")),
        "object_embed": _to_list(out.get("object_embed")),
    })


# ---------------------------------------------------------------------------
# API 11 — dataset_sample
# ---------------------------------------------------------------------------

def _ensure_dataset(setting: str):
    if setting in STATE.datasets:
        return STATE.datasets[setting]
    from data_utils.dataset import PIAD
    data_root = os.path.join(PROJECT_ROOT, "Data", setting)
    ds = PIAD(
        run_type='val', setting_type=setting,
        point_path=os.path.join(data_root, "Point_Test.txt"),
        img_path=os.path.join(data_root, "Img_Test.txt"),
        box_path=os.path.join(data_root, "Box_Test.txt"),
    )
    STATE.datasets[setting] = ds
    return ds


async def api_dataset_sample(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    setting = payload.get("setting", "Seen")
    try:
        ds = _ensure_dataset(setting)
    except Exception as e:
        return _err(f"dataset load failed: {e}")

    import random
    idx = payload.get("index")
    if idx is None:
        idx = random.randint(0, len(ds) - 1)
    sample = ds[idx]
    Img, Point, aff_label, img_path, point_path, sub_box, obj_box = sample

    object_name = os.path.basename(img_path).split('_')[-3]
    affordance = os.path.basename(img_path).split('_')[-2]

    raw_img = np.array(Image.open(img_path).convert('RGB').resize((224, 224)))

    return _ok({
        "index": idx,
        "image": _encode_image_b64(raw_img),
        "points": Point.T.tolist(),  # back to [N, 3]
        "gt_preference": np.asarray(aff_label).flatten().tolist(),
        "object": object_name,
        "affordance": affordance,
        "sub_box": sub_box.tolist() if hasattr(sub_box, 'tolist') else list(sub_box),
        "obj_box": obj_box.tolist() if hasattr(obj_box, 'tolist') else list(obj_box),
        "img_path": img_path,
    })


# ---------------------------------------------------------------------------
# API 12 — train_main (admin)
# ---------------------------------------------------------------------------

async def api_train_main(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny

    # Spawn the existing train_textemb.py as a subprocess so the FastAPI loop
    # stays responsive. Live logs are streamed to the requester via WebSocket.
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "train_textemb.py")]
    for flag in ("epochs", "batch_size", "lr"):
        if flag in payload:
            cmd += [f"--{flag}", str(payload[flag])]
    if "save_name" in payload:
        cmd += ["--save_name", str(payload["save_name"]) + "_textemb"]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )

    async def pump():
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            await _send(payload.get("uuid", ""), {
                "type": "train_log",
                "payload": {"stream": "main", "line": line.decode(errors='ignore')},
            })
        await proc.wait()
        await _send(payload.get("uuid", ""), {
            "type": "train_done",
            "payload": {"stream": "main", "returncode": proc.returncode},
        })

    asyncio.create_task(pump())
    return _ok({"pid": proc.pid, "cmd": " ".join(cmd)})


# ---------------------------------------------------------------------------
# API 13 — train_annotation (admin)
# ---------------------------------------------------------------------------

async def api_train_annotation(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny

    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "annotation",
                                        "train_annotation.py")]
    for flag in ("epochs", "batch_size", "lr", "scheme"):
        if flag in payload:
            cmd += [f"--{flag}", str(payload[flag])]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )

    async def pump():
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            await _send(payload.get("uuid", ""), {
                "type": "train_log",
                "payload": {"stream": "annotation",
                            "line": line.decode(errors='ignore')},
            })
        await proc.wait()
        await _send(payload.get("uuid", ""), {
            "type": "train_done",
            "payload": {"stream": "annotation", "returncode": proc.returncode},
        })

    asyncio.create_task(pump())
    return _ok({"pid": proc.pid, "cmd": " ".join(cmd)})


# ---------------------------------------------------------------------------
# API 14 — list_memory_cache (admin)
# ---------------------------------------------------------------------------

async def api_list_memory_cache(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """List pending entries in memory_cache_push/ directory.

    Returns two lists:
      - img_entries:  npz files prefixed with 'img_'
      - pref_entries: npz files prefixed with 'pref_'

    Each entry includes filename, timestamp, object/affordance labels, and
    for image entries a base64 thumbnail for preview.
    """
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny

    img_entries = []
    pref_entries = []

    try:
        files = sorted(os.listdir(MEMORY_CACHE_PUSH_DIR))
    except OSError:
        files = []

    for fname in files:
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(MEMORY_CACHE_PUSH_DIR, fname)
        try:
            data = np.load(fpath, allow_pickle=True)
            ts_ms = int(fname.split("_")[1].split(".")[0]) if "_" in fname else 0

            def _npz_str(key: str) -> str:
                try:
                    v = data[key]
                    return v.item() if v.ndim == 0 else str(v)
                except Exception:
                    return ""

            entry = {
                "filename": fname,
                "timestamp": ts_ms,
                "object": _npz_str("object"),
                "affordance": _npz_str("affordance"),
            }

            if fname.startswith("img_"):
                # Decode stored image bytes for thumbnail
                try:
                    img_bytes = data["img_bytes"]
                    img = Image.open(io.BytesIO(img_bytes.tobytes()))
                    img.thumbnail((120, 90))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    entry["thumbnail"] = ("data:image/png;base64," +
                                          base64.b64encode(buf.getvalue()).decode())
                except Exception:
                    entry["thumbnail"] = ""
                img_entries.append(entry)
            elif fname.startswith("pref_"):
                entry["outcome"] = _npz_str("outcome")
                pref_entries.append(entry)
        except Exception:
            continue

    return _ok({"img_entries": img_entries, "pref_entries": pref_entries})


# ---------------------------------------------------------------------------
# API 15 — push_to_memory (admin)
# ---------------------------------------------------------------------------

async def api_push_to_memory(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """Upload selected memory_cache_push/ entries to the active memory stores.

    ``payload.filenames`` is a list of .npz filenames (without directory path).
    ``payload.delete_after`` (bool, default True) removes the .npz after upload.

    For img_* entries: uploads to STATE.image_memory (requires poweron to have
    loaded an image memory store).
    For pref_* entries: uploads to STATE.pref_memory (requires poweron to have
    loaded a preference memory store).
    """
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny

    filenames = payload.get("filenames", [])
    delete_after = payload.get("delete_after", True)
    results = []

    for fname in filenames:
        fpath = os.path.join(MEMORY_CACHE_PUSH_DIR, fname)
        if not os.path.exists(fpath):
            results.append({"filename": fname, "ok": False, "error": "file not found"})
            continue

        try:
            data = np.load(fpath, allow_pickle=True)
            def _str(key):
                try:
                    v = data[key]
                    return v.item() if v.ndim == 0 else str(v)
                except (KeyError, Exception):
                    return ""

            obj_cat = _str("object")
            affordance = _str("affordance")

            if fname.startswith("img_"):
                if STATE.image_memory is None:
                    results.append({"filename": fname, "ok": False,
                                    "error": "image_memory not loaded"})
                    continue

                img_bytes = data["img_bytes"].tobytes()
                img_np = np.array(Image.open(io.BytesIO(img_bytes)))
                sub_box = data["sub_box"] if "sub_box" in data else None
                obj_box = data["obj_box"] if "obj_box" in data else None
                F_i = data["F_i"] if "F_i" in data else None
                F_s = data["F_s"] if "F_s" in data else None
                F_e = data["F_e"] if "F_e" in data else None

                # Use a zero feature for the image feature index (retrieval by key)
                dummy_feat = np.zeros(STATE.image_memory.store.feature_dim,
                                      dtype=np.float32)
                loop = asyncio.get_event_loop()
                entry_id = await loop.run_in_executor(None,
                    lambda: STATE.image_memory.store_image(
                        image=img_np,
                        image_feature=dummy_feat,
                        object_category=obj_cat,
                        affordance_label=affordance,
                        sub_box=sub_box,
                        obj_box=obj_box,
                        confidence=1.0,
                        F_i=F_i,
                        F_s=F_s,
                        F_e=F_e,
                    ))
                results.append({"filename": fname, "ok": True,
                                 "entry_id": entry_id, "type": "image"})

            elif fname.startswith("pref_"):
                if STATE.pref_memory is None:
                    results.append({"filename": fname, "ok": False,
                                    "error": "pref_memory not loaded"})
                    continue

                arm_feat = torch.from_numpy(data["arm_feature"].astype(np.float32))  # CPU: form_memory runs on CPU
                # New format: N_p-space layout
                l3_xyz  = data["l3_xyz"].astype(np.float32)    # [N_p, 3]
                l3_feat = data["l3_features"].astype(np.float32)  # [N_p, C]
                pref_np = data["pref_at_np"].astype(np.float32)   # [N_p]
                outcome = _str("outcome")
                reward = {"优秀": 1.0, "成功": 0.7, "失败": -0.5}.get(outcome, 0.5)

                loop = asyncio.get_event_loop()
                entry_id = await loop.run_in_executor(None,
                    lambda: STATE.pref_memory.form_memory(
                        arm_feature=arm_feat,
                        point_cloud=l3_xyz,      # [N_p, 3]
                        point_features=l3_feat,  # [N_p, C]
                        preference_matrix=pref_np,  # [N_p]
                        reward=reward,
                        outcome=outcome,
                        object_category=obj_cat,
                        affordance_label=affordance,
                        confidence=1.0,
                    ))
                results.append({"filename": fname, "ok": True,
                                 "entry_id": entry_id, "type": "pref"})

            else:
                results.append({"filename": fname, "ok": False,
                                 "error": "unknown entry type"})
                continue

            if delete_after and results[-1]["ok"]:
                try:
                    os.remove(fpath)
                except OSError:
                    pass

        except Exception as e:
            results.append({"filename": fname, "ok": False, "error": str(e)})

    return _ok({"results": results})


# ---------------------------------------------------------------------------
# API 16 — toggle_pref_memory (user)
# ---------------------------------------------------------------------------

async def api_toggle_pref_memory(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """Toggle whether preference memory is used for enhanced localization.

    ``payload.enabled`` (bool): True to enable, False to disable.
    Any user (not just admin) can call this.
    """
    enabled = payload.get("enabled")
    if enabled is None:
        return _err("'enabled' (bool) required")
    STATE.use_pref_memory = bool(enabled)
    return _ok({"use_pref_memory": STATE.use_pref_memory})


# ---------------------------------------------------------------------------
# API 17 / 18 / 19 — pref memory browse / get / delete (admin)
# ---------------------------------------------------------------------------

async def api_list_pref_memory(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """List entries in the active preference (point-cloud) memory store.

    payload.page (int, default 1), payload.per_page (int, default 20).
    """
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny
    if STATE.pref_memory is None:
        return _err("pref_memory not loaded")
    page = int(payload.get("page", 1) or 1)
    per_page = int(payload.get("per_page", 20) or 20)
    try:
        listing = STATE.pref_memory.store.list_all(page=page, per_page=per_page)
        return _ok(listing)
    except Exception as e:
        traceback.print_exc()
        return _err(f"list_pref_memory failed: {e}")


async def api_get_pref_memory_entry(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """Return the full data of a single pref memory entry for visualisation.

    payload.entry_id (str).
    Response contains base64-encoded point_cloud [N,3] and preference_matrix [N].
    """
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny
    if STATE.pref_memory is None:
        return _err("pref_memory not loaded")
    entry_id = payload.get("entry_id")
    if not entry_id:
        return _err("entry_id required")

    entry = STATE.pref_memory.store.get(entry_id)
    if entry is None:
        return _err("entry not found")

    pc = np.asarray(entry.point_cloud, dtype=np.float32)
    if pc.ndim == 1 and pc.size % 3 == 0:
        pc = pc.reshape(-1, 3)
    pref = np.asarray(entry.preference_matrix, dtype=np.float32).flatten()

    # Scene image (optional)
    scene_b64 = ""
    if entry.scene_image is not None and entry.scene_image.size > 0:
        try:
            si = entry.scene_image
            if si.ndim == 1:
                # try to infer square RGB
                n = si.size
                side = int(round((n / 3) ** 0.5))
                if side * side * 3 == n:
                    si = si.reshape(side, side, 3)
            scene_b64 = _encode_image_b64(si)
        except Exception:
            scene_b64 = ""

    return _ok({
        "entry_id": entry.id,
        "object_category": entry.object_category,
        "affordance_label": entry.affordance_label,
        "outcome": entry.outcome,
        "reward": float(entry.reward),
        "confidence": float(entry.confidence),
        "timestamp": float(entry.timestamp),
        "access_count": int(entry.access_count),
        "point_cloud_b64": base64.b64encode(pc.tobytes()).decode("ascii"),
        "n_points": int(pc.shape[0]),
        "preference_b64": base64.b64encode(pref.tobytes()).decode("ascii"),
        "scene_image": scene_b64,
    })


async def api_delete_pref_memory(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """Delete one pref memory entry by id."""
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny
    if STATE.pref_memory is None:
        return _err("pref_memory not loaded")
    entry_id = payload.get("entry_id")
    if not entry_id:
        return _err("entry_id required")
    try:
        STATE.pref_memory.store.remove(entry_id)
        return _ok({"entry_id": entry_id, "deleted": True})
    except Exception as e:
        traceback.print_exc()
        return _err(f"delete failed: {e}")


# ---------------------------------------------------------------------------
# API 20 / 21 / 22 — image memory browse / get / delete (admin)
# ---------------------------------------------------------------------------

async def api_list_image_memory(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """List entries in the image memory store (paginated)."""
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny
    if STATE.image_memory is None:
        return _err("image_memory not loaded")
    page = int(payload.get("page", 1) or 1)
    per_page = int(payload.get("per_page", 20) or 20)
    try:
        listing = STATE.image_memory.store.list_all(page=page, per_page=per_page)
        return _ok(listing)
    except Exception as e:
        traceback.print_exc()
        return _err(f"list_image_memory failed: {e}")


async def api_get_image_memory_entry(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """Return one image memory entry rendered as base64 PNG + boxes."""
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny
    if STATE.image_memory is None:
        return _err("image_memory not loaded")
    entry_id = payload.get("entry_id")
    if not entry_id:
        return _err("entry_id required")

    store = STATE.image_memory.store
    row = store._db_get_by_id(entry_id)
    if row is None:
        return _err("entry not found")

    # Load image from disk
    img_b64 = ""
    img_w = 0
    img_h = 0
    try:
        img_path = row.get("image_path", "")
        if img_path and os.path.exists(img_path):
            arr = np.load(img_path, allow_pickle=True)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)
            img_h, img_w = arr.shape[:2]
            img_b64 = _encode_image_b64(arr)
    except Exception:
        traceback.print_exc()
        img_b64 = ""

    sub_box = store._decode_box(row.get("sub_box") or b"")
    obj_box = store._decode_box(row.get("obj_box") or b"")

    return _ok({
        "entry_id": row["id"],
        "object_category": row["object_category"],
        "affordance_label": row["affordance_label"],
        "confidence": row["confidence"],
        "timestamp": row["timestamp"],
        "access_count": row["access_count"],
        "image": img_b64,
        "image_w": img_w,
        "image_h": img_h,
        "sub_box": sub_box.tolist() if sub_box is not None else None,
        "obj_box": obj_box.tolist() if obj_box is not None else None,
    })


async def api_delete_image_memory(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """Delete one image memory entry by id."""
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny
    if STATE.image_memory is None:
        return _err("image_memory not loaded")
    entry_id = payload.get("entry_id")
    if not entry_id:
        return _err("entry_id required")
    try:
        STATE.image_memory.store.remove(entry_id)
        return _ok({"entry_id": entry_id, "deleted": True})
    except Exception as e:
        traceback.print_exc()
        return _err(f"delete failed: {e}")


# ---------------------------------------------------------------------------
# API 23 — train_index_align (admin)
# ---------------------------------------------------------------------------

async def api_train_index_align(payload: Dict[str, Any], conn_id: str) -> Dict[str, Any]:
    """Train MemoryIndexer + MemoryAligner on the populated pref memory store.

    Accepted optional fields in payload:
        epochs, batch_size, lr, max_points, store_dir, out_dir.
    Falls back to STATE.pref_memory.store.store_dir for store_dir.
    """
    deny = _require_admin(payload.get("uuid", ""))
    if deny:
        return deny

    if STATE.pref_memory is None and not payload.get("store_dir"):
        return _err("pref_memory not loaded and no store_dir provided")

    store_dir = payload.get("store_dir") or STATE.pref_memory.store.store_dir
    out_dir = payload.get("out_dir") or store_dir

    cmd = [sys.executable, "-m", "memory_system.train_index_align",
           "--store_dir", store_dir,
           "--out_dir", out_dir]
    for flag in ("epochs", "batch_size", "lr", "max_points"):
        if flag in payload:
            cmd += [f"--{flag}", str(payload[flag])]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )

    async def pump():
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            await _send(payload.get("uuid", ""), {
                "type": "train_log",
                "payload": {"stream": "index_align",
                            "line": line.decode(errors='ignore')},
            })
        await proc.wait()
        await _send(payload.get("uuid", ""), {
            "type": "train_done",
            "payload": {"stream": "index_align",
                        "returncode": proc.returncode,
                        "out_dir": out_dir},
        })

    asyncio.create_task(pump())
    return _ok({"pid": proc.pid, "cmd": " ".join(cmd),
                "out_dir": out_dir})


# ---------------------------------------------------------------------------
# WebSocket router
# ---------------------------------------------------------------------------

API_TABLE = {
    "register_robot":      api_register_robot,
    "register_user":       api_register_user,
    "watch_robot":         api_watch_robot,
    "register_admin":      api_register_admin,
    "poweron":             api_poweron,
    "infer_from_robot":    api_infer_from_robot,
    "infer_from_user_img": api_infer_from_user_img,
    "infer_user_pref":     api_infer_user_pref,
    "feedback":            api_feedback,
    "annotate":            api_annotate,
    "dataset_sample":      api_dataset_sample,
    "train_main":          api_train_main,
    "train_annotation":    api_train_annotation,
    "list_memory_cache":   api_list_memory_cache,
    "push_to_memory":      api_push_to_memory,
    "toggle_pref_memory":  api_toggle_pref_memory,
    "list_pref_memory":    api_list_pref_memory,
    "get_pref_memory_entry": api_get_pref_memory_entry,
    "delete_pref_memory":  api_delete_pref_memory,
    "list_image_memory":   api_list_image_memory,
    "get_image_memory_entry": api_get_image_memory_entry,
    "delete_image_memory": api_delete_image_memory,
    "train_index_align":   api_train_index_align,
    "train_main":          api_train_main,
    "train_annotation":    api_train_annotation,
}


app = FastAPI(title="Affordance System Backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_loaded": STATE.main_model is not None,
        "robots": STATE.robot_ids,
        "users": len(STATE.user_uuids),
        "admins": len(STATE.admin_uuids),
    }


@app.get("/dataset/load")
async def http_dataset_load(setting: str = "Seen"):
    """Strictly separated REST endpoint for the robot frontend's
    'load dataset' button — does NOT open a WebSocket."""
    if setting not in ("Seen", "Unseen"):
        return {"ok": False, "error": "setting must be Seen or Unseen"}
    try:
        ds = _ensure_dataset(setting)
        return {"ok": True, "data": {"setting": setting, "size": len(ds)}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/dataset/sample")
async def http_dataset_sample(setting: str = "Seen",
                              index: Optional[int] = None):
    """REST counterpart of API 11 — used by the robot frontend after the
    operator has confirmed dataset loading via /dataset/load."""
    payload = {"setting": setting}
    if index is not None:
        payload["index"] = index
    return await api_dataset_sample(payload, conn_id="http")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    conn_id = uuid.uuid4().hex
    STATE.sockets[conn_id] = ws
    STATE.socket_role[conn_id] = "anon"
    try:
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps(_err("invalid JSON")))
                continue
            api_name = msg.get("type")
            request_id = msg.get("request_id") or uuid.uuid4().hex
            payload = msg.get("payload", {}) or {}
            handler = API_TABLE.get(api_name)
            if handler is None:
                await ws.send_text(json.dumps({
                    "type": api_name, "request_id": request_id,
                    **_err(f"unknown api: {api_name}"),
                }))
                continue

            # Re-key the socket if this call carries a uuid / robot_id so
            # subsequent notifications can be routed.
            for key in ("uuid", "robot_id"):
                v = payload.get(key)
                if v and v not in STATE.sockets:
                    STATE.sockets[v] = ws

            try:
                result = await handler(payload, conn_id)
            except Exception as e:
                traceback.print_exc()
                result = _err(f"{type(e).__name__}: {e}")
            await ws.send_text(json.dumps({
                "type": api_name, "request_id": request_id, **result,
            }))
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up any socket entries pointing at this ws
        for k in list(STATE.sockets.keys()):
            if STATE.sockets[k] is ws:
                STATE.sockets.pop(k, None)
                STATE.socket_role.pop(k, None)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--password", required=True,
                   help="admin password for API 4 / 5 / 12 / 13")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--map_json", default=os.path.join(
        PROJECT_ROOT, "model-memory-word-map.json"))
    return p.parse_args()


def main():
    args = parse_args()
    global STATE
    STATE = ServerState(password=args.password, map_json_path=args.map_json)

    if not os.path.exists(args.map_json):
        print(f"[warn] {args.map_json} not found — admin must POST a spec via "
              f"the poweron API before inference works.")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()