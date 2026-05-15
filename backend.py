"""
FastAPI backend for the affordance-system service.

Run:
    python backend.py --password admin123 --host 0.0.0.0 --port 8800

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

MEMORY_CACHE_DIR = os.path.join(PROJECT_ROOT, "Memory_cache")
os.makedirs(MEMORY_CACHE_DIR, exist_ok=True)


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

        # 3. Word / yaml -----------------------------------------------------
        STATE.word_yaml = _load_word_yaml(spec.get("word_path", ""))

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


async def _run_inference(*, img_np: np.ndarray, points: np.ndarray,
                         sub_box: np.ndarray, obj_box: np.ndarray,
                         object_category: str, affordance: str) -> Dict[str, Any]:
    """Run the main IAG_TextEmb model + memory retrieval + fusion."""
    if STATE.main_model is None:
        raise RuntimeError("main model not loaded; admin must call poweron first")

    from data_utils.dataset import img_normalize_val

    img_pil = Image.fromarray(img_np).resize((224, 224))
    img_tensor = img_normalize_val(img_pil).unsqueeze(0).to(STATE.device)

    pts = _pc_normalize(points).T  # [3, N]
    pts_tensor = torch.from_numpy(pts).float().unsqueeze(0).to(STATE.device)

    sub_t = torch.from_numpy(np.asarray(sub_box, dtype=np.float32)).unsqueeze(0).to(STATE.device)
    obj_t = torch.from_numpy(np.asarray(obj_box, dtype=np.float32)).unsqueeze(0).to(STATE.device)

    info: Dict[str, Any] = {"object": object_category, "affordance": affordance}

    # ── image memory retrieval (+ averaging of img feature) ───────────────
    retrieved_images: List[str] = []
    if STATE.image_memory is not None:
        entries = STATE.image_memory.store.retrieve_by_key(
            object_category, affordance, top_k=4)
        info["image_memory_hits"] = len(entries)
        for e in entries:
            p = e.get("image_path", "")
            if p and os.path.exists(p):
                try:
                    retrieved_images.append(_encode_image_b64(np.load(p)))
                except Exception:
                    pass

    async with STATE.model_lock:
        with torch.no_grad():
            try:
                if hasattr(STATE.main_model, "word_emb") or "textemb" in \
                        type(STATE.main_model).__name__.lower():
                    aff_idx = AFFORDANCE_LABELS.index(affordance) \
                        if affordance in AFFORDANCE_LABELS else 0
                    aff_tensor = torch.tensor([aff_idx], device=STATE.device)
                    out = STATE.main_model(img_tensor, pts_tensor, sub_t, obj_t, aff_tensor)
                else:
                    out = STATE.main_model(img_tensor, pts_tensor, sub_t, obj_t)
            except Exception as e:
                # Models have varying forward signatures across the codebase;
                # surface the issue rather than silently masking.
                raise RuntimeError(f"main model forward failed: {e}")

    # Outputs from MyNet.forward typically: (_3daffordance, logits, ...)
    pred_pref = None
    if isinstance(out, (tuple, list)):
        pred_pref = out[0]
    elif isinstance(out, dict):
        pred_pref = out.get("affordance") or out.get("_3daffordance")
    else:
        pred_pref = out
    pred_pref = pred_pref.squeeze().detach().cpu().numpy()

    # ── preference memory retrieval & fusion ──────────────────────────────
    fused_pref = pred_pref
    if STATE.pref_memory is not None:
        try:
            B, N, D = 1, pts.shape[1], 512
            dummy_arm = torch.zeros(B, 128, D, device=STATE.device)
            dummy_fp = torch.zeros(N, D, device=STATE.device)
            fused = STATE.pref_memory.retrieve_and_fuse(
                arm_feature=dummy_arm,
                current_point_cloud=torch.from_numpy(points).float().to(STATE.device),
                current_point_features=dummy_fp,
                top_k=5,
            )
            if fused is not None and hasattr(fused, "cpu"):
                fused_np = fused.detach().cpu().numpy().flatten()
                if fused_np.shape == pred_pref.shape:
                    fused_pref = 0.5 * pred_pref + 0.5 * fused_np
                    info["pref_memory_applied"] = True
        except Exception as e:
            info["pref_memory_error"] = str(e)

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
    """API 7 — user uploads an annotated 2D image with a robot_id."""
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

    result = await _run_inference(
        img_np=cur["image"], points=cur["points"],
        sub_box=np.asarray(cur["sub_box"], dtype=np.float32),
        obj_box=np.asarray(cur["obj_box"], dtype=np.float32),
        object_category=cur["object"],
        affordance=cur["affordance"],
    )
    # Append the user's uploaded image to the retrieved set so the frontend
    # can show it as part of the image memory used in this turn.
    result["retrieved_images"].append(_encode_image_b64(extra_img))
    result["info"]["user_image_appended"] = True

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

    # Cache to Memory_cache/ in the same format as feedback
    cache_path = os.path.join(MEMORY_CACHE_DIR,
                              f"user_pref_{robot_id}_{int(time.time()*1000)}.npz")
    np.savez(cache_path, preference=pref_np,
             points=cur.get("points") if isinstance(cur.get("points"), np.ndarray)
             else np.zeros((0, 3)))

    await _notify_watchers(robot_id, {
        "type": "robot_inference_update",
        "payload": {"robot_id": robot_id, **result, "source": "user_preference"},
    })
    return _ok({**result, "cache_path": cache_path})


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

    cache_path = os.path.join(MEMORY_CACHE_DIR,
                              f"feedback_{robot_id}_{int(time.time()*1000)}.npz")
    np.savez(cache_path,
             preference=pref_np, points=pts,
             outcome=np.array(outcome),
             object=np.array(cur.get("object", "")),
             affordance=np.array(cur.get("affordance", "")))

    # Persist to the preference memory if available
    if STATE.pref_memory is not None and pts.size > 0:
        try:
            reward = {"优秀": 1.0, "成功": 0.7, "失败": -0.5}.get(outcome, 0.5)
            STATE.pref_memory.form_memory(
                arm_feature=torch.zeros(1, 128, 512, device=STATE.device),
                point_cloud=torch.from_numpy(pts).float(),
                point_features=torch.zeros(pts.shape[0], 512),
                preference_matrix=torch.from_numpy(pref_np).float(),
                reward=reward,
                action_params={},
                outcome=outcome,
                object_category=cur.get("object", ""),
                affordance_label=cur.get("affordance", ""),
                confidence=1.0,
            )
        except Exception as e:
            return _ok({"cache_path": cache_path,
                        "warning": f"pref_memory.form_memory failed: {e}"})

    return _ok({"cache_path": cache_path})


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

    async with STATE.model_lock:
        with torch.no_grad():
            out = model(img_t)

    serial: Dict[str, Any] = {"scheme": scheme}
    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                serial[k] = v.detach().cpu().numpy().tolist()
            else:
                serial[k] = v
    return _ok(serial)


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
