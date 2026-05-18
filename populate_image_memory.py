"""
populate_image_memory.py

从 PIAD 数据集（Seen/Unseen，Train/Test 分割）提取图像特征，
使用 IAG_TextEmb 模型的 get_img_and_feature 方法提取 F_i/F_s/F_e，
将结果存入 ImageMemoryManager，每个 (object, affordance) 键最多保存 5 条记录。

文本嵌入方式参考 backend.py 的 _load_word_yaml / _load_emb 方法。

用法:
    python populate_image_memory.py
    python populate_image_memory.py --setting Unseen --split Test
    python populate_image_memory.py --max_samples 500
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.MyNet import IAG_TextEmb
from data_utils.dataset import PIAD
from memory_system.image_memory_manager import ImageMemoryManager

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]

GLOVE_PATH = os.path.join(PROJECT_ROOT, "glove", "glove.6B.300d.txt")
WORD_PATH  = os.path.join(PROJECT_ROOT, "word_dict", "default_seen.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# GloVe / 词向量工具（与 backend.py 一致）
# ─────────────────────────────────────────────────────────────────────────────

def _load_word_yaml(word_path: str) -> Dict[str, Any]:
    import yaml
    if not os.path.exists(word_path):
        return {}
    with open(word_path, 'r') as f:
        return yaml.safe_load(f) or {}


def _load_emb(path: str, word_yaml: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """从 GloVe 文件中加载目标词的向量（与 backend.py 中逻辑一致）。"""
    embeddings: Dict[str, np.ndarray] = {}
    af_list  = word_yaml.get("affordance_labels", []) if word_yaml else []
    obj_list = word_yaml.get("object_labels", [])     if word_yaml else []
    sem_dic  = word_yaml.get("word_map", {})           if word_yaml else {}

    if not os.path.exists(path):
        print(f"[GloVe] WARNING: file not found: {path}  — using zero vectors")
        return embeddings

    print(f"[GloVe] Loading embeddings from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            if word not in af_list and word not in obj_list and [word] not in sem_dic.values():
                continue
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if [word] in sem_dic.values():
                for key, val in sem_dic.items():
                    if val == [word]:
                        embeddings[key] = vector
            else:
                embeddings[word] = vector

    print(f"[GloVe] Loaded {len(embeddings)} word vectors")
    return embeddings


def get_text_emb(affordance: str, emb_dict: Dict[str, np.ndarray],
                 text_dim: int = 300, device: torch.device = None) -> torch.Tensor:
    """将 affordance 字符串转为文本嵌入张量 [1, text_dim]。"""
    vec = emb_dict.get(affordance, None)
    if vec is None:
        vec = np.zeros(text_dim, dtype=np.float32)
    t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def populate_image_memory(
    model_path:         str  = "./model_list/iag_textemb_seen.pt",
    data_root:          str  = "./Data",
    setting:            str  = "Seen",
    split:              str  = "Train",
    store_dir:          str  = "./image_memory_store",
    device_str:         str  = "cuda:0",
    max_samples:        Optional[int] = None,
    max_images_per_key: int  = 5,
    use_faiss:          bool = True,
    num_workers:        int  = 0,
    glove_path:         str  = GLOVE_PATH,
    word_path:          str  = WORD_PATH,
):
    print("=" * 60)
    print("Image Memory Population")
    print(f"  Setting : {setting}  |  Split : {split}")
    print(f"  Store   : {store_dir}")
    print("=" * 60)

    # ── 设备 ─────────────────────────────────────────────────────────────────
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}\n")

    # ── 1. GloVe 文本嵌入 ─────────────────────────────────────────────────────
    print("[1/4] Loading GloVe word embeddings")
    word_yaml = _load_word_yaml(word_path)
    emb_dict  = _load_emb(glove_path, word_yaml)

    # ── 2. 加载模型 ───────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = IAG_TextEmb(
        img_model_path=None,
        pre_train=False,
        normal_channel=False,
        N_p=64,
        emb_dim=512,
        proj_dim=512,
        num_heads=4,
        N_raw=2048,
        num_affordance=17,
        text_dim=300,
    )
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✓ Model loaded")

    # ── 3. 加载数据集 ─────────────────────────────────────────────────────────
    print(f"\n[3/4] Loading PIAD dataset ({setting}, {split})")
    data_path  = os.path.join(data_root, setting)
    point_txt  = os.path.join(data_path, f"Point_{split}.txt")
    img_txt    = os.path.join(data_path, f"Img_{split}.txt")
    box_txt    = os.path.join(data_path, f"Box_{split}.txt")

    for fpath in [point_txt, img_txt, box_txt]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")

    run_type = 'train' if split == 'Train' else 'val'
    dataset = PIAD(
        run_type=run_type,
        setting_type=setting,
        point_path=point_txt,
        img_path=img_txt,
        box_path=box_txt,
    )
    print(f"✓ Dataset size: {len(dataset)} samples  (run_type='{run_type}')")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # ── 4. 初始化 ImageMemoryManager ──────────────────────────────────────────
    print(f"\n[4/4] Initializing ImageMemoryManager at {store_dir}")
    os.makedirs(store_dir, exist_ok=True)
    manager = ImageMemoryManager(
        store_dir=store_dir,
        feature_dim=512,
        use_faiss=use_faiss,
        max_images_per_key=max_images_per_key,
        averaging_strategy="mean",
    )
    print(f"✓ max_images_per_key = {max_images_per_key}")

    # ── 5. 遍历数据集 ─────────────────────────────────────────────────────────
    print("\nProcessing samples...")
    print("-" * 60)

    stored_count = 0
    skip_count   = 0

    for idx, sample in enumerate(tqdm(loader, desc="Extracting features")):
        if max_samples is not None and stored_count >= max_samples:
            break

        try:
            if run_type == 'train':
                # train 返回: (Img, Points_List, affordance_label_List,
                #              affordance_index_List, sub_box, obj_box)
                img              = sample[0]            # [1, 3, H, W]
                points_list      = sample[1]
                labels_list      = sample[2]
                logits_labels    = sample[3]            # list of affordance indices
                sub_box          = sample[4]            # [1, 4]
                obj_box          = sample[5]            # [1, 4]

                # 从 img_files 解析物体类别和 affordance
                img_path_str     = dataset.img_files[idx]
                parts            = os.path.splitext(os.path.basename(img_path_str))[0].split('_')
                object_category  = parts[-3] if len(parts) >= 3 else "unknown"

                # affordance index 来自第一个配对样本（batch_size=1 时只有1对）
                aff_idx_tensor   = logits_labels[0]    # Tensor scalar
                aff_idx          = int(aff_idx_tensor.item()) if hasattr(aff_idx_tensor, 'item') else int(aff_idx_tensor)
                affordance_name  = AFFORDANCE_LABELS[aff_idx] if aff_idx < len(AFFORDANCE_LABELS) else "unknown"

                # 构造 pair_num=1 的扩展批次（与 train 中保持一致）
                pair_num         = 1
                img_expand       = img.repeat_interleave(pair_num, dim=0).to(device).float()
                sub_box_expand   = sub_box.repeat_interleave(pair_num, dim=0).to(device).float()
                obj_box_expand   = obj_box.repeat_interleave(pair_num, dim=0).to(device).float()

            else:
                # val 返回: (Img, Point, affordance_label, img_path,
                #            point_path, sub_box, obj_box)
                img              = sample[0]            # [1, 3, H, W]
                img_path_str     = sample[3][0]
                sub_box          = sample[5]
                obj_box          = sample[6]

                parts            = os.path.splitext(os.path.basename(img_path_str))[0].split('_')
                object_category  = parts[-3] if len(parts) >= 3 else "unknown"
                affordance_name  = parts[-2] if len(parts) >= 2 else "unknown"
                if affordance_name not in AFFORDANCE_LABELS:
                    skip_count += 1
                    continue

                img_expand       = img.to(device).float()
                sub_box_expand   = sub_box.to(device).float()
                obj_box_expand   = obj_box.to(device).float()

            # ── 提取特征 ───────────────────────────────────────────────────
            with torch.no_grad():
                F_i, F_s, F_e = model.get_img_and_feature(
                    img_expand, sub_box_expand, obj_box_expand
                )
                F_I = model.img_encoder(img_expand)

            # 转为 numpy（取第 0 个 batch 元素）
            img_np  = img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, 3]
            F_I_np  = F_I.squeeze(0).cpu().detach().numpy()                     # [C, h, w]
            F_i_np  = F_i.squeeze(0).cpu().detach().numpy()
            F_s_np  = F_s.squeeze(0).cpu().detach().numpy()
            F_e_np  = F_e.squeeze(0).cpu().detach().numpy()

            sub_box_np = sub_box_expand.squeeze(0).cpu().detach().numpy()       # [4]
            obj_box_np = obj_box_expand.squeeze(0).cpu().detach().numpy()       # [4]

            # ── 存入记忆库 ─────────────────────────────────────────────────
            manager.store_image(
                image=img_np,
                image_feature=F_I_np,
                object_category=object_category,
                affordance_label=affordance_name,
                sub_box=sub_box_np,
                obj_box=obj_box_np,
                confidence=1.0,
                F_i=F_i_np,
                F_s=F_s_np,
                F_e=F_e_np,
            )
            stored_count += 1

            if stored_count % 100 == 0:
                print(f"  Stored {stored_count} images  (skipped {skip_count})")

        except Exception as e:
            skip_count += 1
            if skip_count <= 5:
                import traceback
                print(f"\n  [Warning] Sample {idx} failed: {e}")
                traceback.print_exc()
            continue

    # ── 保存 ─────────────────────────────────────────────────────────────────
    manager.save()

    print("\n" + "=" * 60)
    print(f"Done.  Stored: {stored_count}  |  Skipped: {skip_count}")
    stats = manager.get_stats()
    print(f"Total images in store : {stats['total_images']}")
    print(f"Unique (object, aff.) keys : {len(stats['categories'])}")
    print("=" * 60)
    return manager


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Populate image memory from PIAD using IAG_TextEmb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path",         type=str, default="./model_list/iag_textemb_seen.pt")
    parser.add_argument("--data_root",          type=str, default="./Data")
    parser.add_argument("--setting",            type=str, default="Seen", choices=["Seen", "Unseen"])
    parser.add_argument("--split",              type=str, default="Train", choices=["Train", "Test"])
    parser.add_argument("--store_dir",          type=str, default="./image_memory_store")
    parser.add_argument("--device",             type=str, default="cuda:0")
    parser.add_argument("--max_samples",        type=int, default=None)
    parser.add_argument("--max_images_per_key", type=int, default=5)
    parser.add_argument("--no_faiss",           action="store_true")
    parser.add_argument("--num_workers",        type=int, default=0)
    parser.add_argument("--glove_path",         type=str, default=GLOVE_PATH)
    parser.add_argument("--word_path",          type=str, default=WORD_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    populate_image_memory(
        model_path=args.model_path,
        data_root=args.data_root,
        setting=args.setting,
        split=args.split,
        store_dir=args.store_dir,
        device_str=args.device,
        max_samples=args.max_samples,
        max_images_per_key=args.max_images_per_key,
        use_faiss=not args.no_faiss,
        num_workers=args.num_workers,
        glove_path=args.glove_path,
        word_path=args.word_path,
    )
