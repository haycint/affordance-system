"""
prepopulate_image_memory.py

从 PIAD 数据集（Seen + Unseen）加载样本，使用 IAG_TextEmb 模型的
get_img_and_feature 方法提取图像特征（F_i, F_s, F_e），
并将结果存入 ImageMemoryManager。

每个 (object_category, affordance_label) 键最多保存 4 条记录。

用法:
    python prepopulate_image_memory.py
    python prepopulate_image_memory.py --model_path ./model_list/iag_textemb_seen.pt
    python prepopulate_image_memory.py --setting Unseen --max_samples 200
"""

from __future__ import annotations

import os
import sys
import argparse
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
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

GLOVE_DIR = os.path.join(PROJECT_ROOT, 'glove')
GLOVE_6B_300D_FILE = "glove.6B.300d.txt"
GLOVE_URLS = [
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://mirrors.tuna.tsinghua.edu.cn/stanford-nlp/data/glove.6B.zip",
    "https://hf-mirror.com/stanfordnlp/glove/resolve/main/glove.6B.zip",
]

# ─────────────────────────────────────────────────────────────────────────────
# GloVe 工具函数（与 train_textemb.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────

def download_glove(force: bool = False) -> str:
    os.makedirs(GLOVE_DIR, exist_ok=True)
    glove_300d_path = os.path.join(GLOVE_DIR, GLOVE_6B_300D_FILE)

    if os.path.exists(glove_300d_path) and not force:
        print(f"[GloVe] Found: {glove_300d_path}")
        return glove_300d_path

    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
    if not os.path.exists(zip_path) or force:
        import urllib.request
        last_err = None
        for url in GLOVE_URLS:
            try:
                print(f"[GloVe] Downloading from {url} ...")
                urllib.request.urlretrieve(url, zip_path)
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[GloVe] Failed: {e}")
        if last_err is not None:
            print("[GloVe] All mirrors failed. Please download manually.")
            sys.exit(1)

    if not os.path.exists(glove_300d_path):
        print(f"[GloVe] Extracting {GLOVE_6B_300D_FILE} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(GLOVE_6B_300D_FILE, GLOVE_DIR)

    return glove_300d_path


def load_glove_embeddings(glove_path: str, target_words: list) -> dict:
    target_set = set(target_words)
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            if word in target_set:
                embeddings[word] = np.array([float(x) for x in parts[1:]])
    return embeddings


def _split_compound_word(word: str) -> list:
    compound_map = {'wrapgrasp': ['wrap', 'grasp']}
    if word in compound_map:
        return compound_map[word]
    return [word]


def build_affordance_embeddings(glove_embeddings: dict, labels: list):
    text_dim = next(iter(glove_embeddings.values())).shape[0]
    emb_matrix = np.zeros((len(labels), text_dim), dtype=np.float32)
    for i, label in enumerate(labels):
        if label in glove_embeddings:
            emb_matrix[i] = glove_embeddings[label]
        else:
            sub_words = _split_compound_word(label)
            vecs = [glove_embeddings[w] for w in sub_words if w in glove_embeddings]
            if vecs:
                emb_matrix[i] = np.mean(vecs, axis=0)
            else:
                emb_matrix[i] = np.random.randn(text_dim).astype(np.float32) * 0.1
    return emb_matrix


def build_affordance_emb_tensor(device: torch.device) -> torch.Tensor:
    """加载 GloVe 并构建 [17, 300] 的 affordance 嵌入张量。"""
    glove_path = download_glove()

    # 收集所有需要的词（包括复合词的子词）
    all_words = list(AFFORDANCE_LABELS)
    for label in AFFORDANCE_LABELS:
        all_words.extend(_split_compound_word(label))
    all_words = list(set(all_words))

    glove_emb = load_glove_embeddings(glove_path, all_words)
    emb_matrix = build_affordance_embeddings(glove_emb, AFFORDANCE_LABELS)
    return torch.tensor(emb_matrix, dtype=torch.float32).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 从图像路径解析物体类别和 affordance
# ─────────────────────────────────────────────────────────────────────────────

def parse_img_path(img_path: str):
    """
    从图像文件名解析 (object_category, affordance_name)。
    PIAD 文件名格式: ..._<affordance>_<ObjectCategory>_<idx>.jpg
    例: /data/Seen/img/grasp_Chair_001.jpg -> ('Chair', 'grasp')
    实际格式参考 dataset.py: cut_str[-2] 是 affordance，cut_str[-3] 是物体类别。
    """
    basename = os.path.basename(img_path)
    # 去掉扩展名
    name = os.path.splitext(basename)[0]
    parts = name.split('_')

    affordance = None
    object_category = "unknown"

    if len(parts) >= 2:
        candidate = parts[-2]
        if candidate in AFFORDANCE_LABELS:
            affordance = candidate
            # 物体类别在 affordance 前一位
            if len(parts) >= 3:
                object_category = parts[-3]
        else:
            # 备选：扫描整个文件名
            for aff in AFFORDANCE_LABELS:
                if aff in name:
                    affordance = aff
                    break

    return object_category, affordance


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def prepopulate_image_memory(
    model_path: str = "./model_list/iag_textemb_seen.pt",
    data_root: str = "./Data",
    setting: str = "Seen",
    store_dir: str = "./image_memory_store",
    device_str: str = "cuda:0",
    max_samples: int = None,
    max_images_per_key: int = 4,
    use_faiss: bool = True,
    num_workers: int = 0,
):
    """
    预填充图像记忆库。

    Args:
        model_path:         IAG_TextEmb 权重文件路径
        data_root:          PIAD 数据集根目录（含 Seen/Unseen 子目录）
        setting:            'Seen' 或 'Unseen'
        store_dir:          图像记忆库存储目录
        device_str:         计算设备
        max_samples:        最大处理样本数（None 表示全部）
        max_images_per_key: 每个 (object, affordance) 键最多保存的条数
        use_faiss:          是否使用 FAISS 索引
        num_workers:        DataLoader 工作进程数
    """
    print("=" * 60)
    print("Image Memory Pre-population")
    print("=" * 60)

    # ── 设备 ──────────────────────────────────────────────────────────────
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # ── 1. 加载模型 ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading model from {model_path}")
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
    # 兼容带 'module.' 前缀的 DDP 权重
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✓ Model loaded")

    # ── 2. 构建 affordance 文本嵌入 ────────────────────────────────────────
    print("\n[2/4] Building affordance text embeddings (GloVe 300d)")
    affordance_emb_tensor = build_affordance_emb_tensor(device)
    print(f"✓ Affordance embedding tensor: {affordance_emb_tensor.shape}")

    # ── 3. 加载数据集 ──────────────────────────────────────────────────────
    print(f"\n[3/4] Loading PIAD dataset ({setting})")
    data_path = os.path.join(data_root, setting)

    for fname in ['Point_Test.txt', 'Img_Test.txt', 'Box_Test.txt']:
        fpath = os.path.join(data_path, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")

    dataset = PIAD(
        run_type='val',
        setting_type=setting,
        point_path=os.path.join(data_path, 'Point_Test.txt'),
        img_path=os.path.join(data_path, 'Img_Test.txt'),
        box_path=os.path.join(data_path, 'Box_Test.txt'),
    )
    print(f"✓ Dataset size: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # ── 4. 初始化 ImageMemoryManager ──────────────────────────────────────
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

    # ── 5. 遍历数据集，提取特征并存储 ─────────────────────────────────────
    print("\nProcessing samples...")
    print("-" * 60)

    stored_count = 0
    skip_count = 0

    # val 数据集返回: (Img, Point, affordance_label, img_path, point_path, sub_box, obj_box)
    for batch_idx, sample in enumerate(tqdm(loader, desc="Extracting features")):
        if max_samples is not None and stored_count >= max_samples:
            break

        try:
            img        = sample[0].to(device)          # [1, 3, H, W]
            # sample[1]: Point  (不需要)
            # sample[2]: affordance_label (不需要)
            img_path   = sample[3][0]                  # str
            # sample[4]: point_path (不需要)
            sub_box    = sample[5].to(device)          # [1, 4]
            obj_box    = sample[6].to(device)          # [1, 4]

            # 解析物体类别和 affordance
            object_category, affordance_name = parse_img_path(img_path)
            if affordance_name is None:
                skip_count += 1
                continue

            aff_idx = AFFORDANCE_LABELS.index(affordance_name)

            # 提取图像特征 F_i, F_s, F_e
            with torch.no_grad():
                F_I = model.img_encoder(img)                          # [1, 512, h, w]
                F_i, F_s, F_e = model.get_img_and_feature(
                    img, sub_box, obj_box
                )

            # 转为 numpy
            img_np  = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            F_I_np  = F_I.squeeze(0).cpu().numpy()                     # [512, h, w]
            F_i_np  = F_i.squeeze(0).cpu().numpy()                     # [512, 4, 4]
            F_s_np  = F_s.squeeze(0).cpu().numpy()                     # [512, 4, 4]
            F_e_np  = F_e.squeeze(0).cpu().numpy()                     # [512, 4, 4]

            sub_box_np = sub_box.squeeze(0).cpu().numpy()              # [4]
            obj_box_np = obj_box.squeeze(0).cpu().numpy()              # [4]

            # 存入记忆库
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
                print(f"  Stored {stored_count} memories "
                      f"(skipped {skip_count})")

        except Exception as e:
            skip_count += 1
            if skip_count <= 5:
                print(f"\n  [Warning] Sample {batch_idx} failed: {e}")
            continue

    # ── 保存索引 ───────────────────────────────────────────────────────────
    manager.save()

    print("\n" + "=" * 60)
    print(f"Done. Stored: {stored_count}  Skipped: {skip_count}")
    stats = manager.get_stats()
    print(f"Total images in store: {stats['total_images']}")
    print(f"Unique (object, affordance) keys: {len(stats['categories'])}")
    print("=" * 60)

    return manager


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-populate image memory from PIAD dataset using IAG_TextEmb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str,
        default="./model_list/iag_textemb_seen.pt",
        help="Path to IAG_TextEmb weights (.pt file)",
    )
    parser.add_argument(
        "--data_root", type=str,
        default="./Data",
        help="PIAD dataset root directory (contains Seen/ and Unseen/)",
    )
    parser.add_argument(
        "--setting", type=str, default="Seen",
        choices=["Seen", "Unseen"],
        help="Dataset setting",
    )
    parser.add_argument(
        "--store_dir", type=str,
        default="./image_memory_store",
        help="Directory to save the image memory store",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Compute device (cuda:0 or cpu)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to process (None = all)",
    )
    parser.add_argument(
        "--max_images_per_key", type=int, default=4,
        help="Maximum images stored per (object, affordance) key",
    )
    parser.add_argument(
        "--no_faiss", action="store_true",
        help="Disable FAISS, use NumPy brute-force instead",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader worker processes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepopulate_image_memory(
        model_path=args.model_path,
        data_root=args.data_root,
        setting=args.setting,
        store_dir=args.store_dir,
        device_str=args.device,
        max_samples=args.max_samples,
        max_images_per_key=args.max_images_per_key,
        use_faiss=not args.no_faiss,
        num_workers=args.num_workers,
    )
