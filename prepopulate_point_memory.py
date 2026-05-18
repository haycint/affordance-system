"""
prepopulate_point_memory.py

预填充点云记忆库：从 PIAD 数据集加载所有样本（Seen + Unseen），
使用 IAG_TextEmb 模型提取 ARM 特征和点云特征，
根据 Ground Truth 每点标签生成正偏好（标签为1的点 + 其最近的30个邻居），
将记忆存储到 MemoryManager。
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

try:
    from scipy.spatial import KDTree
except ImportError:
    print("Warning: scipy not installed. Install with: pip install scipy")
    KDTree = None

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.MyNet import IAG_TextEmb
from memory_system.memory_manager import MemoryManager

# 全局 affordance 标签列表（与 PIAD 数据集一致）
AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]


def get_arm_feature(model, img, xyz, sub_box, obj_box, text_emb):
    """
    捕获 ARM 模块的输出特征。

    Args:
        model: IAG_TextEmb 模型
        img: 图像张量 [B, 3, H, W]
        xyz: 点云张量 [B, 3, N]
        sub_box: subject box [B, 4]
        obj_box: object box [B, 4]
        text_emb: 文本嵌入 [B, text_dim]

    Returns:
        arm_feature: ARM 输出特征 [B, N_p+N_i, C]
    """
    arm_feature = None

    def hook(module, input, output):
        nonlocal arm_feature
        arm_feature = output.detach()

    # 注册 hook 到 ARM 模块
    if hasattr(model, 'ARM'):
        handle = model.ARM.register_forward_hook(hook)
    else:
        raise AttributeError("Model does not have ARM module")

    try:
        with torch.no_grad():
            _ = model(img, xyz, sub_box, obj_box, text_emb)
    finally:
        handle.remove()

    return arm_feature


def get_point_features(model, img, xyz, sub_box, obj_box, text_emb):
    """
    提取每点特征（通过点云编码器的最后一层，并上采样回原始分辨率）。

    Args:
        model: IAG_TextEmb 模型
        img: 图像张量 [B, 3, H, W]
        xyz: 点云张量 [B, 3, N]
        sub_box: subject box [B, 4]
        obj_box: object box [B, 4]
        text_emb: 文本嵌入 [B, text_dim]

    Returns:
        point_features: 每点特征 [B, N, C]
    """
    # 如果模型有专门的方法，直接使用
    if hasattr(model, 'get_point_features'):
        with torch.no_grad():
            return model.get_point_features(xyz)

    # 否则使用 hook 捕获 decoder 的输出
    point_feat = None

    def hook(module, input, output):
        nonlocal point_feat
        # output 应该是 [B, C, N] 或 [B, N, C]
        point_feat = output.detach()

    # 尝试找到合适的 hook 点
    hook_target = None
    if hasattr(model, 'decoder'):
        if hasattr(model.decoder, 'fp1'):
            hook_target = model.decoder.fp1
        elif hasattr(model.decoder, 'final_layer'):
            hook_target = model.decoder.final_layer

    if hook_target is None:
        # 如果找不到合适的 hook 点，返回简单的点云坐标作为特征
        print("Warning: Could not find suitable hook point for point features, using xyz coordinates")
        return xyz.permute(0, 2, 1)  # [B, N, 3]

    handle = hook_target.register_forward_hook(hook)

    try:
        with torch.no_grad():
            _ = model(img, xyz, sub_box, obj_box, text_emb)
    finally:
        handle.remove()

    if point_feat is not None:
        # 确保输出格式为 [B, N, C]
        if point_feat.dim() == 3 and point_feat.shape[1] != xyz.shape[2]:
            point_feat = point_feat.permute(0, 2, 1)
        return point_feat
    else:
        raise RuntimeError("Failed to capture point features")


def generate_preference_from_gt_labels(gt_labels, points_xyz, neighbor_k=30):
    """
    根据 ground truth 二进制标签生成偏好矩阵。
    对于每个标签为 1 的点，找到它最近的 neighbor_k 个点（欧氏距离），
    将这些点标记为正偏好（+1），其余点为 0。

    Args:
        gt_labels: Ground truth 标签 [N]
        points_xyz: 点云坐标 [N, 3]
        neighbor_k: 每个正点的邻居数量

    Returns:
        pref: 偏好矩阵 [N]
    """
    N = len(gt_labels)
    pref = np.zeros(N, dtype=np.float32)
    positive_indices = np.where(gt_labels > 0.5)[0]

    if len(positive_indices) == 0:
        return pref

    if KDTree is None:
        # 如果没有 scipy，使用简单的方法
        for idx in positive_indices:
            pref[idx] = 1.0
        return pref

    # 使用 KDTree 进行快速最近邻搜索
    tree = KDTree(points_xyz)
    selected = set()

    for idx in positive_indices:
        # 查询最近的 neighbor_k 个点（包括自身）
        dists, neighbors = tree.query(points_xyz[idx], k=min(neighbor_k, N))
        selected.update(neighbors)

    selected = list(selected)
    pref[selected] = 1.0
    return pref


def prepopulate_point_memory(
    model_weights_path: str,
    data_root: str,
    memory_dir: str = "./point_memory",
    device: str = "cuda:0",
    batch_size: int = 1,
    max_samples: int = None,
    neighbor_k: int = 30,
    max_memories: int = 5000,
    use_faiss: bool = True,
):
    """
    预填充点云记忆库。

    Args:
        model_weights_path: 预训练的 IAG_TextEmb 权重文件路径
        data_root: PIAD 数据集根目录
        memory_dir: 记忆库存储目录
        device: 计算设备
        batch_size: 批次大小（建议 1）
        max_samples: 最大处理样本数（用于调试）
        neighbor_k: 每个正点附近包含的邻居点数
        max_memories: 记忆库最大容量
        use_faiss: 是否使用 FAISS 索引
    """
    print("=" * 60)
    print("Starting Point Memory Prepopulation")
    print("=" * 60)

    # 检查设备
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # 1. 加载模型
    print(f"\n[1/5] Loading model from {model_weights_path}")
    try:
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
            text_dim=300
        )

        # 加载权重
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

        state_dict = torch.load(model_weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise

    # 2. 创建 MemoryManager
    print(f"\n[2/5] Initializing MemoryManager at {memory_dir}")
    try:
        os.makedirs(memory_dir, exist_ok=True)
        memory_manager = MemoryManager(
            emb_dim=512,
            index_dim=128,
            feat_dim=512,
            store_dir=memory_dir,
            max_memories=max_memories,
            default_top_k=5,
            fusion_alpha=0.3,
            use_faiss=use_faiss,
            async_formation=False
        )
        print(f"✓ MemoryManager initialized")
    except Exception as e:
        print(f"✗ Failed to initialize MemoryManager: {e}")
        raise

    # 3. 加载 PIAD 数据集
    print(f"\n[3/5] Loading PIAD dataset from {data_root}")
    try:
        from data_utils.dataset import PIAD as PIADDataset

        datasets = []

        # 加载 Seen 数据集
        seen_point_path = os.path.join(data_root, 'Seen', 'point')
        seen_img_path = os.path.join(data_root, 'Seen', 'img')
        seen_box_path = os.path.join(data_root, 'Seen', 'box')

        if os.path.exists(seen_point_path):
            print(f"  Loading Seen dataset...")
            seen_dataset = PIADDataset(
                run_type='train',
                setting_type='Seen',
                point_path=seen_point_path,
                img_path=seen_img_path,
                box_path=seen_box_path,
                transform=None
            )
            datasets.append(seen_dataset)
            print(f"  ✓ Seen dataset: {len(seen_dataset)} samples")

        # 加载 Unseen 数据集
        unseen_point_path = os.path.join(data_root, 'Unseen', 'point')
        unseen_img_path = os.path.join(data_root, 'Unseen', 'img')
        unseen_box_path = os.path.join(data_root, 'Unseen', 'box')

        if os.path.exists(unseen_point_path):
            print(f"  Loading Unseen dataset...")
            unseen_dataset = PIADDataset(
                run_type='val',
                setting_type='Unseen',
                point_path=unseen_point_path,
                img_path=unseen_img_path,
                box_path=unseen_box_path,
                transform=None
            )
            datasets.append(unseen_dataset)
            print(f"  ✓ Unseen dataset: {len(unseen_dataset)} samples")

        if len(datasets) == 0:
            raise ValueError("No valid datasets found")

        full_dataset = ConcatDataset(datasets)
        print(f"✓ Total samples: {len(full_dataset)}")

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        raise

    # 4. 创建数据加载器
    print(f"\n[4/5] Creating DataLoader")
    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 5. 遍历样本，生成记忆
    print(f"\n[5/5] Processing samples and forming memories")
    print(f"  Max samples: {max_samples if max_samples else 'All'}")
    print(f"  Neighbor k: {neighbor_k}")
    print("-" * 60)

    stored_count = 0
    error_count = 0

    for batch_idx, sample in enumerate(tqdm(loader, desc="Processing")):
        if max_samples is not None and stored_count >= max_samples:
            break

        try:
            # 解析样本
            img = sample[0].to(device)
            points_list = sample[1]
            labels_list = sample[2]
            indices_list = sample[3]
            sub_box = sample[4].to(device)
            obj_box = sample[5].to(device)

            # 处理每个点云
            for point_cloud, gt_labels, aff_idx in zip(points_list, labels_list, indices_list):
                if max_samples is not None and stored_count >= max_samples:
                    break

                # 准备数据
                point_cloud = point_cloud.float().to(device)
                gt_labels = gt_labels.squeeze(0).cpu().numpy()
                aff_idx_val = aff_idx.item() if isinstance(aff_idx, torch.Tensor) else aff_idx
                aff_label = AFFORDANCE_LABELS[aff_idx_val] if aff_idx_val < len(AFFORDANCE_LABELS) else "unknown"

                # 文本嵌入（使用零向量）
                text_emb = torch.zeros(1, 300).to(device)

                # 提取 ARM 特征
                try:
                    arm_feat = get_arm_feature(model, img, point_cloud, sub_box, obj_box, text_emb)
                    if arm_feat is None:
                        error_count += 1
                        continue
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"\n  Warning: ARM feature extraction failed: {e}")
                    continue

                # 提取点云特征
                try:
                    point_feats = get_point_features(model, img, point_cloud, sub_box, obj_box, text_emb)
                    point_feats_np = point_feats.squeeze(0).cpu().numpy()
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"\n  Warning: Point features extraction failed: {e}")
                    continue

                # 点云坐标
                point_xyz = point_cloud.squeeze(0).cpu().numpy().T

                # 生成偏好矩阵
                pref = generate_preference_from_gt_labels(gt_labels, point_xyz, neighbor_k=neighbor_k)

                # 存储记忆
                try:
                    entry_id = memory_manager.form_memory(
                        arm_feature=arm_feat,
                        point_cloud=point_xyz,
                        point_features=point_feats_np,
                        preference_matrix=pref,
                        reward=1.0,
                        outcome="success",
                        object_category="object",
                        affordance_label=aff_label,
                        confidence=1.0,
                        text_embedding=text_emb.squeeze(0).cpu().numpy()
                    )
                    stored_count += 1

                    if stored_count % 100 == 0:
                        print(f"\n  Progress: {stored_count} memories stored, {error_count} errors")

                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"\n  Warning: Memory formation failed: {e}")
                    continue

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n  Warning: Batch {batch_idx} failed: {e}")
            continue

    # 保存记忆库
    print(f"\n" + "=" * 60)
    print("Saving memory store...")
    try:
        memory_manager.save()
        print(f"✓ Memory store saved to {memory_dir}")
    except Exception as e:
        print(f"✗ Failed to save memory store: {e}")
        raise

    print("=" * 60)
    print("Prepopulation Summary:")
    print(f"  Total memories stored: {stored_count}")
    print(f"  Total errors: {error_count}")
    if stored_count + error_count > 0:
        print(f"  Success rate: {stored_count / (stored_count + error_count) * 100:.1f}%")
    print("=" * 60)

    return memory_manager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepopulate point memory for IAG_TextEmb")
    parser.add_argument("--weights", type=str, required=True, help="Path to IAG_textemb_seen.pt")
    parser.add_argument("--data_root", type=str, required=True, help="PIAD dataset root directory")
    parser.add_argument("--memory_dir", type=str, default="./point_memory", help="Directory to save memory store")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0 or cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (recommend 1)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--neighbor_k", type=int, default=30, help="Number of nearest neighbors for positive preference")
    parser.add_argument("--max_memories", type=int, default=5000, help="Maximum memory capacity")
    parser.add_argument("--no_faiss", action="store_true", help="Disable FAISS (use NumPy brute-force)")
    args = parser.parse_args()

    prepopulate_point_memory(
        model_weights_path=args.weights,
        data_root=args.data_root,
        memory_dir=args.memory_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        neighbor_k=args.neighbor_k,
        max_memories=args.max_memories,
        use_faiss=not args.no_faiss
    )

