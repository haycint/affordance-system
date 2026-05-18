#!/usr/bin/env python3
"""
预填充记忆库脚本
用法：python prefill_memory.py --setting Seen --memory_manager_path /path/to/memory_system
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from data_utils.dataset import PIADInference, pc_normalize
from memory_system import MemoryManager, ImageMemoryManager, MemoryEntry

def extract_image_feature(img_path, device='cpu'):
    """使用预训练ResNet提取图像特征（1024维）"""
    import torchvision.models as models
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    # 去掉最后的分类层
    modules = list(model.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(img_tensor).squeeze().cpu().numpy()
    return feat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, required=True, help='Seen or Unseen')
    parser.add_argument('--data_dir', type=str, default='./Data', help='PIAD数据根目录')
    parser.add_argument('--pc_memory_store', type=str, default='./memory_store', help='点云记忆库存储目录')
    parser.add_argument('--img_memory_store', type=str, default='./image_memory_store', help='图像记忆库存储目录')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / args.setting
    # 优先用Test列表，否则Train
    point_txt = data_dir / 'Point_Test.txt'
    img_txt   = data_dir / 'Img_Test.txt'
    box_txt   = data_dir / 'Box_Test.txt'
    if not point_txt.exists():
        point_txt = data_dir / 'Point_Train.txt'
        img_txt   = data_dir / 'Img_Train.txt'
        box_txt   = data_dir / 'Box_Train.txt'

    dataset = PIADInference(
        point_path=str(point_txt),
        img_path=str(img_txt),
        box_path=str(box_txt),
        img_size=(224, 224)
    )

    # 初始化记忆管理器
    pc_manager = MemoryManager(emb_dim=512, index_dim=128, store_dir=args.pc_memory_store)
    pc_manager.load()
    img_manager = ImageMemoryManager(store_dir=args.img_memory_store, feature_dim=1024)
    img_manager.load()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"开始预填充 {args.setting} 数据集，共 {len(dataset)} 样本...")

    for i in range(len(dataset)):
        # 解包 PIADInference 返回值
        img_tensor, point_tensor, affordance_label, img_path, point_path, sub_box, obj_box, affordance_index = dataset[i]

        # ---------- 处理点云 ----------
        # point_tensor 形状 (3, N)，转置为 (N,3)
        points = point_tensor.cpu().numpy() if torch.is_tensor(point_tensor) else np.array(point_tensor)
        if points.shape[0] == 3 and points.shape[1] != 3:
            points = points.transpose(1, 0)
        # GT affordance label (N,)
        gt_label = affordance_label.cpu().numpy().flatten() if torch.is_tensor(affordance_label) else np.array(affordance_label).flatten()

        # 从文件名解析物体类别和可供性
        sample_id = Path(img_path).stem
        parts = sample_id.split('_')
        affordance_name = parts[-2] if len(parts) >= 2 else "unknown"
        object_category = parts[-3] if len(parts) >= 3 else "unknown"

        # 构造点云记忆条目（示例中使用GT作为偏好，奖励1.0）
        entry = MemoryEntry(
            arm_feature=np.random.rand(512).astype(np.float32),  # 实际应计算ARM特征，此处暂用随机
            preference_matrix=gt_label.astype(np.float32),
            reward=1.0,  # GT作为高质量偏好
            affordance_label=affordance_name,
            object_category=object_category,
            extra_data={'sample_id': sample_id}
        )
        pc_manager.add_entry(entry)

        # ---------- 处理图像 ----------
        # 提取图像特征（使用ResNet50）
        img_feat = extract_image_feature(img_path, device)
        # 存储图像特征及标签
        img_manager.add_image(
            image_path=img_path,
            feature=img_feat,
            object_category=object_category,
            affordance_label=affordance_name,
            sub_box=sub_box.tolist() if torch.is_tensor(sub_box) else sub_box,
            obj_box=obj_box.tolist() if torch.is_tensor(obj_box) else obj_box
        )

        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1}/{len(dataset)} 样本")

    # 保存记忆库
    pc_manager.save()
    img_manager.save()
    print("记忆库预填充完成！")

if __name__ == '__main__':
    main()