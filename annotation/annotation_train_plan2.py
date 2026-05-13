"""
Training Script for Annotation Model Plan 2
多小模型协同标注 — 训练脚本

三个模型分别训练：
1. BBoxDetector      — 主客体边界框检测
2. ItemClassifier     — 物品类别分类
3. InteractionClassifier — 交互动作分类

也支持联合训练 (Plan2CombinedTrainer)，按顺序训练三个模型。

用法
----
::

    # 单独训练检测器
    python annotation_train_plan2.py --model bbox --epochs 50 --data_dir ./Data

    # 单独训练物品分类器
    python annotation_train_plan2.py --model item --epochs 50 --data_dir ./Data

    # 单独训练交互分类器
    python annotation_train_plan2.py --model interaction --epochs 50 --data_dir ./Data

    # 联合训练全部模型
    python annotation_train_plan2.py --model all --epochs 50 --data_dir ./Data
"""

import os
import sys
import random
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotation.annotation_model_plan2 import (
    BBoxDetector, ItemClassifier, InteractionClassifier,
    BBoxDetectorLoss, ItemClassifierLoss, InteractionClassifierLoss,
    AFFORDANCE_LABELS, DEFAULT_ITEM_CATEGORIES,
    build_bbox_detector, build_item_classifier, build_interaction_classifier,
)


# ============================================================
# 数据集类
# ============================================================

AFFORDANCE_LABELS_LIST = AFFORDANCE_LABELS
ITEM_CATEGORIES_LIST = DEFAULT_ITEM_CATEGORIES


class BBoxDetectorDataset(Dataset):
    """
    BBoxDetector 训练数据集

    输出: (image, {subject_box, object_box})，坐标归一化到 [0, 1]
    """

    def __init__(self, data_dir, setting='Seen', split='train',
                 img_size=(224, 224), augment=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # 加载数据列表
        self.samples = self._load_samples(setting, split)
        print(f"[BBoxDetectorDataset] {len(self.samples)} samples ({split})")

    def _load_samples(self, setting, split):
        """从 PIAD 数据集加载数据列表"""
        setting_dir = os.path.join(self.data_dir, setting)
        samples = []

        if split == 'train':
            img_list_file = os.path.join(setting_dir, 'Img_Train.txt')
            box_list_file = os.path.join(setting_dir, 'Box_Train.txt')
        else:
            img_list_file = os.path.join(setting_dir, 'Img_Test.txt')
            box_list_file = os.path.join(setting_dir, 'Box_Test.txt')

        img_files = []
        box_files = []

        if os.path.exists(img_list_file):
            with open(img_list_file, 'r') as f:
                img_files = [line.strip() for line in f if line.strip()]
        if os.path.exists(box_list_file):
            with open(box_list_file, 'r') as f:
                box_files = [line.strip() for line in f if line.strip()]

        for img_path, box_path in zip(img_files, box_files):
            if os.path.exists(img_path) and os.path.exists(box_path):
                samples.append({'image_path': img_path, 'box_path': box_path})

        return samples

    def _parse_box_json(self, box_path, original_size):
        """解析 LabelMe 格式标注文件"""
        import json
        with open(box_path, 'r') as f:
            data = json.load(f)

        subject_box = None
        object_box = None

        for shape in data.get('shapes', []):
            if shape['label'] == 'subject':
                pts = shape['points']
                subject_box = torch.tensor([
                    min(pts[0][0], pts[1][0]),
                    min(pts[0][1], pts[1][1]),
                    max(pts[0][0], pts[1][0]),
                    max(pts[0][1], pts[1][1]),
                ], dtype=torch.float32)
            elif shape['label'] == 'object':
                pts = shape['points']
                object_box = torch.tensor([
                    min(pts[0][0], pts[1][0]),
                    min(pts[0][1], pts[1][1]),
                    max(pts[0][0], pts[1][0]),
                    max(pts[0][1], pts[1][1]),
                ], dtype=torch.float32)

        # 默认值
        if subject_box is None:
            subject_box = torch.tensor([0.1, 0.1, 0.4, 0.4], dtype=torch.float32)
        if object_box is None:
            object_box = torch.tensor([0.4, 0.4, 0.9, 0.9], dtype=torch.float32)

        # 归一化到 [0, 1]
        w, h = original_size
        subject_box[0] /= w; subject_box[2] /= w
        subject_box[1] /= h; subject_box[3] /= h
        object_box[0] /= w;  object_box[2] /= w
        object_box[1] /= h;  object_box[3] /= h

        return subject_box, object_box

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        original_size = image.size  # (W, H)

        subject_box, object_box = self._parse_box_json(
            sample['box_path'], original_size
        )

        # 数据增强
        if self.augment and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            subject_box[0], subject_box[2] = 1 - subject_box[2], 1 - subject_box[0]
            object_box[0], object_box[2] = 1 - object_box[2], 1 - object_box[0]

        image = self.transform(image)

        return image, {
            'subject_box': subject_box,
            'object_box': object_box,
        }


class ItemClassifierDataset(Dataset):
    """
    ItemClassifier 训练数据集

    输出: (cropped_image, item_category)
    从图像中根据 GT 物体框裁剪物体区域，并分类物品类别。
    """

    def __init__(self, data_dir, setting='Seen', split='train',
                 img_size=(224, 224), augment=False,
                 item_categories=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.item_categories = item_categories or ITEM_CATEGORIES_LIST

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.samples = self._load_samples(setting, split)
        print(f"[ItemClassifierDataset] {len(self.samples)} samples ({split})")

    def _load_samples(self, setting, split):
        """加载数据，提取物品裁剪 + 类别"""
        setting_dir = os.path.join(self.data_dir, setting)
        samples = []

        if split == 'train':
            img_list_file = os.path.join(setting_dir, 'Img_Train.txt')
            box_list_file = os.path.join(setting_dir, 'Box_Train.txt')
        else:
            img_list_file = os.path.join(setting_dir, 'Img_Test.txt')
            box_list_file = os.path.join(setting_dir, 'Box_Test.txt')

        img_files = []
        box_files = []

        if os.path.exists(img_list_file):
            with open(img_list_file, 'r') as f:
                img_files = [line.strip() for line in f if line.strip()]
        if os.path.exists(box_list_file):
            with open(box_list_file, 'r') as f:
                box_files = [line.strip() for line in f if line.strip()]

        for img_path, box_path in zip(img_files, box_files):
            if os.path.exists(img_path) and os.path.exists(box_path):
                # 从文件路径提取物品类别
                item_cat = self._extract_item_category(img_path)
                samples.append({
                    'image_path': img_path,
                    'box_path': box_path,
                    'item_category': item_cat,
                })

        return samples

    def _extract_item_category(self, path):
        """从文件路径提取物品类别索引"""
        filename = os.path.basename(path).lower()
        for i, cat in enumerate(self.item_categories):
            if cat != 'other' and cat in filename:
                return i
        return len(self.item_categories) - 1  # 'other'

    def _get_object_box(self, box_path, original_size):
        """获取物体边界框"""
        import json
        with open(box_path, 'r') as f:
            data = json.load(f)

        for shape in data.get('shapes', []):
            if shape['label'] == 'object':
                pts = shape['points']
                return [
                    min(pts[0][0], pts[1][0]),
                    min(pts[0][1], pts[1][1]),
                    max(pts[0][0], pts[1][0]),
                    max(pts[0][1], pts[1][1]),
                ]
        # 默认框
        w, h = original_size
        return [w * 0.2, h * 0.2, w * 0.8, h * 0.8]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')

        obj_box = self._get_object_box(sample['box_path'], image.size)

        # 裁剪物体区域
        x1, y1, x2, y2 = [int(v) for v in obj_box]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.width, x2); y2 = min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, image.width, image.height

        crop = image.crop((x1, y1, x2, y2))
        crop = self.transform(crop)

        item_cat = torch.tensor(sample['item_category'], dtype=torch.long)

        return crop, item_cat


class InteractionClassifierDataset(Dataset):
    """
    InteractionClassifier 训练数据集

    输出: (subject_crop, object_crop, subject_box, object_box, interaction_label)
    """

    def __init__(self, data_dir, setting='Seen', split='train',
                 img_size=(224, 224), augment=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.samples = self._load_samples(setting, split)
        print(f"[InteractionClassifierDataset] {len(self.samples)} samples ({split})")

    def _load_samples(self, setting, split):
        setting_dir = os.path.join(self.data_dir, setting)
        samples = []

        if split == 'train':
            img_list_file = os.path.join(setting_dir, 'Img_Train.txt')
            box_list_file = os.path.join(setting_dir, 'Box_Train.txt')
        else:
            img_list_file = os.path.join(setting_dir, 'Img_Test.txt')
            box_list_file = os.path.join(setting_dir, 'Box_Test.txt')

        img_files = []
        box_files = []

        if os.path.exists(img_list_file):
            with open(img_list_file, 'r') as f:
                img_files = [line.strip() for line in f if line.strip()]
        if os.path.exists(box_list_file):
            with open(box_list_file, 'r') as f:
                box_files = [line.strip() for line in f if line.strip()]

        for img_path, box_path in zip(img_files, box_files):
            if os.path.exists(img_path) and os.path.exists(box_path):
                interaction = self._extract_interaction(img_path)
                samples.append({
                    'image_path': img_path,
                    'box_path': box_path,
                    'interaction': interaction,
                })

        return samples

    def _extract_interaction(self, path):
        """从文件名提取交互类型索引"""
        filename = os.path.basename(path).lower()
        for i, aff in enumerate(AFFORDANCE_LABELS_LIST):
            if aff in filename:
                return i
        return 0  # 默认 grasp

    def _get_boxes(self, box_path, original_size):
        """解析标注文件获取主体和客体框"""
        import json
        with open(box_path, 'r') as f:
            data = json.load(f)

        subject_box = None
        object_box = None

        for shape in data.get('shapes', []):
            if shape['label'] == 'subject':
                pts = shape['points']
                subject_box = [min(pts[0][0], pts[1][0]),
                               min(pts[0][1], pts[1][1]),
                               max(pts[0][0], pts[1][0]),
                               max(pts[0][1], pts[1][1])]
            elif shape['label'] == 'object':
                pts = shape['points']
                object_box = [min(pts[0][0], pts[1][0]),
                              min(pts[0][1], pts[1][1]),
                              max(pts[0][0], pts[1][0]),
                              max(pts[0][1], pts[1][1])]

        w, h = original_size
        if subject_box is None:
            subject_box = [w * 0.1, h * 0.1, w * 0.4, h * 0.4]
        if object_box is None:
            object_box = [w * 0.4, h * 0.4, w * 0.9, h * 0.9]

        return subject_box, object_box

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        w, h = image.size

        sub_box, obj_box = self._get_boxes(sample['box_path'], image.size)

        # 裁剪主体和客体区域
        def safe_crop(img, box):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img.width, x2); y2 = min(img.height, y2)
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, img.width, img.height
            return img.crop((x1, y1, x2, y2))

        s_crop = self.transform(safe_crop(image, sub_box))
        o_crop = self.transform(safe_crop(image, obj_box))

        # 归一化边界框坐标
        sub_norm = torch.tensor(sub_box, dtype=torch.float32)
        sub_norm[0] /= w; sub_norm[2] /= w
        sub_norm[1] /= h; sub_norm[3] /= h

        obj_norm = torch.tensor(obj_box, dtype=torch.float32)
        obj_norm[0] /= w; obj_norm[2] /= w
        obj_norm[1] /= h; obj_norm[3] /= h

        interaction = torch.tensor(sample['interaction'], dtype=torch.long)

        return s_crop, o_crop, sub_norm, obj_norm, interaction


class SyntheticBBoxDataset(Dataset):
    """合成数据集 (用于测试/演示 BBoxDetector)"""

    def __init__(self, num_samples=500, img_size=(224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(
            np.random.randint(0, 255, (*self.img_size, 3), dtype=np.uint8)
        )
        img = self.transform(img)

        # 随机归一化框 [0, 1]
        cx1, cy1 = random.uniform(0.05, 0.4), random.uniform(0.05, 0.4)
        cx2, cy2 = random.uniform(0.5, 0.95), random.uniform(0.5, 0.95)
        subject_box = torch.tensor([cx1, cy1, cx1 + 0.3, cy1 + 0.3],
                                   dtype=torch.float32).clamp(0, 1)
        object_box = torch.tensor([cx2, cy2, cx2 + 0.3, cy2 + 0.3],
                                  dtype=torch.float32).clamp(0, 1)

        return img, {'subject_box': subject_box, 'object_box': object_box}


class SyntheticItemDataset(Dataset):
    """合成数据集 (用于测试/演示 ItemClassifier)"""

    def __init__(self, num_samples=500, num_classes=20):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        img = self.transform(img)
        label = random.randint(0, self.num_classes - 1)
        return img, torch.tensor(label, dtype=torch.long)


class SyntheticInteractionDataset(Dataset):
    """合成数据集 (用于测试/演示 InteractionClassifier)"""

    def __init__(self, num_samples=500, num_interactions=17):
        self.num_samples = num_samples
        self.num_interactions = num_interactions
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s_crop = self.transform(Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        ))
        o_crop = self.transform(Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        ))
        sub_box = torch.tensor([random.uniform(0, 0.3), random.uniform(0, 0.3),
                                random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)],
                               dtype=torch.float32)
        obj_box = torch.tensor([random.uniform(0.3, 0.7), random.uniform(0.3, 0.7),
                                random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)],
                               dtype=torch.float32)
        interaction = torch.tensor(random.randint(0, self.num_interactions - 1),
                                   dtype=torch.long)
        return s_crop, o_crop, sub_box, obj_box, interaction


# ============================================================
# Collate Functions
# ============================================================

def bbox_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {
        'subject_box': torch.stack([b[1]['subject_box'] for b in batch]),
        'object_box': torch.stack([b[1]['object_box'] for b in batch]),
    }
    return images, targets


def item_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return images, labels


def interaction_collate_fn(batch):
    s_crops = torch.stack([b[0] for b in batch])
    o_crops = torch.stack([b[1] for b in batch])
    sub_boxes = torch.stack([b[2] for b in batch])
    obj_boxes = torch.stack([b[3] for b in batch])
    labels = torch.stack([b[4] for b in batch])
    return s_crops, o_crops, sub_boxes, obj_boxes, labels


# ============================================================
# Trainer: BBoxDetector
# ============================================================

class BBoxDetectorTrainer:
    """BBoxDetector 训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}

        self.save_dir = config.get('save_dir', './checkpoints/annotation_plan2')
        os.makedirs(self.save_dir, exist_ok=True)

        # 模型
        self.model = build_bbox_detector(
            pretrained=config.get('pretrained', True)
        ).to(self.device)

        # 损失
        self.criterion = BBoxDetectorLoss(
            coord_weight=config.get('coord_weight', 5.0),
            noobj_weight=config.get('noobj_weight', 0.5),
        )

        # 优化器
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        lr = config.get('lr', 1e-4)
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': other_params, 'lr': lr},
        ], weight_decay=config.get('weight_decay', 1e-4))

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 100), eta_min=1e-6
        )

        # 数据
        self._init_dataloaders()

        total_p = sum(p.numel() for p in self.model.parameters())
        print(f"[BBoxDetector] {total_p/1e6:.2f}M params on {self.device}")

    def _init_dataloaders(self):
        dataset_type = self.config.get('dataset_type', 'synthetic')
        data_dir = self.config.get('data_dir')
        bs = self.config.get('batch_size', 8)

        if dataset_type == 'piad' and data_dir:
            train_ds = BBoxDetectorDataset(data_dir, split='train', augment=True)
            val_ds = BBoxDetectorDataset(data_dir, split='val')
        else:
            train_ds = SyntheticBBoxDataset(num_samples=500)
            val_ds = SyntheticBBoxDataset(num_samples=100)

        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                       num_workers=4, collate_fn=bbox_collate_fn)
        self.val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                      num_workers=4, collate_fn=bbox_collate_fn)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'[BBox] Epoch {self.current_epoch + 1}')

        for images, targets in pbar:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs['predictions'], targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_dict.items()
                             if 'total' not in k})

        avg = total_loss / max(len(self.train_loader), 1)
        self.history['train_loss'].append(avg)
        return avg

    def validate(self):
        # 临时切换到训练模式，让模型输出 'predictions'
        self.model.train()
        total_loss = 0
    
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
    
                outputs = self.model(images)          # 此时 outputs 包含 'predictions'
                loss, _ = self.criterion(outputs['predictions'], targets)
                total_loss += loss.item()
    
        # 恢复为评估模式（如果后续需要推理）
        self.model.eval()
    
        avg = total_loss / max(len(self.val_loader), 1)
        self.history['val_loss'].append(avg)
        return avg

    def train(self, epochs, save_every=5):
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()

            print(f"  Epoch {epoch+1}: train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save('best_bbox.pt', epoch)

            if (epoch + 1) % save_every == 0:
                self._save(f'epoch_{epoch+1}_bbox.pt', epoch)

        self._save('final_bbox.pt', epochs - 1)

    def _save(self, filename, epoch):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.current_epoch = ckpt['epoch'] + 1
        self.best_loss = ckpt.get('best_loss', float('inf'))
        self.history = ckpt.get('history', self.history)
        print(f"[BBox] Loaded checkpoint from epoch {ckpt['epoch']}")


# ============================================================
# Trainer: ItemClassifier
# ============================================================

class ItemClassifierTrainer:
    """ItemClassifier 训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        self.save_dir = config.get('save_dir', './checkpoints/annotation_plan2')
        os.makedirs(self.save_dir, exist_ok=True)

        num_item_classes = config.get('num_item_classes', 20)
        self.model = build_item_classifier(
            num_item_classes=num_item_classes,
            pretrained=config.get('pretrained', True)
        ).to(self.device)

        self.criterion = ItemClassifierLoss()

        lr = config.get('lr', 1e-4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                     weight_decay=config.get('weight_decay', 1e-4))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 100), eta_min=1e-6
        )

        self._init_dataloaders()

        total_p = sum(p.numel() for p in self.model.parameters())
        print(f"[ItemClassifier] {total_p/1e6:.2f}M params on {self.device}")

    def _init_dataloaders(self):
        dataset_type = self.config.get('dataset_type', 'synthetic')
        data_dir = self.config.get('data_dir')
        bs = self.config.get('batch_size', 8)
        num_item_classes = self.config.get('num_item_classes', 20)

        if dataset_type == 'piad' and data_dir:
            train_ds = ItemClassifierDataset(data_dir, split='train', augment=True)
            val_ds = ItemClassifierDataset(data_dir, split='val')
        else:
            train_ds = SyntheticItemDataset(num_samples=500, num_classes=num_item_classes)
            val_ds = SyntheticItemDataset(num_samples=100, num_classes=num_item_classes)

        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                        num_workers=4, collate_fn=item_collate_fn)
        self.val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                      num_workers=4, collate_fn=item_collate_fn)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc=f'[Item] Epoch {self.current_epoch + 1}')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss, _ = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{correct/max(total,1):.3f}'})

        avg = total_loss / max(len(self.train_loader), 1)
        self.history['train_loss'].append(avg)
        return avg

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss, _ = self.criterion(logits, labels)
                total_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        acc = correct / max(total, 1)
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(acc)
        return avg_loss, acc

    def train(self, epochs, save_every=5):
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            print(f"  Epoch {epoch+1}: train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  acc={val_acc:.3f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save('best_item.pt', epoch)

            if (epoch + 1) % save_every == 0:
                self._save(f'epoch_{epoch+1}_item.pt', epoch)

        self._save('final_item.pt', epochs - 1)

    def _save(self, filename, epoch):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.current_epoch = ckpt['epoch'] + 1
        self.best_loss = ckpt.get('best_loss', float('inf'))
        self.history = ckpt.get('history', self.history)
        print(f"[Item] Loaded checkpoint from epoch {ckpt['epoch']}")


# ============================================================
# Trainer: InteractionClassifier
# ============================================================

class InteractionClassifierTrainer:
    """InteractionClassifier 训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        self.save_dir = config.get('save_dir', './checkpoints/annotation_plan2')
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = build_interaction_classifier(
            num_interactions=config.get('num_interactions', 17)
        ).to(self.device)

        self.criterion = InteractionClassifierLoss()

        lr = config.get('lr', 1e-4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                     weight_decay=config.get('weight_decay', 1e-4))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 100), eta_min=1e-6
        )

        self._init_dataloaders()

        total_p = sum(p.numel() for p in self.model.parameters())
        print(f"[InteractionClassifier] {total_p/1e6:.2f}M params on {self.device}")

    def _init_dataloaders(self):
        dataset_type = self.config.get('dataset_type', 'synthetic')
        data_dir = self.config.get('data_dir')
        bs = self.config.get('batch_size', 8)

        if dataset_type == 'piad' and data_dir:
            train_ds = InteractionClassifierDataset(data_dir, split='train', augment=True)
            val_ds = InteractionClassifierDataset(data_dir, split='val')
        else:
            train_ds = SyntheticInteractionDataset(num_samples=500)
            val_ds = SyntheticInteractionDataset(num_samples=100)

        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                        num_workers=4, collate_fn=interaction_collate_fn)
        self.val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                      num_workers=4, collate_fn=interaction_collate_fn)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc=f'[Interact] Epoch {self.current_epoch + 1}')

        for s_crop, o_crop, sub_box, obj_box, labels in pbar:
            s_crop = s_crop.to(self.device)
            o_crop = o_crop.to(self.device)
            sub_box = sub_box.to(self.device)
            obj_box = obj_box.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(s_crop, o_crop, sub_box, obj_box)
            loss, _ = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{correct/max(total,1):.3f}'})

        avg = total_loss / max(len(self.train_loader), 1)
        self.history['train_loss'].append(avg)
        return avg

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for s_crop, o_crop, sub_box, obj_box, labels in self.val_loader:
                s_crop = s_crop.to(self.device)
                o_crop = o_crop.to(self.device)
                sub_box = sub_box.to(self.device)
                obj_box = obj_box.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(s_crop, o_crop, sub_box, obj_box)
                loss, _ = self.criterion(logits, labels)
                total_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        acc = correct / max(total, 1)
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(acc)
        return avg_loss, acc

    def train(self, epochs, save_every=5):
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            print(f"  Epoch {epoch+1}: train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  acc={val_acc:.3f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save('best_interaction.pt', epoch)

            if (epoch + 1) % save_every == 0:
                self._save(f'epoch_{epoch+1}_interaction.pt', epoch)

        self._save('final_interaction.pt', epochs - 1)

    def _save(self, filename, epoch):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.current_epoch = ckpt['epoch'] + 1
        self.best_loss = ckpt.get('best_loss', float('inf'))
        self.history = ckpt.get('history', self.history)
        print(f"[Interaction] Loaded checkpoint from epoch {ckpt['epoch']}")


# ============================================================
# Combined Trainer: 顺序训练全部模型
# ============================================================

class Plan2CombinedTrainer:
    """
    Plan 2 联合训练器

    按顺序训练三个模型:
    1. BBoxDetector (检测主客体框)
    2. ItemClassifier (分类物品)
    3. InteractionClassifier (分类交互)

    每个模型独立训练、独立保存检查点。
    """

    def __init__(self, config):
        self.config = config
        self.epochs = config.get('epochs', 100)
        self.save_every = config.get('save_every', 5)

    def train(self):
        print('\n' + '=' * 60)
        print('Plan 2: 多小模型协同训练 — 开始')
        print('=' * 60)

        # ── Phase 1: BBoxDetector ───────────────────────────────────
        print('\n' + '-' * 40)
        print('Phase 1/3: BBoxDetector 训练')
        print('-' * 40)
        bbox_trainer = BBoxDetectorTrainer(self.config)
        if self.config.get('resume_bbox'):
            bbox_trainer.load_checkpoint(self.config['resume_bbox'])
        bbox_trainer.train(self.epochs, self.save_every)

        # ── Phase 2: ItemClassifier ─────────────────────────────────
        print('\n' + '-' * 40)
        print('Phase 2/3: ItemClassifier 训练')
        print('-' * 40)
        item_trainer = ItemClassifierTrainer(self.config)
        if self.config.get('resume_item'):
            item_trainer.load_checkpoint(self.config['resume_item'])
        item_trainer.train(self.epochs, self.save_every)

        # ── Phase 3: InteractionClassifier ──────────────────────────
        print('\n' + '-' * 40)
        print('Phase 3/3: InteractionClassifier 训练')
        print('-' * 40)
        inter_trainer = InteractionClassifierTrainer(self.config)
        if self.config.get('resume_interaction'):
            inter_trainer.load_checkpoint(self.config['resume_interaction'])
        inter_trainer.train(self.epochs, self.save_every)

        print('\n' + '=' * 60)
        print('Plan 2: 全部模型训练完成')
        print('=' * 60)

        return {
            'bbox_history': bbox_trainer.history,
            'item_history': item_trainer.history,
            'interaction_history': inter_trainer.history,
        }


# ============================================================
# CLI
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description='Train Annotation Model Plan 2 (Multi-Model Collaborative)'
    )
    parser.add_argument('--model', type=str, default='all',
                        choices=['bbox', 'item', 'interaction', 'all'],
                        help='Which model to train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints/annotation_plan2')
    parser.add_argument('--dataset_type', type=str, default='synthetic',
                        choices=['piad', 'synthetic'])
    parser.add_argument('--num_item_classes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_bbox', type=str, default=None)
    parser.add_argument('--resume_item', type=str, default=None)
    parser.add_argument('--resume_interaction', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()
    set_seed(args.seed)

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': args.device,
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'dataset_type': args.dataset_type,
        'num_item_classes': args.num_item_classes,
        'num_interactions': 17,
        'pretrained': True,
        'weight_decay': 1e-4,
        'coord_weight': 5.0,
        'noobj_weight': 0.5,
        'resume_bbox': args.resume_bbox,
        'resume_item': args.resume_item,
        'resume_interaction': args.resume_interaction,
        'save_every': args.save_every,
    }

    if args.model == 'all':
        trainer = Plan2CombinedTrainer(config)
        trainer.train()
    elif args.model == 'bbox':
        trainer = BBoxDetectorTrainer(config)
        if args.resume_bbox:
            trainer.load_checkpoint(args.resume_bbox)
        trainer.train(args.epochs, args.save_every)
    elif args.model == 'item':
        trainer = ItemClassifierTrainer(config)
        if args.resume_item:
            trainer.load_checkpoint(args.resume_item)
        trainer.train(args.epochs, args.save_every)
    elif args.model == 'interaction':
        trainer = InteractionClassifierTrainer(config)
        if args.resume_interaction:
            trainer.load_checkpoint(args.resume_interaction)
        trainer.train(args.epochs, args.save_every)


if __name__ == '__main__':
    main()