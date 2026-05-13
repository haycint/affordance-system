"""
Annotation Model Plan 2: Multi-Model Collaborative Architecture
多小模型协同标注架构

与 Plan 1 的端到端单模型 (ResNet-50 + FPN + RPN + BBoxHead + InteractionClassifier,
约 48M 参数) 形成对比，Plan 2 使用三个专职小模型协同完成标注任务：

┌──────────────────────────────────────────────────────────────────┐
│  Plan 1 (端到端):                                                │
│    Image → [ResNet-50 + FPN + RPN + RoI + BBoxHead + IC]        │
│           → subject_box, object_box, interaction                  │
│    参数量: ~48M                                                   │
│                                                                   │
│  Plan 2 (多模型协同):                                             │
│    Image → [BBoxDetector]      → subject_box, object_box          │
│    Crop  → [ItemClassifier]    → item_category                    │
│    Crops → [InteractionClassifier] → interaction_type             │
│    参数量: ~25M (12M + 11.5M + 1.5M)                              │
└──────────────────────────────────────────────────────────────────┘

三个模型：
1. BBoxDetector  — 主客体边界框检测 (YOLO-style lightweight detector)
2. ItemClassifier — 物品类别分类 (ResNet-18 image classifier)
3. InteractionClassifier — 交互动作分类 (SmallCNN + Spatial + Fusion)

每个模型可独立训练、独立调优、独立部署，也可通过 Plan2Pipeline 串联推理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import nms
import numpy as np


# ============================================================
# 类别定义
# ============================================================

AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]

DEFAULT_ITEM_CATEGORIES = [
    'bowl', 'cup', 'mug', 'bottle', 'chair', 'sofa', 'bed',
    'table', 'desk', 'keyboard', 'scissors', 'knife', 'hammer',
    'phone', 'book', 'shoe', 'hat', 'door', 'headphones', 'other'
]


# ============================================================
# 1. BBoxDetector — YOLO-style 轻量主客体检测器
# ============================================================

class BBoxDetector(nn.Module):
    """
    轻量级主客体检测器 (YOLO-style single-stage detector)

    架构
    ----
    ::

        Image [B, 3, 224, 224]
          │
          ResNet-18 backbone → [B, 512, 7, 7]
          │
          Neck (2× Conv3×3)  → [B, 256, 7, 7]
          │
          Head (Conv1×1)     → [B, 10, 7, 7]
                               │
                               reshape → [B, 2, 5, 7, 7]
                                         │
                                         2 classes × (1 conf + 4 box)
                                         subject: [conf, cx, cy, w, h]
                                         object:  [conf, cx, cy, w, h]

    每个网格单元 (grid cell) 预测一个 subject 框和一个 object 框。
    坐标编码采用归一化 [0, 1]，置信度使用 sigmoid 激活。
    推理时：置信度筛选 + NMS 去重。

    参数量: ~12M (vs Plan1 的 ~48M)
    """

    def __init__(self, grid_size=7, pretrained=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = 2  # subject, object

        # ── Backbone: ResNet-18 ─────────────────────────────────────
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        # Output: [B, 512, 7, 7] for 224×224 input

        # ── Neck ────────────────────────────────────────────────────
        self.neck = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # ── Detection Head ─────────────────────────────────────────
        # 2 classes × (1 confidence + 4 box coords) = 10
        self.head = nn.Conv2d(256, self.num_classes * 5, 1)

        self._init_detection_weights()

    def _init_detection_weights(self):
        """初始化新增层权重"""
        for m in [self.neck, self.head]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, 0, 0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, images):
        """
        前向传播

        Args:
            images: [B, 3, H, W] 输入图像 (建议 224×224)

        Returns:
            训练模式: {'predictions': [B, 2, 5, S, S]}
            推理模式: {'subject_boxes': [N_s, 4], 'object_boxes': [N_o, 4], ...}
        """
        B = images.size(0)
        S = self.grid_size

        feat = self.backbone(images)   # [B, 512, 7, 7]
        feat = self.neck(feat)          # [B, 256, 7, 7]
        pred = self.head(feat)          # [B, 10, 7, 7]

        # Reshape: [B, 10, S, S] → [B, 2, 5, S, S]
        pred = pred.view(B, self.num_classes, 5, S, S)

        if self.training:
            return {'predictions': pred}

        return self._decode_predictions(pred, images.size(2), images.size(3))

    def _decode_predictions(self, pred, img_h, img_w,
                            conf_threshold=0.5, nms_threshold=0.5):
        """推理时将 YOLO-style 预测解码为边界框"""
        B, C, _, S, _ = pred.shape
        device = pred.device

        results = {
            'subject_boxes': [],
            'object_boxes': [],
            'subject_scores': [],
            'object_scores': [],
        }

        for b in range(B):
            for cls_idx, cls_name in [(0, 'subject'), (1, 'object')]:
                cp = pred[b, cls_idx]  # [5, S, S]

                # 置信度
                conf = torch.sigmoid(cp[0])  # [S, S]

                # 边界框坐标 (归一化 [0, 1])
                bcx = torch.sigmoid(cp[1])   # center x
                bcy = torch.sigmoid(cp[2])   # center y
                bw  = torch.sigmoid(cp[3])   # width
                bh  = torch.sigmoid(cp[4])   # height

                boxes, scores = [], []
                for i in range(S):
                    for j in range(S):
                        if conf[i, j] < conf_threshold:
                            continue
                        cx = bcx[i, j].item() * img_w
                        cy = bcy[i, j].item() * img_h
                        w  = bw[i, j].item()  * img_w
                        h  = bh[i, j].item()  * img_h

                        x1 = max(0, cx - w / 2)
                        y1 = max(0, cy - h / 2)
                        x2 = min(img_w, cx + w / 2)
                        y2 = min(img_h, cy + h / 2)

                        boxes.append([x1, y1, x2, y2])
                        scores.append(conf[i, j].item())

                if boxes:
                    bt = torch.tensor(boxes, device=device, dtype=torch.float32)
                    st = torch.tensor(scores, device=device, dtype=torch.float32)
                    keep = nms(bt, st, nms_threshold)
                    results[f'{cls_name}_boxes'].append(bt[keep])
                    results[f'{cls_name}_scores'].append(st[keep])
                else:
                    results[f'{cls_name}_boxes'].append(
                        torch.zeros((0, 4), device=device))
                    results[f'{cls_name}_scores'].append(
                        torch.zeros((0,), device=device))

        # 合并 batch 结果
        for key in results:
            parts = [p for p in results[key] if p.numel() > 0]
            if parts:
                results[key] = torch.cat(parts, dim=0)
            elif 'box' in key:
                results[key] = torch.zeros((0, 4), device=device)
            else:
                results[key] = torch.zeros((0,), device=device)

        return results


# ============================================================
# 2. ItemClassifier — 物品类别分类器
# ============================================================

class ItemClassifier(nn.Module):
    """
    物品类别分类器

    架构
    ----
    ::

        Cropped Object Image [B, 3, 224, 224]
          │
          ResNet-18 (pretrained, no FC) → [B, 512]
          │
          FC Head: 512 → 256 → num_item_classes
          │
          Item category logits [B, num_item_classes]

    输入为从图像中裁剪出的物体区域，输出该物体的类别概率。
    物品类别需根据实际数据集定义 (默认 20 类)。

    参数量: ~11.2M
    """

    def __init__(self, num_item_classes=20, pretrained=True):
        super().__init__()
        self.num_item_classes = num_item_classes

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_item_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.cls_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images):
        """
        Args:
            images: [B, 3, 224, 224] 裁剪的物体图像

        Returns:
            logits: [B, num_item_classes]
        """
        feat = self.backbone(images)      # [B, 512, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 512]
        logits = self.cls_head(feat)        # [B, num_item_classes]
        return logits


# ============================================================
# 3. InteractionClassifier — 交互动作分类器
# ============================================================

class SmallFeatureCNN(nn.Module):
    """轻量级 CNN 特征提取器 (用于 ROI 区域特征)

    ::

        Input [B, 3, 224, 224]
          │
          Conv(3→64, 7×7, s=2) → BN → ReLU → MaxPool(3, s=2)
          Conv(64→128, 3×3)    → BN → ReLU → MaxPool(2)
          Conv(128→256, 3×3)   → BN → ReLU → AdaptiveAvgPool(1)
          │
          FC(256 → out_dim)

    参数量: ~0.7M
    """

    def __init__(self, out_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.conv_layers(x)   # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        return self.fc(x)           # [B, out_dim]


class InteractionClassifier(nn.Module):
    """
    交互动作分类器

    架构
    ----
    ::

        Subject Crop → SmallFeatureCNN (shared) → s_feat [B, 256]
        Object Crop  → SmallFeatureCNN (shared) → o_feat [B, 256]
        Spatial [B, 4] → MLP → sp_feat [B, 128]
          │
        Concat [s_feat, o_feat, sp_feat] = [B, 640]
          │
        Fusion MLP: 640 → 512 → 256 → 17
          │
        Interaction logits [B, 17]

    空间关系编码:
        dx = (obj_cx - sub_cx) / sub_w
        dy = (obj_cy - sub_cy) / sub_h
        dw = log(obj_w / sub_w)
        dh = log(obj_h / sub_h)

    参数量: ~1.5M
    """

    def __init__(self, num_interactions=17, feat_dim=256):
        super().__init__()
        self.num_interactions = num_interactions

        # 共享特征提取器 (主体和客体共用同一网络)
        self.feature_extractor = SmallFeatureCNN(out_dim=feat_dim)

        # 空间关系编码器
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        # 融合分类器
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2 + 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_interactions),
        )

    def compute_spatial(self, sub_box, obj_box):
        """计算主体-客体空间关系向量 [dx, dy, dw, dh]

        Args:
            sub_box: [B, 4] (x1, y1, x2, y2) 归一化坐标
            obj_box: [B, 4] (x1, y1, x2, y2) 归一化坐标

        Returns:
            spatial: [B, 4]
        """
        s_cx = (sub_box[:, 0] + sub_box[:, 2]) / 2
        s_cy = (sub_box[:, 1] + sub_box[:, 3]) / 2
        s_w  = sub_box[:, 2] - sub_box[:, 0]
        s_h  = sub_box[:, 3] - sub_box[:, 1]

        o_cx = (obj_box[:, 0] + obj_box[:, 2]) / 2
        o_cy = (obj_box[:, 1] + obj_box[:, 3]) / 2
        o_w  = obj_box[:, 2] - obj_box[:, 0]
        o_h  = obj_box[:, 3] - obj_box[:, 1]

        dx = (o_cx - s_cx) / (s_w + 1e-6)
        dy = (o_cy - s_cy) / (s_h + 1e-6)
        dw = torch.log(o_w / (s_w + 1e-6) + 1e-6)
        dh = torch.log(o_h / (s_h + 1e-6) + 1e-6)

        return torch.stack([dx, dy, dw, dh], dim=1)

    def forward(self, sub_crop, obj_crop, sub_box, obj_box):
        """
        Args:
            sub_crop: [B, 3, H, W] 主体裁剪图像
            obj_crop: [B, 3, H, W] 客体裁剪图像
            sub_box:  [B, 4] 主体边界框 (归一化坐标)
            obj_box:  [B, 4] 客体边界框 (归一化坐标)

        Returns:
            logits: [B, num_interactions]
        """
        s_feat = self.feature_extractor(sub_crop)   # [B, feat_dim]
        o_feat = self.feature_extractor(obj_crop)    # [B, feat_dim]

        spatial = self.compute_spatial(sub_box, obj_box)   # [B, 4]
        sp_feat = self.spatial_encoder(spatial)             # [B, 128]

        combined = torch.cat([s_feat, o_feat, sp_feat], dim=1)  # [B, feat_dim*2+128]
        logits = self.fusion(combined)                           # [B, 17]

        return logits


# ============================================================
# 4. Plan2Pipeline — 组合推理流水线
# ============================================================

class Plan2Pipeline:
    """
    Plan 2 组合推理流水线

    串联三个小模型，完成完整标注任务::

        ┌────────────┐    subject_box, object_box
        │ BBoxDetector├──────────────────────────────┐
        └──────┬─────┘                                │
               │                                      │
               │ crop object region                   │ crop both regions
               ▼                                      ▼
        ┌──────────────┐                    ┌────────────────────────┐
        │ ItemClassifier│→ item_category    │ InteractionClassifier   │→ interaction
        └──────────────┘                    └────────────────────────┘
    """

    def __init__(self, bbox_detector, item_classifier, interaction_classifier):
        self.bbox_detector = bbox_detector
        self.item_classifier = item_classifier
        self.interaction_classifier = interaction_classifier

    @torch.no_grad()
    def predict(self, image_tensor, img_pil=None, crop_size=224):
        """
        完整推理流水线

        Args:
            image_tensor: [1, 3, H, W] 归一化后的图像张量
            img_pil: PIL Image (用于裁剪，若 None 则从 tensor 反变换)
            crop_size: 裁剪区域缩放尺寸

        Returns:
            dict: 包含所有标注结果
        """
        from torchvision import transforms as T

        device = next(self.bbox_detector.parameters()).device
        image_tensor = image_tensor.to(device)

        crop_transform = T.Compose([
            T.Resize((crop_size, crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ── Step 1: 检测主客体框 ────────────────────────────────────
        det_result = self.bbox_detector(image_tensor)

        result = {
            'subject_boxes': det_result['subject_boxes'],
            'object_boxes': det_result['object_boxes'],
            'subject_scores': det_result.get('subject_scores'),
            'object_scores': det_result.get('object_scores'),
            'item_logits': None,
            'item_predictions': None,
            'interaction_logits': None,
            'interaction_predictions': None,
        }

        # ── Step 2: 分类物品类别 ────────────────────────────────────
        obj_boxes = det_result['object_boxes']
        if obj_boxes.numel() > 0 and img_pil is not None:
            item_logits_list = []
            for box in obj_boxes:
                x1, y1, x2, y2 = [int(v.clamp(0)) for v in box]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_pil.width, x2)
                y2 = min(img_pil.height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img_pil.crop((x1, y1, x2, y2))
                crop_t = crop_transform(crop).unsqueeze(0).to(device)
                logits = self.item_classifier(crop_t)
                item_logits_list.append(logits)

            if item_logits_list:
                result['item_logits'] = torch.cat(item_logits_list, dim=0)
                result['item_predictions'] = result['item_logits'].argmax(dim=1)

        # ── Step 3: 分类交互动作 ────────────────────────────────────
        sub_boxes = det_result['subject_boxes']
        if sub_boxes.numel() > 0 and obj_boxes.numel() > 0 and img_pil is not None:
            img_h, img_w = image_tensor.shape[2], image_tensor.shape[3]
            interaction_logits_list = []

            for s_box in sub_boxes:
                for o_box in obj_boxes:
                    # 裁剪主体和客体区域
                    sx1, sy1, sx2, sy2 = [int(v.clamp(0)) for v in s_box]
                    ox1, oy1, ox2, oy2 = [int(v.clamp(0)) for v in o_box]
                    sx2, sy2 = min(img_pil.width, sx2), min(img_pil.height, sy2)
                    ox2, oy2 = min(img_pil.width, ox2), min(img_pil.height, oy2)

                    if sx2 <= sx1 or sy2 <= sy1 or ox2 <= ox1 or oy2 <= oy1:
                        continue

                    s_crop = img_pil.crop((sx1, sy1, sx2, sy2))
                    o_crop = img_pil.crop((ox1, oy1, ox2, oy2))

                    s_t = crop_transform(s_crop).unsqueeze(0).to(device)
                    o_t = crop_transform(o_crop).unsqueeze(0).to(device)

                    # 归一化坐标
                    s_norm = s_box.unsqueeze(0).clone()
                    o_norm = o_box.unsqueeze(0).clone()
                    s_norm[:, [0, 2]] /= img_w
                    s_norm[:, [1, 3]] /= img_h
                    o_norm[:, [0, 2]] /= img_w
                    o_norm[:, [1, 3]] /= img_h

                    logits = self.interaction_classifier(s_t, o_t, s_norm, o_norm)
                    interaction_logits_list.append(logits)

            if interaction_logits_list:
                result['interaction_logits'] = torch.cat(interaction_logits_list, dim=0)
                result['interaction_predictions'] = result['interaction_logits'].argmax(dim=1)

        return result


# ============================================================
# 5. 损失函数
# ============================================================

class BBoxDetectorLoss(nn.Module):
    """
    YOLO-style 检测损失

    组成:
    - 置信度损失 (BCE): 该网格单元是否包含主体/客体中心
    - 无目标损失 (BCE): 惩罚误报置信度
    - 坐标损失 (MSE): 预测框与 GT 框的坐标偏差

    权重:
    - coord_weight: 坐标损失权重 (默认 5.0, 强调定位精度)
    - noobj_weight: 无目标损失权重 (默认 0.5, 降低负样本影响)
    """

    def __init__(self, coord_weight=5.0, noobj_weight=0.5):
        super().__init__()
        self.coord_weight = coord_weight
        self.noobj_weight = noobj_weight

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 2, 5, S, S] 模型原始输出
            targets: dict
                'subject_box': [B, 4] 归一化坐标 [x1, y1, x2, y2]
                'object_box':  [B, 4] 归一化坐标

        Returns:
            total_loss: scalar
            loss_dict: 各项损失明细
        """
        B, C, _, S, _ = predictions.shape
        device = predictions.device

        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        for cls_idx, cls_name in [(0, 'subject'), (1, 'object')]:
            cp = predictions[:, cls_idx]  # [B, 5, S, S]

            # 解码预测
            conf_pred = cp[:, 0]                # [B, S, S] (logit)
            cx_pred   = torch.sigmoid(cp[:, 1]) # [B, S, S]
            cy_pred   = torch.sigmoid(cp[:, 2])
            w_pred    = torch.sigmoid(cp[:, 3])
            h_pred    = torch.sigmoid(cp[:, 4])

            # GT 信息
            gt_box = targets[f'{cls_name}_box'].to(device)  # [B, 4] 归一化
            gt_cx = (gt_box[:, 0] + gt_box[:, 2]) / 2  # [B]
            gt_cy = (gt_box[:, 1] + gt_box[:, 3]) / 2
            gt_w  = gt_box[:, 2] - gt_box[:, 0]
            gt_h  = gt_box[:, 3] - gt_box[:, 1]

            # 确定哪个 grid cell 包含 GT 中心
            gt_j = (gt_cx * S).long().clamp(0, S - 1)  # [B]
            gt_i = (gt_cy * S).long().clamp(0, S - 1)

            # 构建 objectness mask
            obj_mask = torch.zeros(B, S, S, device=device)
            noobj_mask = torch.ones(B, S, S, device=device)

            for b in range(B):
                obj_mask[b, gt_i[b], gt_j[b]] = 1.0
                noobj_mask[b, gt_i[b], gt_j[b]] = 0.0

            # ── 置信度损失 (object cells) ──────────────────────────
            conf_target = obj_mask
            obj_conf_loss = F.binary_cross_entropy_with_logits(
                conf_pred, conf_target, reduction='none'
            )
            obj_conf_loss = (obj_conf_loss * obj_mask).sum()

            # ── 置信度损失 (no-object cells) ───────────────────────
            noobj_conf_loss = F.binary_cross_entropy_with_logits(
                conf_pred, torch.zeros_like(conf_pred), reduction='none'
            )
            noobj_conf_loss = (noobj_conf_loss * noobj_mask).sum()

            # ── 坐标损失 (only object cells) ───────────────────────
            coord_loss = torch.tensor(0.0, device=device)
            for b in range(B):
                i, j = gt_i[b], gt_j[b]
                coord_loss += F.mse_loss(cx_pred[b, i, j].unsqueeze(0),
                                         gt_cx[b].unsqueeze(0))
                coord_loss += F.mse_loss(cy_pred[b, i, j].unsqueeze(0),
                                         gt_cy[b].unsqueeze(0))
                coord_loss += F.mse_loss(w_pred[b, i, j].unsqueeze(0),
                                         gt_w[b].unsqueeze(0))
                coord_loss += F.mse_loss(h_pred[b, i, j].unsqueeze(0),
                                         gt_h[b].unsqueeze(0))

            cls_loss = (obj_conf_loss
                        + self.noobj_weight * noobj_conf_loss
                        + self.coord_weight * coord_loss)
            total_loss = total_loss + cls_loss

            loss_dict[f'{cls_name}_conf_loss'] = obj_conf_loss.item()
            loss_dict[f'{cls_name}_noobj_loss'] = noobj_conf_loss.item()
            loss_dict[f'{cls_name}_coord_loss'] = coord_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


class ItemClassifierLoss(nn.Module):
    """物品分类损失 (CrossEntropy)"""

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        return loss, {'item_cls_loss': loss.item()}


class InteractionClassifierLoss(nn.Module):
    """交互分类损失 (CrossEntropy)"""

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        return loss, {'interaction_cls_loss': loss.item()}


# ============================================================
# 6. 工厂函数
# ============================================================

def build_bbox_detector(pretrained=True):
    """构建 BBoxDetector"""
    return BBoxDetector(pretrained=pretrained)


def build_item_classifier(num_item_classes=20, pretrained=True):
    """构建 ItemClassifier"""
    return ItemClassifier(num_item_classes=num_item_classes, pretrained=pretrained)


def build_interaction_classifier(num_interactions=17):
    """构建 InteractionClassifier"""
    return InteractionClassifier(num_interactions=num_interactions)


# ============================================================
# 7. 参数量统计
# ============================================================

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    print('=' * 60)
    print('Plan 2: 多小模型协同架构 — 参数量统计')
    print('=' * 60)

    # BBoxDetector
    det = build_bbox_detector()
    p = count_parameters(det)
    print(f'\n1. BBoxDetector:  {p["total"]/1e6:.2f}M params ({p["trainable"]/1e6:.2f}M trainable)')

    # ItemClassifier
    cls_item = build_item_classifier()
    p = count_parameters(cls_item)
    print(f'2. ItemClassifier: {p["total"]/1e6:.2f}M params ({p["trainable"]/1e6:.2f}M trainable)')

    # InteractionClassifier
    cls_int = build_interaction_classifier()
    p = count_parameters(cls_int)
    print(f'3. InteractionClassifier: {p["total"]/1e6:.2f}M params ({p["trainable"]/1e6:.2f}M trainable)')

    # 总计
    total_all = sum(count_parameters(m)['total']
                    for m in [det, cls_item, cls_int])
    print(f'\nPlan 2 总计: {total_all/1e6:.2f}M params')
    print(f'Plan 1 参考: ~48.00M params')
    print(f'参数缩减比: {total_all/48e6*100:.1f}%')

    # 测试前向传播
    print('\n--- 前向传播测试 ---')
    x = torch.randn(2, 3, 224, 224)

    det.train()
    out = det(x)
    print(f'BBoxDetector output: {out["predictions"].shape}')

    cls_item.eval()
    out2 = cls_item(x)
    print(f'ItemClassifier output: {out2.shape}')

    cls_int.eval()
    s_crop = torch.randn(2, 3, 224, 224)
    o_crop = torch.randn(2, 3, 224, 224)
    s_box = torch.rand(2, 4)
    o_box = torch.rand(2, 4)
    out3 = cls_int(s_crop, o_crop, s_box, o_box)
    print(f'InteractionClassifier output: {out3.shape}')