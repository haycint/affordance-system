"""
IAG_TextEmb Model Pipeline Slicing
IAG_TextEmb 模型按数据流方式的层级流水线切分

与 IAG (MyNet) 的差异仅在 Stage 6:
  - IAG          使用 Decoder       (2*emb_dim → 17)
  - IAG_TextEmb  使用 Decoder_TextEmb (3*emb_dim → 17, 需要text_emb输入)

数据流:
  Stage 0: Img_Encoder      — ResNet-18 图像特征提取
  Stage 1: Mask Feature     — 掩码特征提取 (主体/客体/场景)
  Stage 2: ROI Align        — 场景特征裁剪
  Stage 3: Point_Encoder    — PointNet++ MSG 点云编码
  Stage 4: JRA              — 联合区域对齐
  Stage 5: ARM              — 可供性揭示模块
  Stage 6: Decoder_TextEmb  — 文本增强解码 (额外接收 text_emb)

完整数据流图:
  img ──→ [Stage 0: Img_Encoder] ──→ F_I
    │                                    │
    │           ┌─── sub_box, obj_box ────┤
    │           │                        │
    │           ▼                        ▼
    │    [Stage 1: Mask Feature] ──→ F_i, F_s, F_e
    │                                    │
    │                              [Stage 2: ROI Align] ──→ F_e_aligned
    │                                    │
    xyz ──→ [Stage 3: Point_Encoder] ──→ F_p_wise
                                             │
              F_i + F_p_wise[-1]             │
                    │                        │
                    ▼                        │
              [Stage 4: JRA] ──→ F_j        │
                    │                        │
              F_j + F_s + F_e_aligned        │
                    │                        │
                    ▼                        │
              [Stage 5: ARM] ──→ affordance  │
                    │                        │
              F_j + affordance + F_p_wise + text_emb
                    │                        │
                    ▼                        │
              [Stage 6: Decoder_TextEmb] ──→ _3daffordance, logits, to_KL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .pipeline_dataflow import PipelineStage, DataFlowPipeline, MicroBatchPipeline


def build_iag_textemb_pipeline(model, num_slices=None):
    """
    构建 IAG_TextEmb 的层级流水线

    Args:
        model: IAG_TextEmb 实例
        num_slices: 切片数量 (None 则使用完整7阶段)

    Returns:
        DataFlowPipeline 实例
    """
    stages = []

    # ================================================================
    # Stage 0: Img_Encoder — ResNet-18 图像特征提取
    # ================================================================
    stages.append(PipelineStage(
        name='img_encoder',
        module=model.img_encoder,
        input_keys=['img'],
        output_keys=['F_I'],
        description='ResNet-18 图像特征提取 → [B, 512, 7, 7]',
    ))

    # ================================================================
    # Stage 1: Mask Feature — 掩码特征提取
    # ================================================================
    def mask_feature_execute(img, F_I, sub_box, obj_box):
        device = F_I.device
        B = F_I.size(0)
        raw_size = img.size(2)
        current_size = F_I.size(2)
        scale_factor = current_size / raw_size

        sub_box_scaled = sub_box.clone()
        obj_box_scaled = obj_box.clone()
        sub_box_scaled[:, :] = sub_box_scaled[:, :] * scale_factor
        obj_box_scaled[:, :] = obj_box_scaled[:, :] * scale_factor

        obj_mask = torch.zeros_like(F_I)
        obj_roi_box = []
        for i in range(B):
            obj_mask[i, :, int(obj_box_scaled[i][1] + 0.5):int(obj_box_scaled[i][3] + 0.5),
            int(obj_box_scaled[i][0] + 0.5):int(obj_box_scaled[i][2] + 0.5)] = 1
            roi_obj = [obj_box_scaled[i][0], obj_box_scaled[i][1],
                       obj_box_scaled[i][2] + 0.5, obj_box_scaled[i][3]]
            roi_obj.insert(0, i)
            obj_roi_box.append(roi_obj)
        obj_roi_box = torch.tensor(obj_roi_box).float().to(device)

        sub_roi_box = []
        Scene_mask = obj_mask.clone()
        for i in range(B):
            Scene_mask[i, :, int(sub_box_scaled[i][1] + 0.5):int(sub_box_scaled[i][3] + 0.5),
            int(sub_box_scaled[i][0] + 0.5):int(sub_box_scaled[i][2] + 0.5)] = 1
            roi_sub = [sub_box_scaled[i][0], sub_box_scaled[i][1],
                       sub_box_scaled[i][2], sub_box_scaled[i][3]]
            roi_sub.insert(0, i)
            sub_roi_box.append(roi_sub)
        Scene_mask = torch.abs(Scene_mask - 1)
        Scene_mask_feature = F_I * Scene_mask
        sub_roi_box = torch.tensor(sub_roi_box).float().to(device)
        obj_feature = roi_align(F_I, obj_roi_box, output_size=(4, 4), sampling_ratio=4)
        sub_feature = roi_align(F_I, sub_roi_box, output_size=(4, 4), sampling_ratio=4)

        return {
            'F_i': obj_feature,
            'F_s': sub_feature,
            'F_e': Scene_mask_feature,
        }

    stages.append(PipelineStage(
        name='mask_feature',
        input_keys=['img', 'F_I', 'sub_box', 'obj_box'],
        output_keys=['F_i', 'F_s', 'F_e'],
        execute_fn=mask_feature_execute,
        description='掩码特征提取: 客体F_i + 主体F_s + 场景F_e',
    ))

    # ================================================================
    # Stage 2: ROI Align — 场景特征裁剪
    # ================================================================
    def roi_align_execute(F_e):
        device = F_e.device
        B = F_e.size(0)
        roi_box = []
        for i in range(B):
            roi_box.append([i, 0.0, 0.0, 6.0, 6.0])
        ROI_box = torch.tensor(roi_box).float().to(device)
        F_e_aligned = roi_align(F_e, ROI_box, output_size=(4, 4))
        return {'F_e_aligned': F_e_aligned}

    stages.append(PipelineStage(
        name='roi_align',
        input_keys=['F_e'],
        output_keys=['F_e_aligned'],
        execute_fn=roi_align_execute,
        description='ROI Align 场景特征裁剪 → [B, 512, 4, 4]',
    ))

    # ================================================================
    # Stage 3: Point_Encoder — PointNet++ MSG 点云编码
    # ================================================================
    stages.append(PipelineStage(
        name='point_encoder',
        module=model.point_encoder,
        input_keys=['xyz'],
        output_keys=['F_p_wise'],
        description='PointNet++ MSG 点云编码 → 4层层级特征',
    ))

    # ================================================================
    # Stage 4: JRA — 联合区域对齐
    # ================================================================
    def jra_execute(F_i, F_p_wise):
        F_j = model.JRA(F_i, F_p_wise[-1][1])
        return {'F_j': F_j}

    stages.append(PipelineStage(
        name='jra',
        module=model.JRA,
        input_keys=['F_i', 'F_p_wise'],
        output_keys=['F_j'],
        execute_fn=jra_execute,
        description='Joint Region Alignment 联合区域对齐',
    ))

    # ================================================================
    # Stage 5: ARM — 可供性揭示模块
    # ================================================================
    def arm_execute(F_j, F_s, F_e_aligned):
        affordance = model.ARM(F_j, F_s, F_e_aligned)
        return {'affordance': affordance}

    stages.append(PipelineStage(
        name='arm',
        module=model.ARM,
        input_keys=['F_j', 'F_s', 'F_e_aligned'],
        output_keys=['affordance'],
        execute_fn=arm_execute,
        description='Affordance Revealed Module 可供性揭示',
    ))

    # ================================================================
    # Stage 6: Decoder_TextEmb — 文本增强解码
    # ================================================================
    # 与 IAG 的 Decoder 不同点:
    #   - 额外接收 text_emb [B, text_dim] 输入
    #   - 分类头维度为 3*emb_dim (visual+text) 而非 2*emb_dim
    #   - 逐点输出头维度为 emb_dim+emb_dim (text-conditioned)
    def decoder_textemb_execute(F_j, affordance, F_p_wise, text_emb):
        _3daffordance, logits, to_KL = model.decoder(F_j, affordance, F_p_wise, text_emb)
        return {
            '_3daffordance': _3daffordance,
            'logits': logits,
            'to_KL': to_KL,
        }

    stages.append(PipelineStage(
        name='decoder_textemb',
        module=model.decoder,
        input_keys=['F_j', 'affordance', 'F_p_wise', 'text_emb'],
        output_keys=['_3daffordance', 'logits', 'to_KL'],
        execute_fn=decoder_textemb_execute,
        description='Decoder_TextEmb 文本增强解码 → 3D affordance + logits + KL',
    ))

    # 如果指定了合并切片数量，合并相邻阶段
    if num_slices is not None and num_slices < len(stages):
        from .iag_pipeline import _merge_stages as _merge
        stages = _merge(stages, num_slices, model)

    pipeline = DataFlowPipeline(stages, name='IAG_TextEmb_Pipeline')
    return pipeline