"""
IAG (MyNet) Model Pipeline Slicing
IAG 模型按数据流方式的层级流水线切分

数据流:
  Stage 0: Img_Encoder      — ResNet-18 图像特征提取
  Stage 1: Mask Feature     — 掩码特征提取 (主体/客体/场景)
  Stage 2: ROI Align        — 场景特征裁剪
  Stage 3: Point_Encoder    — PointNet++ MSG 点云编码
  Stage 4: JRA              — 联合区域对齐
  Stage 5: ARM              — 可供性揭示模块
  Stage 6: Decoder          — 解码输出 (3D affordance + logits)

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
              F_j + affordance + F_p_wise    │
                    │                        │
                    ▼                        │
              [Stage 6: Decoder] ──→ _3daffordance, logits, to_KL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .pipeline_dataflow import PipelineStage, DataFlowPipeline, MicroBatchPipeline


def build_iag_pipeline(model, num_slices=None):
    """
    构建 IAG (MyNet) 的层级流水线

    Args:
        model: MyNet 实例
        num_slices: 切片数量 (None 则使用完整7阶段)

    Returns:
        DataFlowPipeline 实例
    """
    stages = []

    # ================================================================
    # Stage 0: Img_Encoder — ResNet-18 图像特征提取
    # ================================================================
    # 输入: img [B, 3, H, W]
    # 输出: F_I [B, 512, 7, 7]
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
    # 输入: raw_img, img_feature, sub_box, obj_box
    # 输出: F_i (obj_feature), F_s (sub_feature), F_e (Scene_mask_feature)
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
            'F_i': obj_feature,       # [B, 512, 4, 4] 客体特征
            'F_s': sub_feature,       # [B, 512, 4, 4] 主体特征
            'F_e': Scene_mask_feature, # [B, 512, 7, 7] 场景遮罩特征
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
    # 输入: F_e [B, 512, 7, 7]
    # 输出: F_e_aligned [B, 512, 4, 4]
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
    # 输入: xyz [B, 3, 2048]
    # 输出: F_p_wise (层级特征列表)
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
    # 输入: F_i [B, 512, 4, 4], F_p_wise[-1][1] [B, 512, N_p]
    # 输出: F_j [B, N_p+N_i, C]
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
    # 输入: F_j, F_s, F_e_aligned
    # 输出: affordance [B, N_p+N_i, C]
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
    # Stage 6: Decoder — 解码输出
    # ================================================================
    # 输入: F_j, affordance, F_p_wise
    # 输出: _3daffordance [B, N, 1], logits [B, 17], to_KL
    def decoder_execute(F_j, affordance, F_p_wise):
        _3daffordance, logits, to_KL = model.decoder(F_j, affordance, F_p_wise)
        return {
            '_3daffordance': _3daffordance,
            'logits': logits,
            'to_KL': to_KL,
        }

    stages.append(PipelineStage(
        name='decoder',
        module=model.decoder,
        input_keys=['F_j', 'affordance', 'F_p_wise'],
        output_keys=['_3daffordance', 'logits', 'to_KL'],
        execute_fn=decoder_execute,
        description='Decoder 解码 → 3D affordance + logits + KL features',
    ))

    # 如果指定了合并切片数量，合并相邻阶段
    if num_slices is not None and num_slices < len(stages):
        stages = _merge_stages(stages, num_slices, model)

    pipeline = DataFlowPipeline(stages, name='IAG_Pipeline')
    return pipeline


def _merge_stages(stages, num_slices, model):
    """将7个细粒度阶段合并为指定数量的粗粒度阶段"""
    # 预定义的合并策略：按功能耦合度分组
    merge_plans = {
        2: [
            ['img_encoder', 'mask_feature', 'roi_align', 'point_encoder'],  # 编码器组
            ['jra', 'arm', 'decoder'],                                        # 解码器组
        ],
        3: [
            ['img_encoder', 'mask_feature', 'roi_align'],   # 图像编码+ROI
            ['point_encoder'],                                # 点云编码
            ['jra', 'arm', 'decoder'],                        # 融合+解码
        ],
        4: [
            ['img_encoder'],                                  # 图像编码
            ['mask_feature', 'roi_align'],                    # 掩码+ROI
            ['point_encoder'],                                 # 点云编码
            ['jra', 'arm', 'decoder'],                         # 融合+解码
        ],
        5: [
            ['img_encoder'],                # 图像编码
            ['mask_feature', 'roi_align'],  # 掩码+ROI
            ['point_encoder'],               # 点云编码
            ['jra', 'arm'],                  # 融合+ARM
            ['decoder'],                     # 解码
        ],
    }

    if num_slices not in merge_plans:
        raise ValueError(f"num_slices must be 2-5 (got {num_slices}). Available: {list(merge_plans.keys())}")

    plan = merge_plans[num_slices]
    stage_map = {s.name: s for s in stages}
    merged = []

    for group in plan:
        group_stages = [stage_map[n] for n in group]

        # 收集输入/输出 key
        all_inputs = []
        all_outputs = []
        internal_keys = set()
        for gs in group_stages:
            all_inputs.extend(gs.input_keys)
            all_outputs.extend(gs.output_keys)
            internal_keys.update(gs.input_keys)
            internal_keys.update(gs.output_keys)

        # 外部输入 = 组内所有输入 - 组内所有输出
        outer_inputs = [k for k in dict.fromkeys(all_inputs) if k not in set(all_outputs)]
        outer_outputs = [k for k in dict.fromkeys(all_outputs) if k not in internal_keys or all_outputs.count(k) > 1]
        # 简化：取组内最后一个阶段的输出作为组输出
        outer_outputs = group_stages[-1].output_keys
        # 外部输入 = 第一个阶段的输入 + 后续阶段中不在前面输出中的输入
        outer_inputs = list(dict.fromkeys(all_inputs))
        seen_outputs = set()
        filtered_inputs = []
        for k in outer_inputs:
            if k not in seen_outputs:
                filtered_inputs.append(k)
            if k in all_outputs:
                seen_outputs.add(k)
        outer_inputs = filtered_inputs

        # 创建合并后的执行函数
        captured_stages = group_stages

        def make_merged_fn(stages_list):
            def merged_execute(**kwargs):
                from .pipeline_dataflow import DataRegistry
                reg = DataRegistry()
                for k, v in kwargs.items():
                    reg.register(k, v, source='merged_input')
                for s in stages_list:
                    s.execute(reg)
                outputs = {}
                for k in stages_list[-1].output_keys:
                    if reg.has(k):
                        outputs[k] = reg.get(k)
                return outputs
            return merged_execute

        merged.append(PipelineStage(
            name='+'.join(group),
            input_keys=outer_inputs,
            output_keys=outer_outputs,
            execute_fn=make_merged_fn(captured_stages),
            description=f"Merged: {' → '.join(group)}",
        ))

    return merged