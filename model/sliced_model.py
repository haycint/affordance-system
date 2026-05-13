"""
SlicedModel — IAG / IAG_TextEmb 模型的 API 切片推理层

本模块将 IAG 和 IAG_TextEmb 模型的推理过程切分为可流式响应的阶段，
专为前端 API 响应场景设计。

核心能力:
  - predict():  标准一次性推理，返回完整结果
  - predict_streaming():  生成器模式，每完成一个阶段即 yield 中间结果
  - predict_to_stage():  推理到指定阶段后暂停，返回中间状态
  - predict_from_stage():  从保存的中间状态恢复推理
  - predict_with_memory():  集成记忆系统的推理

典型 API 使用场景:
  1. 前端请求推理 → 后端调用 predict_streaming() → 每阶段通过 WebSocket 推送进度
  2. 前端请求低延迟预览 → 后端调用 predict_to_stage('arm') → 快速返回中间结果
  3. 前端分步交互 → 先 predict_to_stage() 获取中间结果 → 用户确认后 predict_from_stage() 继续

数据流阶段 (7 阶段):
  Stage 0: Img_Encoder      — 图像特征提取 (ResNet-18)
  Stage 1: Mask Feature     — 掩码特征提取 (主体/客体/场景)
  Stage 2: ROI Align        — 场景特征裁剪
  Stage 3: Point_Encoder    — 点云编码 (PointNet++ MSG)
  Stage 4: JRA              — 联合区域对齐
  Stage 5: ARM              — 可供性揭示模块
  Stage 6: Decoder          — 解码输出 (3D affordance + logits)
"""

import time
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import Dict, List, Optional, Any, Generator, Union
from datetime import datetime

from .pipeline_dataflow import (
    DataRegistry,
    PipelineStage,
    DataFlowPipeline,
    PipelineSerializer,
)
from .MyNet import MyNet, IAG_TextEmb


# ============================================================
# 阶段 API 响应函数 — 每个阶段完成后返回给前端的数据
# ============================================================

AFFORDANCE_LABELS = [
    "grasp", "contain", "lift", "open", "lay", "sit", "support",
    "wrapgrasp", "pour", "move", "display", "push", "listen",
    "wear", "press", "cut", "stab",
]


def _img_encoder_api_response(F_I):
    """Stage 0: 图像编码器完成后的 API 响应。"""
    return {
        'feature_map': {
            'shape': list(F_I.shape),
            'stats': {
                'min': float(F_I.min()),
                'max': float(F_I.max()),
                'mean': float(F_I.mean()),
            }
        },
        'message': 'Image features extracted',
    }


def _mask_feature_api_response(F_i, F_s, F_e):
    """Stage 1: 掩码特征提取完成后的 API 响应。"""
    return {
        'object_feature': {'shape': list(F_i.shape)},
        'subject_feature': {'shape': list(F_s.shape)},
        'scene_feature': {'shape': list(F_e.shape)},
        'message': 'Mask features extracted (object/subject/scene)',
    }


def _roi_align_api_response(F_e_aligned):
    """Stage 2: ROI Align 完成后的 API 响应。"""
    return {
        'aligned_feature': {'shape': list(F_e_aligned.shape)},
        'message': 'ROI aligned scene feature',
    }


def _point_encoder_api_response(F_p_wise):
    """Stage 3: 点云编码完成后的 API 响应。"""
    level_shapes = []
    for level in F_p_wise:
        xyz, feat = level
        level_shapes.append({
            'xyz_shape': list(xyz.shape),
            'feat_shape': list(feat.shape),
        })
    return {
        'hierarchy_levels': level_shapes,
        'num_levels': len(F_p_wise),
        'message': 'Point cloud encoded (4-level hierarchy)',
    }


def _jra_api_response(F_j):
    """Stage 4: JRA 完成后的 API 响应。"""
    return {
        'joint_feature': {
            'shape': list(F_j.shape),
            'stats': {
                'min': float(F_j.min()),
                'max': float(F_j.max()),
                'mean': float(F_j.mean()),
            }
        },
        'message': 'Joint region alignment completed',
    }


def _arm_api_response(affordance):
    """Stage 5: ARM 完成后的 API 响应。"""
    return {
        'affordance_feature': {
            'shape': list(affordance.shape),
            'stats': {
                'min': float(affordance.min()),
                'max': float(affordance.max()),
                'mean': float(affordance.mean()),
            }
        },
        'message': 'Affordance revealed',
    }


def _decoder_api_response(_3daffordance, logits, to_KL):
    """Stage 6: Decoder 完成后的 API 响应 — 最终预测结果。"""
    probs = F.softmax(logits, dim=-1)
    pred_class = torch.argmax(probs, dim=-1)
    confidence = torch.max(probs, dim=-1).values

    # 3D affordance 统计
    affordance_np = _3daffordance.squeeze(-1).cpu().numpy()
    active_ratio = float((affordance_np > 0.5).mean())

    return {
        'affordance_3d': {
            'shape': list(_3daffordance.shape),
            'active_ratio': active_ratio,
            'stats': {
                'min': float(_3daffordance.min()),
                'max': float(_3daffordance.max()),
                'mean': float(_3daffordance.mean()),
            }
        },
        'classification': {
            'logits_shape': list(logits.shape),
            'predicted_class_indices': pred_class.cpu().tolist(),
            'predicted_class_names': [
                AFFORDANCE_LABELS[idx] if idx < len(AFFORDANCE_LABELS) else f"class_{idx}"
                for idx in pred_class.cpu().tolist()
            ],
            'confidence': confidence.cpu().tolist(),
            'top5': _get_top5_predictions(probs),
        },
        'message': 'Decoding completed — final prediction',
    }


def _decoder_textemb_api_response(_3daffordance, logits, to_KL):
    """Stage 6 (TextEmb): Decoder_TextEmb 完成后的 API 响应。"""
    # 复用基础 decoder 响应，添加文本增强标记
    response = _decoder_api_response(_3daffordance, logits, to_KL)
    response['text_enhanced'] = True
    response['message'] = 'Text-enhanced decoding completed — final prediction'
    return response


def _get_top5_predictions(probs: torch.Tensor) -> List[Dict[str, Any]]:
    """获取 top-5 预测结果。"""
    batch_results = []
    for b in range(probs.size(0)):
        p = probs[b]
        top5_val, top5_idx = p.topk(min(5, p.size(0)))
        batch_results.append([
            {
                'class_index': int(top5_idx[i]),
                'class_name': AFFORDANCE_LABELS[int(top5_idx[i])]
                if int(top5_idx[i]) < len(AFFORDANCE_LABELS) else f"class_{int(top5_idx[i])}",
                'probability': float(top5_val[i]),
            }
            for i in range(len(top5_idx))
        ])
    return batch_results


# ============================================================
# 阶段执行函数 — 从模型中提取各阶段逻辑
# ============================================================

def _make_mask_feature_execute(model):
    """创建 Mask Feature 阶段的执行函数。"""
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
    return mask_feature_execute


def _make_roi_align_execute():
    """创建 ROI Align 阶段的执行函数。"""
    def roi_align_execute(F_e):
        device = F_e.device
        B = F_e.size(0)
        roi_box = []
        for i in range(B):
            roi_box.append([i, 0.0, 0.0, 6.0, 6.0])
        ROI_box = torch.tensor(roi_box).float().to(device)
        F_e_aligned = roi_align(F_e, ROI_box, output_size=(4, 4))
        return {'F_e_aligned': F_e_aligned}
    return roi_align_execute


def _make_jra_execute(model):
    """创建 JRA 阶段的执行函数。"""
    def jra_execute(F_i, F_p_wise):
        F_j = model.JRA(F_i, F_p_wise[-1][1])
        return {'F_j': F_j}
    return jra_execute


def _make_arm_execute(model):
    """创建 ARM 阶段的执行函数。"""
    def arm_execute(F_j, F_s, F_e_aligned):
        affordance = model.ARM(F_j, F_s, F_e_aligned)
        return {'affordance': affordance}
    return arm_execute


def _make_decoder_execute(model):
    """创建 IAG Decoder 阶段的执行函数。"""
    def decoder_execute(F_j, affordance, F_p_wise):
        _3daffordance, logits, to_KL = model.decoder(F_j, affordance, F_p_wise)
        return {
            '_3daffordance': _3daffordance,
            'logits': logits,
            'to_KL': to_KL,
        }
    return decoder_execute


def _make_decoder_textemb_execute(model):
    """创建 IAG_TextEmb Decoder 阶段的执行函数。"""
    def decoder_textemb_execute(F_j, affordance, F_p_wise, text_emb):
        _3daffordance, logits, to_KL = model.decoder(F_j, affordance, F_p_wise, text_emb)
        return {
            '_3daffordance': _3daffordance,
            'logits': logits,
            'to_KL': to_KL,
        }
    return decoder_textemb_execute


# ============================================================
# SlicedModel — 核心 API 切片推理类
# ============================================================

class SlicedModel:
    """
    IAG / IAG_TextEmb 模型的 API 切片推理封装。

    将模型推理过程拆分为 7 个阶段，支持:
    - 一次性推理 (predict)
    - 流式推理 (predict_streaming) — 每阶段 yield 响应
    - 阶段级暂停/恢复 (predict_to_stage / predict_from_stage)
    - 记忆增强推理 (predict_with_memory)

    Parameters
    ----------
    model : MyNet 或 IAG_TextEmb 实例
        已加载权重的模型。
    device : str
        推理设备，如 'cuda:0' 或 'cpu'。
    model_type : str
        'iag' 或 'iag_textemb'。
    num_slices : int, optional
        合并阶段数，None 则保持完整 7 阶段。
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        model_type: str = 'iag',
        num_slices: Optional[int] = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model_type = model_type
        self.num_slices = num_slices

        # 将模型移到设备
        self.model.to(self.device)
        self.model.eval()

        # 构建流水线
        if model_type == 'iag_textemb':
            self.pipeline = self._build_textemb_pipeline()
        else:
            self.pipeline = self._build_iag_pipeline()

        # 缓存上一次推理的中间状态
        self._last_registry_state: Optional[Dict[str, Any]] = None

    # ================================================================
    # 构建流水线
    # ================================================================

    def _build_iag_pipeline(self) -> DataFlowPipeline:
        """构建 IAG (MyNet) 的 7 阶段流水线。"""
        model = self.model

        stages = [
            PipelineStage(
                name='img_encoder',
                module=model.img_encoder,
                input_keys=['img'],
                output_keys=['F_I'],
                description='ResNet-18 图像特征提取',
                api_response_fn=_img_encoder_api_response,
            ),
            PipelineStage(
                name='mask_feature',
                input_keys=['img', 'F_I', 'sub_box', 'obj_box'],
                output_keys=['F_i', 'F_s', 'F_e'],
                execute_fn=_make_mask_feature_execute(model),
                description='掩码特征提取: 客体F_i + 主体F_s + 场景F_e',
                api_response_fn=_mask_feature_api_response,
            ),
            PipelineStage(
                name='roi_align',
                input_keys=['F_e'],
                output_keys=['F_e_aligned'],
                execute_fn=_make_roi_align_execute(),
                description='ROI Align 场景特征裁剪',
                api_response_fn=_roi_align_api_response,
            ),
            PipelineStage(
                name='point_encoder',
                module=model.point_encoder,
                input_keys=['xyz'],
                output_keys=['F_p_wise'],
                description='PointNet++ MSG 点云编码',
                api_response_fn=_point_encoder_api_response,
            ),
            PipelineStage(
                name='jra',
                input_keys=['F_i', 'F_p_wise'],
                output_keys=['F_j'],
                execute_fn=_make_jra_execute(model),
                description='Joint Region Alignment 联合区域对齐',
                api_response_fn=_jra_api_response,
            ),
            PipelineStage(
                name='arm',
                input_keys=['F_j', 'F_s', 'F_e_aligned'],
                output_keys=['affordance'],
                execute_fn=_make_arm_execute(model),
                description='Affordance Revealed Module 可供性揭示',
                api_response_fn=_arm_api_response,
            ),
            PipelineStage(
                name='decoder',
                input_keys=['F_j', 'affordance', 'F_p_wise'],
                output_keys=['_3daffordance', 'logits', 'to_KL'],
                execute_fn=_make_decoder_execute(model),
                description='Decoder 解码',
                api_response_fn=_decoder_api_response,
            ),
        ]

        return DataFlowPipeline(stages, name='IAG_SlicedModel')

    def _build_textemb_pipeline(self) -> DataFlowPipeline:
        """构建 IAG_TextEmb 的 7 阶段流水线。"""
        model = self.model

        stages = [
            PipelineStage(
                name='img_encoder',
                module=model.img_encoder,
                input_keys=['img'],
                output_keys=['F_I'],
                description='ResNet-18 图像特征提取',
                api_response_fn=_img_encoder_api_response,
            ),
            PipelineStage(
                name='mask_feature',
                input_keys=['img', 'F_I', 'sub_box', 'obj_box'],
                output_keys=['F_i', 'F_s', 'F_e'],
                execute_fn=_make_mask_feature_execute(model),
                description='掩码特征提取: 客体F_i + 主体F_s + 场景F_e',
                api_response_fn=_mask_feature_api_response,
            ),
            PipelineStage(
                name='roi_align',
                input_keys=['F_e'],
                output_keys=['F_e_aligned'],
                execute_fn=_make_roi_align_execute(),
                description='ROI Align 场景特征裁剪',
                api_response_fn=_roi_align_api_response,
            ),
            PipelineStage(
                name='point_encoder',
                module=model.point_encoder,
                input_keys=['xyz'],
                output_keys=['F_p_wise'],
                description='PointNet++ MSG 点云编码',
                api_response_fn=_point_encoder_api_response,
            ),
            PipelineStage(
                name='jra',
                input_keys=['F_i', 'F_p_wise'],
                output_keys=['F_j'],
                execute_fn=_make_jra_execute(model),
                description='Joint Region Alignment 联合区域对齐',
                api_response_fn=_jra_api_response,
            ),
            PipelineStage(
                name='arm',
                input_keys=['F_j', 'F_s', 'F_e_aligned'],
                output_keys=['affordance'],
                execute_fn=_make_arm_execute(model),
                description='Affordance Revealed Module 可供性揭示',
                api_response_fn=_arm_api_response,
            ),
            PipelineStage(
                name='decoder_textemb',
                input_keys=['F_j', 'affordance', 'F_p_wise', 'text_emb'],
                output_keys=['_3daffordance', 'logits', 'to_KL'],
                execute_fn=_make_decoder_textemb_execute(model),
                description='Decoder_TextEmb 文本增强解码',
                api_response_fn=_decoder_textemb_api_response,
            ),
        ]

        return DataFlowPipeline(stages, name='IAG_TextEmb_SlicedModel')

    # ================================================================
    # 输入预处理
    # ================================================================

    def _prepare_inputs(
        self,
        img: Union[torch.Tensor, np.ndarray],
        xyz: Union[torch.Tensor, np.ndarray],
        sub_box: Union[torch.Tensor, np.ndarray],
        obj_box: Union[torch.Tensor, np.ndarray],
        text_emb: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """将输入转换为流水线所需的格式并添加 batch 维度。"""
        def to_tensor(v):
            if isinstance(v, np.ndarray):
                return torch.from_numpy(v).float()
            return v.float()

        img_t = to_tensor(img).to(self.device)
        xyz_t = to_tensor(xyz).to(self.device)
        sub_box_t = to_tensor(sub_box).to(self.device)
        obj_box_t = to_tensor(obj_box).to(self.device)

        # 添加 batch 维度
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        if xyz_t.dim() == 2:
            xyz_t = xyz_t.unsqueeze(0)
        if sub_box_t.dim() == 1:
            sub_box_t = sub_box_t.unsqueeze(0)
        if obj_box_t.dim() == 1:
            obj_box_t = obj_box_t.unsqueeze(0)

        inputs = {
            'img': img_t,
            'xyz': xyz_t,
            'sub_box': sub_box_t,
            'obj_box': obj_box_t,
        }

        if text_emb is not None:
            text_emb_t = to_tensor(text_emb).to(self.device)
            if text_emb_t.dim() == 1:
                text_emb_t = text_emb_t.unsqueeze(0)
            inputs['text_emb'] = text_emb_t
        elif self.model_type == 'iag_textemb':
            # IAG_TextEmb 需要 text_emb，默认填充零向量
            text_dim = getattr(self.model, 'text_dim', 300)
            B = img_t.size(0)
            inputs['text_emb'] = torch.zeros(B, text_dim, device=self.device)

        return inputs

    # ================================================================
    # 推理接口
    # ================================================================

    @torch.no_grad()
    def predict(
        self,
        img: Union[torch.Tensor, np.ndarray],
        xyz: Union[torch.Tensor, np.ndarray],
        sub_box: Union[torch.Tensor, np.ndarray],
        obj_box: Union[torch.Tensor, np.ndarray],
        text_emb: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """标准一次性推理，返回完整结果。

        Returns
        -------
        dict:
            - affordance_3d: [B, N, 1] 3D affordance 分数
            - logits: [B, num_affordance] 分类 logits
            - predicted_class: List[str] 预测类别名
            - confidence: List[float] 置信度
            - inference_time_ms: float 推理耗时
        """
        inputs = self._prepare_inputs(img, xyz, sub_box, obj_box, text_emb)
        t0 = time.perf_counter()

        outputs = self.pipeline.run(inputs, profile=False)

        t1 = time.perf_counter()

        _3daffordance = outputs['_3daffordance']
        logits = outputs['logits']

        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1).values

        # 缓存中间状态
        self._last_registry_state = self.pipeline.registry.to_dict()

        return {
            'affordance_3d': _3daffordance,
            'logits': logits,
            'probabilities': probs,
            'predicted_class_indices': pred_class.cpu().tolist(),
            'predicted_class_names': [
                AFFORDANCE_LABELS[idx] if idx < len(AFFORDANCE_LABELS) else f"class_{idx}"
                for idx in pred_class.cpu().tolist()
            ],
            'confidence': confidence.cpu().tolist(),
            'inference_time_ms': (t1 - t0) * 1000,
        }

    @torch.no_grad()
    def predict_streaming(
        self,
        img: Union[torch.Tensor, np.ndarray],
        xyz: Union[torch.Tensor, np.ndarray],
        sub_box: Union[torch.Tensor, np.ndarray],
        obj_box: Union[torch.Tensor, np.ndarray],
        text_emb: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式推理，每完成一个阶段即 yield 中间结果。

        Yields
        ------
        dict:
            阶段响应，包含:
            - stage: str 阶段名
            - stage_index: int 阶段序号
            - total_stages: int 总阶段数
            - progress: float 0.0-1.0 进度
            - status: str 'completed'
            - data: dict 阶段特定的响应数据
            - timestamp: str ISO 时间戳
        """
        inputs = self._prepare_inputs(img, xyz, sub_box, obj_box, text_emb)

        for stage_response in self.pipeline.run_streaming(inputs, profile=True):
            yield stage_response

        # 缓存最后的 registry 状态
        self._last_registry_state = self.pipeline.registry.to_dict()

    @torch.no_grad()
    def predict_to_stage(
        self,
        stage_name: str,
        img: Union[torch.Tensor, np.ndarray],
        xyz: Union[torch.Tensor, np.ndarray],
        sub_box: Union[torch.Tensor, np.ndarray],
        obj_box: Union[torch.Tensor, np.ndarray],
        text_emb: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_serialized: bool = False,
    ) -> Dict[str, Any]:
        """推理到指定阶段后暂停，返回中间状态。

        Parameters
        ----------
        stage_name : str
            目标阶段名，如 'img_encoder', 'arm', 'decoder' 等。
        return_serialized : bool
            是否返回序列化的中间状态（可用于 API 传输或断点续推）。

        Returns
        -------
        dict:
            - stage: str 到达的阶段名
            - progress: float 进度
            - api_data: dict 阶段 API 响应数据
            - intermediate_state: dict 序列化中间状态 (return_serialized=True)
            - registry_summary: dict 注册表轻量摘要
        """
        inputs = self._prepare_inputs(img, xyz, sub_box, obj_box, text_emb)

        outputs = self.pipeline.run_until(stage_name, inputs, profile=True)

        # 找到阶段索引
        stage_idx = None
        for idx, stage in enumerate(self.pipeline.stages):
            if stage.name == stage_name:
                stage_idx = idx
                break

        # 构建该阶段的 API 响应
        stage = self.pipeline._stage_map[stage_name]
        api_data = stage.build_api_response(self.pipeline.registry)

        result = {
            'stage': stage_name,
            'stage_index': stage_idx,
            'total_stages': len(self.pipeline.stages),
            'progress': (stage_idx + 1) / len(self.pipeline.stages) if stage_idx is not None else 0.0,
            'api_data': api_data,
            'registry_summary': PipelineSerializer.serialize_for_api(self.pipeline.registry),
            'timestamp': datetime.now().isoformat(),
        }

        if return_serialized:
            # 序列化完整的中间状态（含张量数据），用于断点续推
            result['intermediate_state'] = PipelineSerializer.serialize_registry(
                self.pipeline.registry
            )

        # 缓存
        self._last_registry_state = self.pipeline.registry.to_dict()

        return result

    @torch.no_grad()
    def predict_from_stage(
        self,
        stage_name: str,
        intermediate_state: Optional[Dict[str, Any]] = None,
        text_emb: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """从保存的中间状态恢复，从指定阶段继续推理。

        Parameters
        ----------
        stage_name : str
            起始阶段名。
        intermediate_state : dict, optional
            序列化的中间状态。如果为 None，使用上次缓存的状态。
        text_emb : Tensor, optional
            文本嵌入（IAG_TextEmb 模型在 decoder 阶段需要）。

        Returns
        -------
        dict: 与 predict() 相同格式的完整推理结果。
        """
        if intermediate_state is not None:
            # 反序列化
            registry_state = PipelineSerializer.deserialize_registry(intermediate_state)
        elif self._last_registry_state is not None:
            registry_state = self._last_registry_state
        else:
            raise ValueError("No intermediate state available. Run predict_to_stage first.")

        # 如果需要补充 text_emb
        if text_emb is not None and 'text_emb' not in registry_state:
            text_emb_t = text_emb
            if isinstance(text_emb_t, np.ndarray):
                text_emb_t = torch.from_numpy(text_emb_t).float()
            if text_emb_t.dim() == 1:
                text_emb_t = text_emb_t.unsqueeze(0)
            registry_state['text_emb'] = text_emb_t.to(self.device)
        elif self.model_type == 'iag_textemb' and 'text_emb' not in registry_state:
            text_dim = getattr(self.model, 'text_dim', 300)
            B = registry_state.get('img', registry_state.get('F_I', torch.zeros(1))).size(0)
            registry_state['text_emb'] = torch.zeros(B, text_dim, device=self.device)

        outputs = self.pipeline.run_from(stage_name, registry_state, profile=True)

        _3daffordance = outputs['_3daffordance']
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1).values

        self._last_registry_state = self.pipeline.registry.to_dict()

        return {
            'affordance_3d': _3daffordance,
            'logits': logits,
            'probabilities': probs,
            'predicted_class_indices': pred_class.cpu().tolist(),
            'predicted_class_names': [
                AFFORDANCE_LABELS[idx] if idx < len(AFFORDANCE_LABELS) else f"class_{idx}"
                for idx in pred_class.cpu().tolist()
            ],
            'confidence': confidence.cpu().tolist(),
            'resumed_from_stage': stage_name,
        }

    # ================================================================
    # 记忆增强推理
    # ================================================================

    @torch.no_grad()
    def predict_with_memory(
        self,
        img: Union[torch.Tensor, np.ndarray],
        xyz: Union[torch.Tensor, np.ndarray],
        sub_box: Union[torch.Tensor, np.ndarray],
        obj_box: Union[torch.Tensor, np.ndarray],
        memory_manager=None,
        affordance_label: Optional[str] = None,
        alpha: float = 0.3,
        top_k: int = 5,
        text_emb: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """带记忆增强的推理。

        先正常推理，然后使用 MemoryManager 检索相关记忆，
        将融合的偏好向量作为残差加到模型输出上。

        Parameters
        ----------
        memory_manager : MemoryManager
            记忆系统管理器实例。
        affordance_label : str, optional
            供记忆检索过滤使用的可供性标签。
        alpha : float
            记忆残差缩放因子。
        top_k : int
            检索的最近邻记忆数。

        Returns
        -------
        dict: 与 predict() 格式相同，额外包含:
            - raw_prediction: 原始预测（记忆增强前）
            - memory_applied: bool 是否应用了记忆
            - fused_preference: 融合的偏好向量
        """
        # 先执行标准推理
        result = self.predict(img, xyz, sub_box, obj_box, text_emb)

        if memory_manager is None:
            result['raw_prediction'] = result['affordance_3d']
            result['memory_applied'] = False
            result['fused_preference'] = None
            return result

        # 获取中间特征用于记忆检索
        try:
            from .pipeline_dataflow import PipelineSerializer

            # 从缓存的 registry 中获取 ARM 特征和点云特征
            arm_feature = self._last_registry_state.get('affordance')

            # 获取点云和点特征
            xyz_t = self._prepare_inputs(img, xyz, sub_box, obj_box, text_emb)['xyz']

            # 获取点云编码的层级特征
            point_feat_raw = self._last_registry_state.get('F_p_wise')
            if point_feat_raw is not None:
                # 使用最深层的点特征
                point_features = point_feat_raw[-1][1]  # [B, C, N_p]
            else:
                point_features = None

            if arm_feature is not None and point_features is not None:
                # 转换为 MemoryManager 所需格式
                arm_feature_input = arm_feature
                point_cloud_np = xyz_t.squeeze(0).cpu().numpy().T  # [N, 3]
                point_feat_np = point_features.squeeze(0).permute(1, 0).cpu().numpy()  # [N_p, C]

                # 扩展点特征到完整点云尺寸（用零填充或最近邻插值）
                N_raw = point_cloud_np.shape[0]
                N_p = point_feat_np.shape[0]
                if N_p < N_raw:
                    # 简单重复插值
                    indices = np.linspace(0, N_p - 1, N_raw).astype(int)
                    point_feat_full = point_feat_np[indices]
                else:
                    point_feat_full = point_feat_np[:N_raw]

                pref_fused = memory_manager.retrieve_and_fuse(
                    arm_feature=arm_feature_input,
                    current_point_cloud=point_cloud_np,
                    current_point_features=point_feat_full,
                    top_k=top_k,
                    affordance_label=affordance_label,
                )

                if np.abs(pref_fused).sum() > 1e-6:
                    raw_output = result['affordance_3d'].squeeze(-1).cpu().numpy()
                    enhanced_raw = memory_manager.apply_memory_to_output(
                        raw_output, pref_fused, alpha=alpha
                    )
                    prediction = 1.0 / (1.0 + np.exp(-enhanced_raw))  # sigmoid
                    result['affordance_3d'] = torch.from_numpy(
                        prediction
                    ).unsqueeze(-1).unsqueeze(0).to(self.device)
                    result['memory_applied'] = True
                else:
                    result['memory_applied'] = False

                result['fused_preference'] = pref_fused
            else:
                result['memory_applied'] = False
                result['fused_preference'] = None

        except Exception as e:
            print(f"[SlicedModel.predict_with_memory] Memory retrieval failed: {e}")
            result['memory_applied'] = False
            result['fused_preference'] = None

        result['raw_prediction'] = result.get('raw_prediction', result['affordance_3d'])
        return result

    # ================================================================
    # 辅助方法
    # ================================================================

    def get_stage_info(self) -> List[Dict[str, Any]]:
        """获取流水线各阶段的信息。"""
        return self.pipeline.stage_info()

    def get_dataflow_graph(self) -> str:
        """获取数据流图描述。"""
        return self.pipeline.dataflow_graph()

    def get_profile_report(self) -> str:
        """获取性能分析报告。"""
        return self.pipeline.profile_report()

    def get_available_stages(self) -> List[str]:
        """获取可用的阶段名列表。"""
        return [s.name for s in self.pipeline.stages]

    def numpy_to_api_response(
        self,
        affordance_3d: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, Any]:
        """将 3D affordance 预测转换为前端可显示的 API 响应格式。"""
        if isinstance(affordance_3d, torch.Tensor):
            arr = affordance_3d.squeeze().cpu().numpy()
        else:
            arr = np.array(affordance_3d).squeeze()

        if arr.ndim > 1:
            arr = arr.squeeze()

        active_ratio = float((arr > 0.5).mean())
        active_points = int((arr > 0.5).sum())
        total_points = arr.shape[0] if arr.ndim == 1 else arr.size

        return {
            'point_scores': arr.tolist(),
            'active_ratio': active_ratio,
            'active_points': active_points,
            'total_points': total_points,
            'score_stats': {
                'min': float(arr.min()),
                'max': float(arr.max()),
                'mean': float(arr.mean()),
                'std': float(arr.std()) if arr.size > 1 else 0.0,
            },
        }

    def encode_image_for_api(
        self,
        img: Union[torch.Tensor, np.ndarray],
    ) -> str:
        """将图像张量编码为 base64 字符串用于 API 传输。"""
        import io as _io
        from PIL import Image as _Image

        if isinstance(img, torch.Tensor):
            # [C, H, W] -> [H, W, C]
            arr = img.cpu().numpy().transpose(1, 2, 0)
        else:
            arr = img

        # 归一化到 0-255
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        pil_img = _Image.fromarray(arr)
        buf = _io.BytesIO()
        pil_img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('ascii')

    def encode_pointcloud_for_api(
        self,
        xyz: Union[torch.Tensor, np.ndarray],
        scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """将点云和预测分数编码为 API 传输格式。"""
        if isinstance(xyz, torch.Tensor):
            xyz_np = xyz.cpu().numpy()
        else:
            xyz_np = np.array(xyz)

        if xyz_np.ndim == 3:
            xyz_np = xyz_np.squeeze(0)  # [3, N] or [N, 3]

        # 确保 [N, 3] 格式
        if xyz_np.shape[0] == 3 and xyz_np.shape[1] != 3:
            xyz_np = xyz_np.T

        result = {
            'points': xyz_np.tolist(),
            'num_points': xyz_np.shape[0],
        }

        if scores is not None:
            if isinstance(scores, torch.Tensor):
                scores_np = scores.squeeze().cpu().numpy()
            else:
                scores_np = np.array(scores).squeeze()
            result['scores'] = scores_np.tolist()

        return result


# ============================================================
# 工厂函数
# ============================================================

def create_sliced_model(
    model: nn.Module,
    device: str = 'cpu',
    model_type: str = 'iag',
    num_slices: Optional[int] = None,
) -> SlicedModel:
    """创建 SlicedModel 实例的工厂函数。

    Parameters
    ----------
    model : nn.Module
        已加载权重的 IAG 或 IAG_TextEmb 模型。
    device : str
        推理设备。
    model_type : str
        'iag' 或 'iag_textemb'。
    num_slices : int, optional
        合并阶段数 (2-7)。

    Returns
    -------
    SlicedModel
    """
    return SlicedModel(model=model, device=device, model_type=model_type, num_slices=num_slices)


def load_sliced_model_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu',
    model_type: str = 'iag',
    **model_kwargs,
) -> SlicedModel:
    """从检查点文件加载并创建 SlicedModel。

    Parameters
    ----------
    checkpoint_path : str
        模型检查点路径。
    device : str
        推理设备。
    model_type : str
        'iag' 或 'iag_textemb'。
    **model_kwargs
        传递给模型构造函数的额外参数。

    Returns
    -------
    SlicedModel
    """
    dev = torch.device(device)

    if model_type == 'iag_textemb':
        from .MyNet import get_IAG_TextEmb
        model = get_IAG_TextEmb(pre_train=False, **model_kwargs)
    else:
        from .MyNet import get_MyNet
        model = get_MyNet(pre_train=False, **model_kwargs)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return SlicedModel(model=model, device=device, model_type=model_type)