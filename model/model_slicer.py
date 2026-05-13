"""
Model Slicer for IAG / IAG_TextEmb
====================================

将 IAG 和 IAG_TextEmb 模型按照数据流拆分为多个阶段 (Stage)，
支持：

1. **串行推理** — 逐阶段在单设备上执行（用于分析 / 调试）
2. **流水线推理** — 多 GPU 流水线并行（加速推理）
3. **API 响应切片** — 每个阶段独立执行，前端可在任意阶段获取中间结果

模型数据流与阶段划分
--------------------

IAG (MyNet) 的 forward 数据流::

    img ──→ img_encoder ──→ F_I ──┐
                                   ├─→ get_mask_feature ──→ F_i, F_s, F_e
    sub_box, obj_box ──────────────┘          │
                                               ↓
    xyz ──→ point_encoder ──→ F_p_wise ──→ JRA ──→ F_j ──→ ARM ──→ affordance
                                                           │              │
                                                           ↓              ↓
                                                        decoder(F_j, affordance, F_p_wise)
                                                           │
                                                           ↓
                                                   _3daffordance, logits, to_KL

IAG_TextEmb 额外增加 text_emb 输入，修改 decoder 为 Decoder_TextEmb。

**阶段划分（4 阶段）：**

- **Stage 1 — 双编码器并行**：img_encoder + point_encoder
- **Stage 2 — 图像特征提取 + JRA**：get_mask_feature + JRA
- **Stage 3 — ARM**：Affordance_Revealed_Module
- **Stage 4 — 解码器**：Decoder / Decoder_TextEmb

使用方式
--------

1. 从已有模型创建切片::

    from model.model_slicer import IAGStagedModel, IAGTextEmbStagedModel
    from model.MyNet import get_MyNet, get_IAG_TextEmb

    # IAG
    model = get_MyNet(pre_train=False)
    ckpt = torch.load('best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    staged = IAGStagedModel(model)

    # IAG_TextEmb
    model_te = get_IAG_TextEmb(pre_train=False)
    ckpt_te = torch.load('textemb_best.pt', map_location='cpu', weights_only=False)
    model_te.load_state_dict(ckpt_te['model'])
    staged_te = IAGTextEmbStagedModel(model_te)

2. 逐阶段推理（调试 / 可视化中间结果）::

    img = ...      # [1, 3, H, W]
    xyz = ...      # [1, 3, N_raw]
    sub_box = ...  # [1, 4]
    obj_box = ...  # [1, 4]

    # Stage 1
    s1_out = staged.stage1(img, xyz)
    # s1_out = {F_I, F_p_wise, B, device}

    # Stage 2
    s2_out = staged.stage2(s1_out, img, sub_box, obj_box)
    # s2_out = {F_j, F_s, F_e, F_p_wise}

    # Stage 3
    s3_out = staged.stage3(s2_out)
    # s3_out = {affordance, F_j, F_s, F_e, F_p_wise}

    # Stage 4
    s4_out = staged.stage4(s3_out)
    # s4_out = {_3daffordance, logits, to_KL}

3. 一次性推理（等同于 model.forward）::

    result = staged.forward(img, xyz, sub_box, obj_box)

4. 流水线推理（多 GPU）::

    from model.model_slicer import PipelineInference

    pipeline = PipelineInference(staged, devices=['cuda:0', 'cuda:1'])
    result = pipeline(img, xyz, sub_box, obj_box)

5. API 响应切片（FastAPI 集成）::

    # 在 FastAPI 路由中
    @app.post("/api/inference/stage/{stage_id}")
    async def inference_stage(stage_id: int, req: InferenceRequest):
        ...
        if stage_id == 1:
            cache = staged.stage1(img, xyz)
            return {"stage": 1, "intermediate_keys": list(cache.keys())}
        elif stage_id == 2:
            cache = staged.stage2(prev_cache, img, sub_box, obj_box)
            return {"stage": 2, "intermediate_keys": list(cache.keys())}
        ...
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

# ── Project imports ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# Stage 1: Dual Encoders (Image + Point Cloud, parallel)
# ============================================================================

def run_stage1_encoders(
    model: nn.Module,
    img: torch.Tensor,
    xyz: torch.Tensor,
) -> Dict[str, Any]:
    """Stage 1: Run image and point cloud encoders in parallel.

    Parameters
    ----------
    model : MyNet or IAG_TextEmb
    img : [B, 3, H, W]
    xyz : [B, 3, N_raw]

    Returns
    -------
    dict with keys:
        F_I       : [B, C, h, w]    image feature map
        F_p_wise  : list of 4 (xyz, points) tuples from point_encoder
        B         : int              batch size
        device    : torch.device
    """
    B, C, N = xyz.size()
    if model.local_rank is not None:
        device = torch.device('cuda', model.local_rank)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    F_I = model.img_encoder(img)
    F_p_wise = model.point_encoder(xyz)

    return {
        'F_I': F_I,
        'F_p_wise': F_p_wise,
        'B': B,
        'device': device,
    }


# ============================================================================
# Stage 2: Mask Feature Extraction + Joint Region Alignment
# ============================================================================

def run_stage2_jra(
    model: nn.Module,
    s1_out: Dict[str, Any],
    img: torch.Tensor,
    sub_box: torch.Tensor,
    obj_box: torch.Tensor,
) -> Dict[str, Any]:
    """Stage 2: Extract masked features and run JRA.

    Parameters
    ----------
    model : MyNet or IAG_TextEmb
    s1_out : output from run_stage1_encoders
    img    : [B, 3, H, W]   (needed for get_mask_feature)
    sub_box : [B, 4]
    obj_box : [B, 4]

    Returns
    -------
    dict with keys:
        F_j      : [B, N_p + N_i, C]   joint aligned feature
        F_s      : [B, C, 4, 4]        subject feature
        F_e      : [B, C, 4, 4]        scene feature (after roi_align)
        F_p_wise : point encoder hierarchy (passed through for decoder)
    """
    F_I = s1_out['F_I']
    F_p_wise = s1_out['F_p_wise']
    B = s1_out['B']
    device = s1_out['device']

    ROI_box = model.get_roi_box(B).to(device)
    F_i, F_s, F_e = model.get_mask_feature(img, F_I, sub_box, obj_box, device)
    F_e = roi_align(F_e, ROI_box, output_size=(4, 4))

    F_j = model.JRA(F_i, F_p_wise[-1][1])

    return {
        'F_j': F_j,
        'F_s': F_s,
        'F_e': F_e,
        'F_p_wise': F_p_wise,
    }


# ============================================================================
# Stage 3: Affordance Revealed Module (ARM)
# ============================================================================

def run_stage3_arm(
    model: nn.Module,
    s2_out: Dict[str, Any],
) -> Dict[str, Any]:
    """Stage 3: Run ARM.

    Parameters
    ----------
    model : MyNet or IAG_TextEmb
    s2_out : output from run_stage2_jra

    Returns
    -------
    dict with keys:
        affordance : [B, N_p + N_i, C]   ARM output
        F_j        : passed through
        F_s        : passed through
        F_e        : passed through
        F_p_wise  : passed through
    """
    affordance = model.ARM(s2_out['F_j'], s2_out['F_s'], s2_out['F_e'])

    return {
        'affordance': affordance,
        'F_j': s2_out['F_j'],
        'F_s': s2_out['F_s'],
        'F_e': s2_out['F_e'],
        'F_p_wise': s2_out['F_p_wise'],
    }


# ============================================================================
# Stage 4: Decoder (MyNet)
# ============================================================================

def run_stage4_decoder(
    model: nn.Module,
    s3_out: Dict[str, Any],
) -> Dict[str, Any]:
    """Stage 4: Run Decoder (IAG / MyNet variant).

    Parameters
    ----------
    model : MyNet
    s3_out : output from run_stage3_arm

    Returns
    -------
    dict with keys:
        _3daffordance : [B, N_raw, 1]
        logits        : [B, num_affordance]
        to_KL         : list [F_ia, I_align]
    """
    _3daffordance, logits, to_KL = model.decoder(
        s3_out['F_j'],
        s3_out['affordance'],
        s3_out['F_p_wise'],
    )

    return {
        '_3daffordance': _3daffordance,
        'logits': logits,
        'to_KL': to_KL,
    }


# ============================================================================
# Stage 4: Decoder (IAG_TextEmb)
# ============================================================================

def run_stage4_decoder_textemb(
    model: nn.Module,
    s3_out: Dict[str, Any],
    text_emb: torch.Tensor,
) -> Dict[str, Any]:
    """Stage 4: Run Decoder_TextEmb (IAG_TextEmb variant).

    Parameters
    ----------
    model : IAG_TextEmb
    s3_out : output from run_stage3_arm
    text_emb : [B, text_dim]  GloVe embedding

    Returns
    -------
    dict with keys:
        _3daffordance : [B, N_raw, 1]
        logits        : [B, num_affordance]
        to_KL         : list [F_ia, I_align]
    """
    _3daffordance, logits, to_KL = model.decoder(
        s3_out['F_j'],
        s3_out['affordance'],
        s3_out['F_p_wise'],
        text_emb,
    )

    return {
        '_3daffordance': _3daffordance,
        'logits': logits,
        'to_KL': to_KL,
    }


# ============================================================================
# IAGStagedModel — staged inference wrapper for MyNet (IAG)
# ============================================================================

class IAGStagedModel:
    """Staged inference wrapper for IAG (MyNet).

    Provides both stage-by-stage inference and full forward pass.

    Usage
    -----
    >>> model = get_MyNet(pre_train=False)
    >>> model.load_state_dict(torch.load('best.pt')['model'])
    >>> staged = IAGStagedModel(model)
    >>>
    >>> # Full forward
    >>> result = staged(img, xyz, sub_box, obj_box)
    >>>
    >>> # Stage-by-stage
    >>> s1 = staged.stage1(img, xyz)
    >>> s2 = staged.stage2(s1, img, sub_box, obj_box)
    >>> s3 = staged.stage3(s2)
    >>> s4 = staged.stage4(s3)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def stage1(self, img: torch.Tensor, xyz: torch.Tensor) -> Dict[str, Any]:
        """Stage 1: Dual encoders (image + point cloud)."""
        return run_stage1_encoders(self.model, img, xyz)

    def stage2(
        self,
        s1_out: Dict[str, Any],
        img: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
    ) -> Dict[str, Any]:
        """Stage 2: Mask feature extraction + JRA."""
        return run_stage2_jra(self.model, s1_out, img, sub_box, obj_box)

    def stage3(self, s2_out: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: ARM."""
        return run_stage3_arm(self.model, s2_out)

    def stage4(self, s3_out: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 4: Decoder."""
        return run_stage4_decoder(self.model, s3_out)

    @torch.no_grad()
    def forward(
        self,
        img: torch.Tensor,
        xyz: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Full forward pass (equivalent to model.forward).

        Returns
        -------
        _3daffordance : [B, N_raw, 1]
        logits        : [B, num_affordance]
        to_KL         : [F_ia, I_align]
        """
        s1 = self.stage1(img, xyz)
        s2 = self.stage2(s1, img, sub_box, obj_box)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return s4['_3daffordance'], s4['logits'], s4['to_KL']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ── Convenience: get intermediate features ──────────────────────────

    def get_stage_features(
        self,
        img: torch.Tensor,
        xyz: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run all stages and return ALL intermediate results.

        Useful for visualization, debugging, and API responses.
        """
        s1 = self.stage1(img, xyz)
        s2 = self.stage2(s1, img, sub_box, obj_box)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        return {
            'stage1': s1,
            'stage2': s2,
            'stage3': s3,
            'stage4': s4,
        }


# ============================================================================
# IAGTextEmbStagedModel — staged inference wrapper for IAG_TextEmb
# ============================================================================

class IAGTextEmbStagedModel:
    """Staged inference wrapper for IAG_TextEmb.

    Same as IAGStagedModel but Stage 4 takes text_emb.

    Usage
    -----
    >>> model = get_IAG_TextEmb(pre_train=False)
    >>> model.load_state_dict(torch.load('best.pt')['model'])
    >>> staged = IAGTextEmbStagedModel(model)
    >>>
    >>> # Full forward
    >>> result = staged(img, xyz, sub_box, obj_box, text_emb)
    >>>
    >>> # Stage-by-stage
    >>> s1 = staged.stage1(img, xyz)
    >>> s2 = staged.stage2(s1, img, sub_box, obj_box)
    >>> s3 = staged.stage3(s2)
    >>> s4 = staged.stage4(s3, text_emb)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def stage1(self, img: torch.Tensor, xyz: torch.Tensor) -> Dict[str, Any]:
        """Stage 1: Dual encoders (image + point cloud)."""
        return run_stage1_encoders(self.model, img, xyz)

    def stage2(
        self,
        s1_out: Dict[str, Any],
        img: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
    ) -> Dict[str, Any]:
        """Stage 2: Mask feature extraction + JRA."""
        return run_stage2_jra(self.model, s1_out, img, sub_box, obj_box)

    def stage3(self, s2_out: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: ARM."""
        return run_stage3_arm(self.model, s2_out)

    def stage4(
        self,
        s3_out: Dict[str, Any],
        text_emb: torch.Tensor,
    ) -> Dict[str, Any]:
        """Stage 4: Decoder_TextEmb.

        Parameters
        ----------
        text_emb : [B, text_dim]  GloVe text embedding
        """
        return run_stage4_decoder_textemb(self.model, s3_out, text_emb)

    @torch.no_grad()
    def forward(
        self,
        img: torch.Tensor,
        xyz: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Full forward pass (equivalent to model.forward)."""
        s1 = self.stage1(img, xyz)
        s2 = self.stage2(s1, img, sub_box, obj_box)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3, text_emb)
        return s4['_3daffordance'], s4['logits'], s4['to_KL']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_stage_features(
        self,
        img: torch.Tensor,
        xyz: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run all stages and return ALL intermediate results."""
        s1 = self.stage1(img, xyz)
        s2 = self.stage2(s1, img, sub_box, obj_box)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3, text_emb)

        return {
            'stage1': s1,
            'stage2': s2,
            'stage3': s3,
            'stage4': s4,
        }


# ============================================================================
# PipelineInference — multi-GPU pipeline parallel inference
# ============================================================================

class PipelineInference:
    """Multi-GPU pipeline parallel inference for staged models.

    Assigns different stages to different GPUs.  For each micro-batch,
    the data flows through the stages in order, with each stage
    executing on its assigned device.

    Parameters
    ----------
    staged_model : IAGStagedModel or IAGTextEmbStagedModel
    devices : list of str or torch.device
        Device for each stage.  Length should be >= 2.
        Stages are assigned round-robin if fewer devices than stages.
    micro_batch_size : int
        Size of each micro-batch for pipeline parallelism.

    Example
    -------
    >>> staged = IAGStagedModel(model)
    >>> pipeline = PipelineInference(staged, devices=['cuda:0', 'cuda:1'])
    >>> result = pipeline(img, xyz, sub_box, obj_box)
    """

    NUM_STAGES = 4

    def __init__(
        self,
        staged_model,
        devices: List = None,
        micro_batch_size: int = 1,
    ):
        self.staged_model = staged_model
        self.micro_batch_size = micro_batch_size

        if devices is None:
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                devices = [f'cuda:{i}' for i in range(n_gpus)]
            else:
                devices = ['cpu']

        self.devices = [torch.device(d) for d in devices]

        # Assign stages to devices (round-robin)
        self.stage_devices = [
            self.devices[i % len(self.devices)]
            for i in range(self.NUM_STAGES)
        ]

        print(f"[PipelineInference] Stage-device assignment:")
        for i, dev in enumerate(self.stage_devices):
            print(f"  Stage {i+1} → {dev}")

        # Move model sub-modules to their respective stage devices
        self._distribute_model()

    def _distribute_model(self):
        """Move model sub-modules to their respective stage devices."""
        model = self.staged_model.model

        # Stage 1: encoders
        dev1 = self.stage_devices[0]
        model.img_encoder = model.img_encoder.to(dev1)
        model.point_encoder = model.point_encoder.to(dev1)

        # Stage 2: JRA (also needs get_mask_feature which uses img_encoder output)
        dev2 = self.stage_devices[1]
        model.JRA = model.JRA.to(dev2)

        # Stage 3: ARM
        dev3 = self.stage_devices[2]
        model.ARM = model.ARM.to(dev3)

        # Stage 4: decoder
        dev4 = self.stage_devices[3]
        model.decoder = model.decoder.to(dev4)

    @torch.no_grad()
    def forward_iag(
        self,
        img: torch.Tensor,
        xyz: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Pipeline forward for IAG (MyNet)."""
        B = img.shape[0]
        results = []

        # Split into micro-batches
        for mb_start in range(0, B, self.micro_batch_size):
            mb_end = min(mb_start + self.micro_batch_size, B)

            mb_img = img[mb_start:mb_end]
            mb_xyz = xyz[mb_start:mb_end]
            mb_sub = sub_box[mb_start:mb_end]
            mb_obj = obj_box[mb_start:mb_end]

            # Stage 1 on device 0
            dev1 = self.stage_devices[0]
            s1 = self.staged_model.stage1(
                mb_img.to(dev1), mb_xyz.to(dev1))

            # Move intermediates to device 1
            dev2 = self.stage_devices[1]
            s1_moved = self._move_intermediate(s1, dev2)
            s2 = self.staged_model.stage2(
                s1_moved,
                mb_img.to(dev2),
                mb_sub.to(dev2),
                mb_obj.to(dev2),
            )

            # Stage 3 on device 2
            dev3 = self.stage_devices[2]
            s2_moved = self._move_intermediate(s2, dev3)
            s3 = self.staged_model.stage3(s2_moved)

            # Stage 4 on device 3
            dev4 = self.stage_devices[3]
            s3_moved = self._move_intermediate(s3, dev4)
            s4 = self.staged_model.stage4(s3_moved)

            results.append(s4)

        # Merge micro-batch results
        return self._merge_results(results)

    @torch.no_grad()
    def forward_iag_textemb(
        self,
        img: torch.Tensor,
        xyz: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Pipeline forward for IAG_TextEmb."""
        B = img.shape[0]
        results = []

        for mb_start in range(0, B, self.micro_batch_size):
            mb_end = min(mb_start + self.micro_batch_size, B)

            mb_img = img[mb_start:mb_end]
            mb_xyz = xyz[mb_start:mb_end]
            mb_sub = sub_box[mb_start:mb_end]
            mb_obj = obj_box[mb_start:mb_end]
            mb_text = text_emb[mb_start:mb_end]

            # Stage 1
            dev1 = self.stage_devices[0]
            s1 = self.staged_model.stage1(
                mb_img.to(dev1), mb_xyz.to(dev1))

            # Stage 2
            dev2 = self.stage_devices[1]
            s1_moved = self._move_intermediate(s1, dev2)
            s2 = self.staged_model.stage2(
                s1_moved,
                mb_img.to(dev2),
                mb_sub.to(dev2),
                mb_obj.to(dev2),
            )

            # Stage 3
            dev3 = self.stage_devices[2]
            s2_moved = self._move_intermediate(s2, dev3)
            s3 = self.staged_model.stage3(s2_moved)

            # Stage 4 (with text_emb)
            dev4 = self.stage_devices[3]
            s3_moved = self._move_intermediate(s3, dev4)
            s4 = self.staged_model.stage4(s3_moved, mb_text.to(dev4))

            results.append(s4)

        return self._merge_results(results)

    def __call__(self, *args, **kwargs):
        """Auto-dispatch based on staged model type."""
        if isinstance(self.staged_model, IAGTextEmbStagedModel):
            return self.forward_iag_textemb(*args, **kwargs)
        else:
            return self.forward_iag(*args, **kwargs)

    # ── Utility methods ─────────────────────────────────────────────────

    @staticmethod
    def _move_intermediate(state_dict: Dict[str, Any], device: torch.device):
        """Move all tensor values in a state dict to the target device."""
        moved = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device)
            elif isinstance(v, list):
                # F_p_wise: list of (xyz_tensor, points_tensor) tuples
                moved_v = []
                for item in v:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        moved_v.append((item[0].to(device), item[1].to(device)))
                    elif isinstance(item, torch.Tensor):
                        moved_v.append(item.to(device))
                    else:
                        moved_v.append(item)
                moved[k] = moved_v
            else:
                # int, device, etc. — keep as-is
                moved[k] = v if not isinstance(v, torch.device) else device
        return moved

    @staticmethod
    def _merge_results(results: List[Dict[str, Any]]):
        """Merge micro-batch results into single output."""
        all_3d = torch.cat([r['_3daffordance'] for r in results], dim=0)
        all_logits = torch.cat([r['logits'] for r in results], dim=0)
        # to_KL is a list of 2 tensors; concatenate along batch dim
        to_KL_0 = torch.cat([r['to_KL'][0] for r in results], dim=0)
        to_KL_1 = torch.cat([r['to_KL'][1] for r in results], dim=0)

        return all_3d, all_logits, [to_KL_0, to_KL_1]


# ============================================================================
# StageInfo — summary of each stage (for API responses)
# ============================================================================

def get_stage_info() -> List[Dict[str, Any]]:
    """Return structured information about each stage (for API docs)."""
    return [
        {
            'stage': 1,
            'name': 'Dual Encoders',
            'description': '并行运行图像编码器(ResNet18)和点云编码器(PointNet++ MSG)',
            'inputs': ['img [B,3,H,W]', 'xyz [B,3,N_raw]'],
            'outputs': ['F_I [B,C,h,w]', 'F_p_wise (4-level hierarchy)'],
            'modules': ['img_encoder', 'point_encoder'],
            'approx_params': '~15M',
            'approx_flops': '~3.5G',
        },
        {
            'stage': 2,
            'name': 'Feature Extraction + JRA',
            'description': 'ROI特征提取 + 联合区域对齐(JRA)',
            'inputs': ['F_I', 'sub_box [B,4]', 'obj_box [B,4]', 'F_p_wise'],
            'outputs': ['F_j [B,N_p+N_i,C]', 'F_s [B,C,4,4]', 'F_e [B,C,4,4]'],
            'modules': ['get_mask_feature', 'JRA'],
            'approx_params': '~2M',
            'approx_flops': '~1.2G',
        },
        {
            'stage': 3,
            'name': 'ARM',
            'description': '可承受性揭示模块: 跨注意力融合',
            'inputs': ['F_j', 'F_s', 'F_e'],
            'outputs': ['affordance [B,N_p+N_i,C]'],
            'modules': ['ARM'],
            'approx_params': '~1M',
            'approx_flops': '~0.8G',
        },
        {
            'stage': 4,
            'name': 'Decoder',
            'description': '特征传播解码 + 3D分割输出',
            'inputs': ['F_j', 'affordance', 'F_p_wise', '(text_emb for IAG_TextEmb)'],
            'outputs': ['_3daffordance [B,N_raw,1]', 'logits [B,17]', 'to_KL'],
            'modules': ['decoder / decoder_textemb'],
            'approx_params': '~3M',
            'approx_flops': '~2.0G',
        },
    ]


# ============================================================================
# CLI: quick test / demo
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IAG/IAG_TextEmb Model Slicer Demo")
    parser.add_argument("--model_type", type=str, default="iag",
                        choices=["iag", "iag_textemb"],
                        help="Model variant to demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for demo")
    args = parser.parse_args()

    print("=" * 70)
    print("IAG / IAG_TextEmb Model Slicer Demo")
    print("=" * 70)

    # Print stage info
    print("\nStage Information:")
    for info in get_stage_info():
        print(f"\n  Stage {info['stage']}: {info['name']}")
        print(f"    {info['description']}")
        print(f"    Inputs:  {info['inputs']}")
        print(f"    Outputs: {info['outputs']}")
        print(f"    Modules: {info['modules']}")

    # Create model and test staged inference
    print("\n" + "=" * 70)
    print("Creating model and running staged inference test...")
    print("=" * 70)

    device = torch.device(args.device)

    if args.model_type == "iag":
        from model.MyNet import get_MyNet
        model = get_MyNet(pre_train=False)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt.get('model', ckpt))
        model = model.to(device).eval()
        staged = IAGStagedModel(model)

        # Create dummy inputs
        B = 2
        img = torch.randn(B, 3, 224, 224).to(device)
        xyz = torch.randn(B, 3, 2048).to(device)
        sub_box = torch.tensor([[10, 20, 100, 200], [15, 25, 110, 210]]).float().to(device)
        obj_box = torch.tensor([[30, 40, 150, 180], [35, 45, 160, 190]]).float().to(device)

        print(f"\nInput shapes: img={img.shape}, xyz={xyz.shape}")
        print(f"  sub_box={sub_box.shape}, obj_box={obj_box.shape}")

        # Stage-by-stage
        print("\n--- Stage 1: Dual Encoders ---")
        s1 = staged.stage1(img, xyz)
        print(f"  F_I shape: {s1['F_I'].shape}")
        print(f"  F_p_wise levels: {len(s1['F_p_wise'])}")

        print("\n--- Stage 2: JRA ---")
        s2 = staged.stage2(s1, img, sub_box, obj_box)
        print(f"  F_j shape: {s2['F_j'].shape}")

        print("\n--- Stage 3: ARM ---")
        s3 = staged.stage3(s2)
        print(f"  affordance shape: {s3['affordance'].shape}")

        print("\n--- Stage 4: Decoder ---")
        s4 = staged.stage4(s3)
        print(f"  _3daffordance shape: {s4['_3daffordance'].shape}")
        print(f"  logits shape: {s4['logits'].shape}")

        # Verify equivalence with full forward
        print("\n--- Verification ---")
        full_out = model(img, xyz, sub_box, obj_box)
        staged_out = staged(img, xyz, sub_box, obj_box)
        match_3d = torch.allclose(full_out[0], staged_out[0], rtol=1e-3, atol=1e-5)
        match_logits = torch.allclose(full_out[1], staged_out[1], rtol=1e-3, atol=1e-5)
        print(f"  _3daffordance match: {match_3d}")
        print(f"  logits match:        {match_logits}")

    else:  # iag_textemb
        from model.MyNet import get_IAG_TextEmb
        model = get_IAG_TextEmb(pre_train=False)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt.get('model', ckpt))
        model = model.to(device).eval()
        staged = IAGTextEmbStagedModel(model)

        B = 2
        img = torch.randn(B, 3, 224, 224).to(device)
        xyz = torch.randn(B, 3, 2048).to(device)
        sub_box = torch.tensor([[10, 20, 100, 200], [15, 25, 110, 210]]).float().to(device)
        obj_box = torch.tensor([[30, 40, 150, 180], [35, 45, 160, 190]]).float().to(device)
        text_emb = torch.randn(B, 300).to(device)

        print(f"\nInput shapes: img={img.shape}, xyz={xyz.shape}, text_emb={text_emb.shape}")

        # Stage-by-stage
        s1 = staged.stage1(img, xyz)
        print(f"Stage 1 → F_I: {s1['F_I'].shape}")

        s2 = staged.stage2(s1, img, sub_box, obj_box)
        print(f"Stage 2 → F_j: {s2['F_j'].shape}")

        s3 = staged.stage3(s2)
        print(f"Stage 3 → affordance: {s3['affordance'].shape}")

        s4 = staged.stage4(s3, text_emb)
        print(f"Stage 4 → _3daffordance: {s4['_3daffordance'].shape}, logits: {s4['logits'].shape}")

        # Verify
        full_out = model(img, xyz, sub_box, obj_box, text_emb)
        staged_out = staged(img, xyz, sub_box, obj_box, text_emb)
        match_3d = torch.allclose(full_out[0], staged_out[0], rtol=1e-3, atol=1e-5)
        print(f"\nVerification: _3daffordance match = {match_3d}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
