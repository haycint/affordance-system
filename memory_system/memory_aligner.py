"""
MemoryAligner – cross-instance preference alignment via feature-space
soft matching.

Given a retrieved memory with historical preference ``Pref_hist`` over
``N_hist`` points, the aligner projects that preference onto the *current*
point cloud of ``N_curr`` points using cross-attention weights computed
from the respective per-point feature vectors.

Core equation::

    S = F_curr @ F_hist.T          # [N_curr, N_hist]  raw similarity
    A = softmax(S / sqrt(D), dim=-1)  # attention weights
    Pref_curr = A @ Pref_hist      # [N_curr]  aligned preference

This is the **central technical component** of the memory system: it
enables a preference matrix stored for *one* object instance to be
meaningfully transferred to a *different* instance with a different
number of points and a different geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import numpy as np


class MemoryAligner(nn.Module):
    """Align historical preference matrices to the current point cloud.

    Parameters
    ----------
    feat_dim : int
        Dimensionality of the per-point feature vectors (``D_feat``).
    num_heads : int
        Number of attention heads for multi-head alignment.
        More heads can capture multiple correspondences simultaneously.
    temperature : float
        Scaling factor for the attention logits.  Lower → sharper
        attention (more peaky); higher → softer (more diffuse).
    use_learned_projection : bool
        If True, project both F_curr and F_hist through a shared linear
        layer before computing similarity.  This allows the alignment to
        learn a task-specific feature space rather than relying solely on
        the raw backbone features.
    """

    def __init__(
        self,
        feat_dim: int = 512,
        num_heads: int = 4,
        temperature: float = 1.0,
        use_learned_projection: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert feat_dim % num_heads == 0, (
            f"feat_dim ({feat_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.temperature = temperature
        self.scale = self.head_dim ** (-0.5)

        # Optional shared projection to a task-specific alignment space
        self.use_learned_projection = use_learned_projection
        if use_learned_projection:
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.ReLU(inplace=True),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        F_curr: torch.Tensor,
        F_hist: torch.Tensor,
        Pref_hist: torch.Tensor,
    ) -> torch.Tensor:
        """将单个历史偏好对齐到当前点云。

        参数:
            F_curr: torch.Tensor - 当前每点特征，形状[N_curr, D]
            F_hist: torch.Tensor - 历史每点特征，形状[N_hist, D]
            Pref_hist: torch.Tensor - 历史每点偏好，形状[N_hist]（值在[-1, +1]范围内）

        返回:
            torch.Tensor - 当前点云的对齐偏好，形状[N_curr]
        """
        return self.align(F_curr, F_hist, Pref_hist)

    def align(
        self,
        F_curr: torch.Tensor,
        F_hist: torch.Tensor,
        Pref_hist: torch.Tensor,
    ) -> torch.Tensor:
        """将单个历史偏好对齐到当前点云。

        参数:
            F_curr: torch.Tensor - 形状[N_curr, D]
            F_hist: torch.Tensor - 形状[N_hist, D]
            Pref_hist: torch.Tensor - 形状[N_hist]

        返回:
            torch.Tensor - 形状[N_curr]
        """
        if self.use_learned_projection:
            F_curr_proj = self.proj(F_curr)  # [N_curr, D]
            F_hist_proj = self.proj(F_hist)  # [N_hist, D]
        else:
            F_curr_proj = F_curr
            F_hist_proj = F_hist

        # Multi-head attention alignment
        Pref_curr = self._multi_head_align(F_curr_proj, F_hist_proj, Pref_hist)
        return Pref_curr

    def align_batch(
        self,
        F_curr: torch.Tensor,
        F_hist_list: list[torch.Tensor],
        Pref_hist_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """将多个历史偏好对齐到相同的当前点云。

        参数:
            F_curr: torch.Tensor - 形状[N_curr, D]
            F_hist_list: list[torch.Tensor] - 每个检索记忆的[N_hist_k, D]
            Pref_hist_list: list[torch.Tensor] - 每个检索记忆的[N_hist_k]

        返回:
            list[torch.Tensor] - 每个检索记忆的[N_curr]对齐偏好
        """
        aligned = []
        for F_hist, Pref_hist in zip(F_hist_list, Pref_hist_list):
            aligned.append(self.align(F_curr, F_hist, Pref_hist))
        return aligned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _multi_head_align(
        self,
        F_curr: torch.Tensor,
        F_hist: torch.Tensor,
        Pref_hist: torch.Tensor,
    ) -> torch.Tensor:
        """多头交叉注意力对齐。

        每个头计算自己的注意力映射，对齐偏好在头之间取平均。
        """
        N_curr = F_curr.size(0)
        N_hist = F_hist.size(0)
        device = F_curr.device

        # Reshape into heads: [num_heads, N, head_dim]
        Q = F_curr.view(N_curr, self.num_heads, self.head_dim).transpose(0, 1)
        K = F_hist.view(N_hist, self.num_heads, self.head_dim).transpose(0, 1)

        # Attention: [num_heads, N_curr, N_hist]
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)

        # Expand Pref_hist for each head: [num_heads, N_hist, 1]
        P = Pref_hist.unsqueeze(0).unsqueeze(-1).expand(self.num_heads, -1, 1)

        # Weighted sum: [num_heads, N_curr, 1]
        Pref_per_head = torch.bmm(attn_weights, P).squeeze(-1)  # [num_heads, N_curr]

        # Average across heads
        Pref_curr = Pref_per_head.mean(dim=0)  # [N_curr]
        return Pref_curr

    # ------------------------------------------------------------------
    # NumPy convenience (for use outside torch.no_grad)
    # ------------------------------------------------------------------

    def align_numpy(
        self,
        F_curr: np.ndarray,
        F_hist: np.ndarray,
        Pref_hist: np.ndarray,
    ) -> np.ndarray:
        """基于NumPy的对齐（无梯度，无学习投影）。

        在轻量级推理或对齐器在训练图外部使用时很有用。
        """
        D = F_curr.shape[1]

        # Simple cosine-similarity based attention
        # Normalise
        F_curr_n = F_curr / (np.linalg.norm(F_curr, axis=1, keepdims=True) + 1e-8)
        F_hist_n = F_hist / (np.linalg.norm(F_hist, axis=1, keepdims=True) + 1e-8)

        # Similarity matrix: [N_curr, N_hist]
        S = F_curr_n @ F_hist_n.T / np.sqrt(D)

        # Softmax
        S_max = S.max(axis=1, keepdims=True)
        exp_S = np.exp(S - S_max)
        A = exp_S / (exp_S.sum(axis=1, keepdims=True) + 1e-8)

        # Alignment
        Pref_curr = A @ Pref_hist
        return Pref_curr
