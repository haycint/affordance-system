"""
MemoryIndexer – generates compact, L2-normalised index vectors from the
ARM (Affordance Revealed Module) output feature of the IAG model.

Pipeline::

    ARM output F_a  [B, N_p + N_i, C]
         │
         ▼  global average pooling
    v_global  [B, C]
         │
         ▼  MLP projection head
    v_proj   [B, index_dim]
         │
         ▼  L2 normalisation
    v_index  [B, index_dim]   ←  used for FAISS retrieval

The projection head is a *trainable* nn.Module so that the index space
can be learned end-to-end if desired (e.g. via a contrastive loss on
same-affordance vs. different-affordance pairs).  By default the weights
are randomly initialised – this already gives reasonable retrieval because
the ARM features are highly discriminative.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MemoryIndexer(nn.Module):
    """Generate L2-normalised index vectors from ARM features.

    Parameters
    ----------
    emb_dim : int
        Channel dimension of the ARM output (``C`` in the IAG model).
    index_dim : int
        Target dimension of the index vector (default 128).
    hidden_dim : int
        Hidden size of the projection MLP (default 256).
    pooling : str
        Pooling strategy – ``"avg"`` (default) or ``"weighted"``.
        ``"weighted"`` applies learned channel attention before pooling,
        which can help the indexer focus on semantically relevant channels.
    """

    def __init__(
        self,
        emb_dim: int = 512,
        index_dim: int = 128,
        hidden_dim: int = 256,
        pooling: str = "avg",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.index_dim = index_dim
        self.pooling = pooling

        # ── Projection head ─────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, index_dim),
        )

        # ── Optional channel-attention for weighted pooling ─────────────
        if pooling == "weighted":
            self.channel_attn = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim // 4, emb_dim),
                nn.Sigmoid(),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, arm_feature: torch.Tensor) -> torch.Tensor:
        """从ARM输出特征生成索引向量。

        参数:
            arm_feature: torch.Tensor - Affordance_Revealed_Module的输出，形状[B, N_p + N_i, C]

        返回:
            torch.Tensor - L2归一化的索引向量，形状[B, index_dim]
        """
        v_global = self._pool(arm_feature)          # [B, C]
        v_proj = self.proj(v_global)                 # [B, index_dim]
        v_index = F.normalize(v_proj, p=2, dim=-1)   # L2 normalise
        return v_index

    def compute_index_numpy(self, arm_feature: torch.Tensor) -> "np.ndarray":
        """便捷方法：返回CPU上的NumPy数组形式的索引向量。

        当从非Torch代码路径调用时很有用（例如存储模块）。
        """
        import numpy as np

        with torch.no_grad():
            v = self.forward(arm_feature)
        return v.cpu().numpy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pool(self, arm_feature: torch.Tensor) -> torch.Tensor:
        """对arm_feature的序列维度进行池化。

        参数:
            arm_feature: torch.Tensor - 形状[B, S, C]

        返回:
            torch.Tensor - 形状[B, C]
        """
        if self.pooling == "avg":
            return arm_feature.mean(dim=1)

        if self.pooling == "weighted":
            # Learned channel attention
            attn = self.channel_attn(arm_feature.mean(dim=1))  # [B, C]
            weighted = arm_feature * attn.unsqueeze(1)         # [B, S, C]
            return weighted.mean(dim=1)

        raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    # ------------------------------------------------------------------
    # Training helpers (optional contrastive learning)
    # ------------------------------------------------------------------

    @staticmethod
    def contrastive_loss(
        v1: torch.Tensor,
        v2: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """用于索引向量训练的监督对比损失。

        正对具有相同的affordance标签；负对具有不同的标签。

        参数:
            v1, v2: torch.Tensor - 两批索引向量，每批形状[B, D]
            labels: torch.Tensor - 整数affordance标签，形状[B]
            temperature: float - softmax的温度缩放

        返回:
            torch.Tensor - 标量损失值
        """
        # Cosine similarity matrix
        sim = torch.mm(v1, v2.t()) / temperature          # [B, B]

        # Positive mask: same label
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.t()).float()          # [B, B]
        pos_mask.fill_diagonal_(0.0)                       # exclude self

        # Loss: log-softmax over negatives, masked by positives
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
        return loss.mean()