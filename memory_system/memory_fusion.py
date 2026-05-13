"""
MemoryFusion – reward-weighted fusion of multiple aligned preferences.

When *K* memories are retrieved and each is aligned onto the current
point cloud, we obtain *K* aligned preference vectors
``{Pref_curr_1, …, Pref_curr_K}`` with corresponding rewards
``{r_1, …, r_K}``.  The fusion module combines them into a single
consensus preference ``Pref_fused`` that emphasises high-reward
(successful) experiences and suppresses low-reward (failed) ones.

Core equation::

    weights = softmax(rewards * temperature)    # [K]
    Pref_fused = Σ_k  w_k * Pref_curr_k         # [N_curr]

The temperature hyper-parameter controls how sharply the weight
concentrates on the highest-reward memories:

* ``temperature → ∞``: uniform weighting (all memories contribute equally)
* ``temperature → 0``: argmax (only the best memory is used)
* ``temperature = 1.0`` (default): moderate emphasis on good memories

Additional features
-------------------
* **Time decay**: optionally factor in recency so that newer memories
  receive higher weight: ``final_score = reward + λ * recency``.
* **Confidence weighting**: optionally multiply the reward by the
  model's confidence at the time the memory was formed.
* **Negative-experience suppression**: negative-reward memories
  contribute with flipped sign – they *subtract* from the fused
  preference in the failure region.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional

import numpy as np


class MemoryFusion:
    """Reward-weighted fusion of multiple aligned preferences.

    This class is stateless (no nn.Module) – it is a pure function
    module that operates on NumPy arrays or Torch tensors.

    Parameters
    ----------
    temperature : float
        Softmax temperature for reward weighting (default 1.0).
    time_decay_lambda : float
        Strength of the recency factor (default 0.0, i.e. disabled).
        When > 0, ``final_score = reward + time_decay_lambda * recency``
        where ``recency`` decays exponentially from 1 (most recent) to 0.
    use_confidence : bool
        If True, multiply reward by confidence when computing weights.
    negative_suppression : bool
        If True, negative-reward memories are applied with their sign
        (i.e. they *suppress* preference in the failure region).
        If False, negative-reward memories are simply down-weighted but
        still contribute positively in their high-preference regions.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        time_decay_lambda: float = 0.0,
        use_confidence: bool = False,
        negative_suppression: bool = True,
    ):
        self.temperature = temperature
        self.time_decay_lambda = time_decay_lambda
        self.use_confidence = use_confidence
        self.negative_suppression = negative_suppression

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------

    def fuse_torch(
        self,
        aligned_prefs: List[torch.Tensor],
        rewards: List[float],
        confidences: Optional[List[float]] = None,
        timestamps: Optional[List[float]] = None,
        current_time: Optional[float] = None,
    ) -> torch.Tensor:
        """使用Torch张量融合对齐偏好。

        参数:
            aligned_prefs: List[torch.Tensor] - 每个[N_curr]张量的列表
            rewards: List[float] - 每个记忆的奖励
            confidences: List[float], optional - 可选的置信度
            timestamps: List[float], optional - Unix时间戳
            current_time: float, optional - 用于计算最近性的当前时间

        返回:
            torch.Tensor - 形状[N_curr]的融合偏好向量
        """
        if not aligned_prefs:
            raise ValueError("aligned_prefs must not be empty")

        K = len(aligned_prefs)
        device = aligned_prefs[0].device

        # Compute fusion scores
        scores = self._compute_scores(rewards, confidences, timestamps, current_time)

        # Softmax weights
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
        weights = F.softmax(scores_tensor * self.temperature, dim=0)

        # Stack preferences: [K, N_curr]
        pref_stack = torch.stack(aligned_prefs, dim=0)

        # Weighted sum
        Pref_fused = (weights.unsqueeze(1) * pref_stack).sum(dim=0)  # [N_curr]

        return Pref_fused

    def fuse_numpy(
        self,
        aligned_prefs: List[np.ndarray],
        rewards: List[float],
        confidences: Optional[List[float]] = None,
        timestamps: Optional[List[float]] = None,
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """使用NumPy数组融合对齐偏好。

        参数:
            aligned_prefs: List[np.ndarray] - 每个[N_curr]数组的列表
            rewards: List[float] - 奖励列表
            confidences: List[float], optional - 可选的置信度
            timestamps: List[float], optional - 可选的时间戳
            current_time: float, optional - 可选的当前时间

        返回:
            np.ndarray - 形状[N_curr]
        """
        if not aligned_prefs:
            raise ValueError("aligned_prefs must not be empty")

        K = len(aligned_prefs)

        # Compute fusion scores
        scores = self._compute_scores(rewards, confidences, timestamps, current_time)

        # Softmax weights (NumPy implementation)
        scores_arr = np.array(scores, dtype=np.float32)
        scores_arr = scores_arr * self.temperature
        scores_arr -= scores_arr.max()  # numerical stability
        exp_scores = np.exp(scores_arr)
        weights = exp_scores / (exp_scores.sum() + 1e-8)

        # Weighted sum
        pref_stack = np.stack(aligned_prefs, axis=0)  # [K, N_curr]
        Pref_fused = (weights[:, np.newaxis] * pref_stack).sum(axis=0)  # [N_curr]

        return Pref_fused

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        rewards: List[float],
        confidences: Optional[List[float]],
        timestamps: Optional[List[float]],
        current_time: Optional[float],
    ) -> List[float]:
        """为每个记忆计算融合分数。

        分数决定记忆接收多少权重：
            score = reward [+ time_decay_lambda * recency] [* confidence]
        """
        K = len(rewards)
        scores = []

        for k in range(K):
            s = rewards[k]

            # Confidence weighting
            if self.use_confidence and confidences is not None:
                s *= confidences[k]

            # Time decay
            if self.time_decay_lambda > 0 and timestamps is not None and current_time is not None:
                age = current_time - timestamps[k]
                # Exponential decay: half-life of 1 hour (3600 s)
                recency = np.exp(-age / 3600.0)
                s += self.time_decay_lambda * recency

            scores.append(s)

        return scores

    # ------------------------------------------------------------------
    # Apply fused preference to model output
    # ------------------------------------------------------------------

    @staticmethod
    def apply_to_output(
        raw_output: "torch.Tensor | np.ndarray",
        pref_fused: "torch.Tensor | np.ndarray",
        alpha: float = 0.3,
    ) -> "torch.Tensor | np.ndarray":
        """将融合偏好作为残差添加到模型的原始输出。

        最终affordance预测为::

            final = sigmoid(raw_output + alpha * pref_fused)

        其中``alpha``控制记忆影响的强度。
        小的alpha（例如0.3）确保记忆是一个*温和的推动*而不是覆盖。

        参数:
            raw_output: torch.Tensor | np.ndarray - 模型的原始logits（sigmoid之前），形状[B, N, 1]或[N, 1]或[N]
            pref_fused: torch.Tensor | np.ndarray - 融合偏好向量，形状[N]
            alpha: float - 残差缩放因子

        返回:
            与raw_output相同类型和形状，已应用记忆
        """
        is_torch = isinstance(raw_output, torch.Tensor)

        if is_torch:
            pref = pref_fused.to(raw_output.device)
            if raw_output.dim() == 3:
                # [B, N, 1] → broadcast over batch
                pref = pref.unsqueeze(0).unsqueeze(-1)
            elif raw_output.dim() == 2:
                # [N, 1]
                pref = pref.unsqueeze(-1)
            return raw_output + alpha * pref
        else:
            pref = pref_fused
            if raw_output.ndim == 3:
                pref = pref[np.newaxis, :, np.newaxis]
            elif raw_output.ndim == 2:
                pref = pref[:, np.newaxis]
            return raw_output + alpha * pref
