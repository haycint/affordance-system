"""
MemoryRetriever – high-level retrieval interface that wraps
:class:`MemoryStore` and adds optional post-filtering (by affordance
label, object category, outcome) before passing results to the aligner.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .memory_entry import MemoryEntry
from .memory_store import MemoryStore


class MemoryRetriever:
    """Retrieve relevant memories from the store with optional filtering.

    Parameters
    ----------
    store : MemoryStore
        The backing dual-path store.
    default_top_k : int
        Default number of nearest neighbours to retrieve.
    filter_by_affordance : bool
        If True, only return memories whose ``affordance_label``
        matches the query affordance (when provided).
    filter_by_outcome : Optional[str]
        If set, only return memories with this outcome
        (e.g. ``"success"``).  ``None`` disables the filter.
    min_reward : float
        Minimum reward threshold; memories below this are discarded.
    similarity_threshold : float
        Minimum cosine similarity for a memory to be considered
        relevant.  Memories with similarity below this are dropped.
    """

    def __init__(
        self,
        store: MemoryStore,
        default_top_k: int = 10,
        filter_by_affordance: bool = True,
        filter_by_outcome: Optional[str] = None,
        min_reward: float = -1.0,
        similarity_threshold: float = 0.0,
    ):
        self.store = store
        self.default_top_k = default_top_k
        self.filter_by_affordance = filter_by_affordance
        self.filter_by_outcome = filter_by_outcome
        self.min_reward = min_reward
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: Optional[int] = None,
        affordance_label: Optional[str] = None,
        object_category: Optional[str] = None,
    ) -> Tuple[List[MemoryEntry], List[float]]:
        """为查询检索最相关的记忆。

        参数:
            query_vector: np.ndarray - L2归一化的索引向量，形状[D]
            top_k: int, optional - 覆盖默认的top_k
            affordance_label: str, optional - 如果提供且filter_by_affordance为True，只返回具有此affordance标签的记忆
            object_category: str, optional - 如果提供，优先选择相同对象类别的记忆（软偏好，不是硬过滤）

        返回:
            Tuple[List[MemoryEntry], List[float]] - (过滤和排序的记忆条目列表, 对应的相似度/距离分数列表)
        """
        k = top_k or self.default_top_k

        # Over-retrieve to compensate for filtering
        # Fetch 3x more than needed, then filter down
        raw_k = min(k * 3, self.store.count())
        if raw_k == 0:
            return [], []

        distances, raw_entries = self.store.search(query_vector, raw_k)

        if not raw_entries:
            return [], []

        # Post-filtering
        filtered_entries = []
        filtered_sims = []

        for entry, dist in zip(raw_entries, distances.tolist()):
            # Affordance filter
            if self.filter_by_affordance and affordance_label is not None:
                if entry.affordance_label != affordance_label:
                    continue

            # Outcome filter
            if self.filter_by_outcome is not None:
                if entry.outcome != self.filter_by_outcome:
                    continue

            # Minimum reward filter
            if entry.reward < self.min_reward:
                continue

            # Similarity threshold
            # For FAISS IndexFlatIP with L2-normalised vectors, the
            # "distance" is actually inner-product (cosine similarity).
            if dist < self.similarity_threshold:
                continue

            filtered_entries.append(entry)
            filtered_sims.append(dist)

            if len(filtered_entries) >= k:
                break

        return filtered_entries, filtered_sims