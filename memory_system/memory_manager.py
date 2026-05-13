"""
MemoryManager – top-level orchestrator that ties together all memory
sub-systems (indexer, store, retriever, aligner, fusion).

Typical lifecycle
-----------------

1. **Initialise** the manager (once per application session)::

       manager = MemoryManager(embed_dim=512, index_dim=128)

2. **During / after inference**, form a memory from the interaction::

       manager.form_memory(
           arm_feature=F_a,
           point_cloud=points,
           point_features=F_p,
           preference_matrix=pref,
           reward=1.0,
           outcome="success",
           ...
       )

3. **Before / during the next inference**, retrieve and fuse memories::

       pref_fused = manager.retrieve_and_fuse(
           arm_feature=F_a,
           current_point_cloud=pts,
           current_point_features=Fp,
           top_k=5,
       )
       # Apply to model output
       final = sigmoid(raw_output + alpha * pref_fused)

4. **Persist** the memory store across sessions::

       manager.save()
       # Later ...
       manager.load()
"""

from __future__ import annotations

import os
import time
import threading
import queue
from typing import Optional, List, Dict, Any

import numpy as np
import torch

from .memory_entry import MemoryEntry
from .memory_indexer import MemoryIndexer
from .memory_store import MemoryStore
from .memory_retriever import MemoryRetriever
from .memory_aligner import MemoryAligner
from .memory_fusion import MemoryFusion


class MemoryManager:
    """Top-level orchestrator for the external memory system.

    Parameters
    ----------
    emb_dim : int
        Channel dimension of ARM features (``C`` in the IAG model).
    index_dim : int
        Dimension of the L2-normalised index vector.
    feat_dim : int
        Dimension of per-point features used for alignment.
        Typically equal to ``emb_dim``.
    store_dir : str
        Directory for persistent storage.
    max_memories : int
        Soft cap on the number of stored memories.
    default_top_k : int
        Default number of memories to retrieve.
    fusion_temperature : float
        Softmax temperature for reward-weighted fusion.
    fusion_alpha : float
        Residual scaling factor when applying fused preference.
    time_decay_lambda : float
        Recency factor for fusion weighting.
    use_faiss : bool
        Try to use FAISS for ANN search.
    async_formation : bool
        If True, memory formation is offloaded to a background thread
        so that it does not block the inference pipeline.
    """

    def __init__(
        self,
        emb_dim: int = 512,
        index_dim: int = 128,
        feat_dim: int = 512,
        store_dir: str = "./memory_store",
        max_memories: int = 5000,
        default_top_k: int = 5,
        fusion_temperature: float = 1.0,
        fusion_alpha: float = 0.3,
        time_decay_lambda: float = 0.0,
        use_faiss: bool = True,
        async_formation: bool = True,
    ):
        # ── Sub-systems ─────────────────────────────────────────────────
        self.indexer = MemoryIndexer(emb_dim=emb_dim, index_dim=index_dim)
        self.store = MemoryStore(
            store_dir=store_dir,
            index_dim=index_dim,
            use_faiss=use_faiss,
            max_memories=max_memories,
        )
        self.retriever = MemoryRetriever(
            store=self.store,
            default_top_k=default_top_k,
        )
        self.aligner = MemoryAligner(feat_dim=feat_dim)
        self.fusion = MemoryFusion(
            temperature=fusion_temperature,
            time_decay_lambda=time_decay_lambda,
        )

        # ── Configuration ───────────────────────────────────────────────
        self.fusion_alpha = fusion_alpha
        self.default_top_k = default_top_k

        # ── Async formation queue ───────────────────────────────────────
        self._async_formation = async_formation
        self._formation_queue: queue.Queue = queue.Queue()
        self._formation_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()

        if async_formation:
            self._start_formation_thread()

    # ==================================================================
    # Memory formation
    # ==================================================================

    def form_memory(
        self,
        arm_feature: torch.Tensor,
        point_cloud: np.ndarray,
        point_features: np.ndarray,
        preference_matrix: np.ndarray,
        reward: float,
        action_parameters: Optional[Dict[str, Any]] = None,
        outcome: str = "unknown",
        object_category: str = "",
        affordance_label: str = "",
        confidence: float = 0.0,
        text_embedding: Optional[np.ndarray] = None,
        scene_image: Optional[np.ndarray] = None,
    ) -> str:
        """从完成的交互形成并存储记忆。

        此方法可以同步或异步调用（由``async_formation``控制）。

        参数:
            arm_feature: torch.Tensor - ARM输出特征，形状[1, N_p + N_i, C]
            point_cloud: np.ndarray - 点云坐标，形状[N_raw, 3]
            point_features: np.ndarray - 每点特征（例如来自PointNet++或JRA），形状[N_raw, D_feat]
            preference_matrix: np.ndarray - 每点偏好，形状[N_raw]
            reward: float - 交互的标量奖励
            action_parameters: dict, optional - 动作参数
            outcome: str - 结果
            object_category: str - 对象类别
            affordance_label: str - affordance标签
            confidence: float - 置信度
            text_embedding: np.ndarray, optional - 用于IAG_TextEmb集成
            scene_image: np.ndarray, optional - 场景图像

        返回:
            str - 新创建的MemoryEntry的id
        """
        # Generate index vector
        with torch.no_grad():
            index_vector = self.indexer.compute_index_numpy(arm_feature)

        # Flatten to 1-D if needed
        if index_vector.ndim > 1:
            index_vector = index_vector.squeeze()

        entry = MemoryEntry(
            index_vector=index_vector,
            point_cloud=point_cloud,
            point_features=point_features,
            scene_image=scene_image,
            preference_matrix=preference_matrix,
            reward=reward,
            action_parameters=action_parameters or {},
            outcome=outcome,
            object_category=object_category,
            affordance_label=affordance_label,
            confidence=confidence,
            text_embedding=text_embedding,
        )

        if self._async_formation:
            self._formation_queue.put(entry)
        else:
            self._store_entry(entry)

        return entry.id

    # ==================================================================
    # Memory retrieval and fusion
    # ==================================================================

    def retrieve_and_fuse(
        self,
        arm_feature: torch.Tensor,
        current_point_cloud: np.ndarray,
        current_point_features: np.ndarray,
        top_k: Optional[int] = None,
        affordance_label: Optional[str] = None,
        object_category: Optional[str] = None,
        use_torch: bool = True,
    ) -> np.ndarray:
        """检索相关记忆、对齐并融合偏好。

        参数:
            arm_feature: torch.Tensor - 当前场景的ARM输出，[1, N_p + N_i, C]
            current_point_cloud: np.ndarray - 当前点云，[N_curr, 3]
            current_point_features: np.ndarray - 当前每点特征，[N_curr, D_feat]
            top_k: int, optional - 覆盖默认的top_k
            affordance_label: str, optional - 按affordance过滤记忆
            object_category: str, optional - 对相同类别对象的软偏好
            use_torch: bool - 如果为True，使用基于Torch的对齐（GPU上更快）

        返回:
            np.ndarray - 融合偏好向量，形状[N_curr]。如果没有找到相关记忆则返回零向量
        """
        N_curr = current_point_cloud.shape[0]

        # 1. Generate query index vector
        with torch.no_grad():
            query_vector = self.indexer.compute_index_numpy(arm_feature)
        if query_vector.ndim > 1:
            query_vector = query_vector.squeeze()

        # 2. Retrieve memories
        entries, similarities = self.retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k or self.default_top_k,
            affordance_label=affordance_label,
            object_category=object_category,
        )

        if not entries:
            # No relevant memories found – return zero preference
            return np.zeros(N_curr, dtype=np.float32)

        # 3. Align each memory's preference onto the current point cloud
        aligned_prefs: List[np.ndarray] = []
        rewards: List[float] = []
        confidences: List[float] = []
        timestamps: List[float] = []

        if use_torch and torch.cuda.is_available():
            F_curr_t = torch.from_numpy(current_point_features).float().cuda()
            for entry in entries:
                F_hist_t = torch.from_numpy(entry.point_features).float().cuda()
                Pref_hist_t = torch.from_numpy(entry.preference_matrix).float().cuda()
                with torch.no_grad():
                    Pref_curr_t = self.aligner(F_curr_t, F_hist_t, Pref_hist_t)
                aligned_prefs.append(Pref_curr_t.cpu().numpy())
                rewards.append(entry.reward)
                confidences.append(entry.confidence)
                timestamps.append(entry.timestamp)
        else:
            for entry in entries:
                Pref_curr = self.aligner.align_numpy(
                    current_point_features,
                    entry.point_features,
                    entry.preference_matrix,
                )
                aligned_prefs.append(Pref_curr)
                rewards.append(entry.reward)
                confidences.append(entry.confidence)
                timestamps.append(entry.timestamp)

        # 4. Fuse
        pref_fused = self.fusion.fuse_numpy(
            aligned_prefs=aligned_prefs,
            rewards=rewards,
            confidences=confidences,
            timestamps=timestamps,
            current_time=time.time(),
        )

        return pref_fused

    def apply_memory_to_output(
        self,
        raw_output: np.ndarray,
        pref_fused: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """将融合偏好作为残差应用到模型输出。

        便捷方法，委托给MemoryFusion.apply_to_output。
        """
        a = alpha if alpha is not None else self.fusion_alpha
        return MemoryFusion.apply_to_output(raw_output, pref_fused, alpha=a)

    # ==================================================================
    # Convenience: one-step memory-enhanced inference
    # ==================================================================

    def enhance_prediction(
        self,
        model_raw_output: np.ndarray,
        arm_feature: torch.Tensor,
        current_point_cloud: np.ndarray,
        current_point_features: np.ndarray,
        affordance_label: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """端到端：检索→对齐→融合→应用到模型输出。

        参数:
            model_raw_output: np.ndarray - 模型的原始logits（sigmoid之前），形状[N_raw, 1]或[1, N_raw, 1]
            arm_feature: torch.Tensor - 当前场景的ARM输出
            current_point_cloud: np.ndarray - 当前点云，[N_raw, 3]
            current_point_features: np.ndarray - 当前每点特征，[N_raw, D]
            affordance_label: str, optional
            alpha: float, optional

        返回:
            np.ndarray - 记忆增强的logits（仍在sigmoid之前）
        """
        pref_fused = self.retrieve_and_fuse(
            arm_feature=arm_feature,
            current_point_cloud=current_point_cloud,
            current_point_features=current_point_features,
            affordance_label=affordance_label,
        )
        return self.apply_memory_to_output(model_raw_output, pref_fused, alpha=alpha)

    # ==================================================================
    # Preference matrix generation utilities
    # ==================================================================

    @staticmethod
    def generate_preference_from_ground_truth(
        gt_label: np.ndarray,
        reward: float = 1.0,
    ) -> np.ndarray:
        """从ground-truth affordance标签生成偏好矩阵。

        这在*训练*或*离线记忆预填充*期间很有用，我们有ground-truth注释。

        参数:
            gt_label: np.ndarray - 二进制affordance标签，形状[N]。值为0（非affordance）或1（affordance）
            reward: float - 分配给affordance区域的奖励

        返回:
            np.ndarray - 偏好值：affordance区域为reward，其他地方为0
        """
        pref = np.zeros_like(gt_label, dtype=np.float32)
        pref[gt_label > 0.5] = reward
        return pref

    @staticmethod
    def generate_preference_from_prediction(
        prediction: np.ndarray,
        gt_label: np.ndarray,
        positive_reward: float = 1.0,
        negative_reward: float = -1.0,
        confidence_threshold: float = 0.7,
    ) -> np.ndarray:
        """从模型预测vs. ground truth生成偏好矩阵。

        这启用*自监督*记忆形成：
        - 正确预测高置信度affordance → 正奖励
        - 错误预测高置信度affordance → 负奖励
        - 低置信度预测 → 零（无偏好）

        参数:
            prediction: np.ndarray - 模型的sigmoid输出，形状[N]，值在[0, 1]范围内
            gt_label: np.ndarray - 二进制ground truth，形状[N]
            positive_reward: float
            negative_reward: float
            confidence_threshold: float - 高于此的预测被认为是"置信的"

        返回:
            np.ndarray - 形状[N]
        """
        pref = np.zeros_like(prediction, dtype=np.float32)
        confident_mask = (prediction > confidence_threshold) | (prediction < (1 - confidence_threshold))
        correct_mask = (prediction > 0.5) == (gt_label > 0.5)

        # Confident & correct → positive
        pos_mask = confident_mask & correct_mask
        pref[pos_mask] = positive_reward

        # Confident & incorrect → negative
        neg_mask = confident_mask & ~correct_mask
        pref[neg_mask] = negative_reward

        return pref

    @staticmethod
    def generate_preference_from_interaction(
        point_cloud: np.ndarray,
        interaction_center_3d: np.ndarray,
        interaction_radius: float = 0.1,
        reward: float = 1.0,
    ) -> np.ndarray:
        """从3D交互点生成偏好矩阵。

        当机器人与对象进行物理交互时，交互发生在特定3D位置。近该位置的点接收奖励值。

        参数:
            point_cloud: np.ndarray - 点云坐标，[N, 3]
            interaction_center_3d: np.ndarray - 交互的3D中心，[3]
            interaction_radius: float - 围绕中心的球形半径，用于标记为交互区域（归一化坐标单位）
            reward: float

        返回:
            np.ndarray - 形状[N]
        """
        dists = np.linalg.norm(point_cloud - interaction_center_3d, axis=1)
        pref = np.zeros(len(point_cloud), dtype=np.float32)
        pref[dists < interaction_radius] = reward
        return pref

    # ==================================================================
    # Store management
    # ==================================================================

    def get_stats(self) -> Dict[str, Any]:
        """返回关于记忆存储的统计信息。"""
        count = self.store.count()
        return {
            "total_memories": count,
            "max_memories": self.store.max_memories,
            "usage_pct": count / max(self.store.max_memories, 1) * 100,
            "index_dim": self.store.index_dim,
            "use_faiss": self.store._use_faiss,
        }

    def save(self):
        """将记忆存储持久化到磁盘。"""
        self.store.save_index()

    def load(self):
        """从磁盘加载记忆存储。"""
        self.store.load_index()

    def clear(self):
        """删除所有记忆。"""
        self.store.clear()

    # ==================================================================
    # Async formation thread
    # ==================================================================

    def _start_formation_thread(self):
        """启动后台记忆形成线程。"""
        self._formation_thread = threading.Thread(
            target=self._formation_worker,
            daemon=True,
            name="MemoryFormation",
        )
        self._formation_thread.start()

    def _formation_worker(self):
        """后台工作者，耗尽形成队列。"""
        while not self._shutdown_flag.is_set():
            try:
                entry = self._formation_queue.get(timeout=1.0)
                self._store_entry(entry)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[MemoryManager] Formation error: {e}")

    def _store_entry(self, entry: MemoryEntry):
        """实际存储条目（从形成线程或直接调用）。"""
        try:
            self.store.add(entry)
        except Exception as e:
            print(f"[MemoryManager] Store error: {e}")

    def shutdown(self):
        """优雅地关闭异步形成线程。"""
        self._shutdown_flag.set()
        if self._formation_thread and self._formation_thread.is_alive():
            self._formation_thread.join(timeout=5.0)