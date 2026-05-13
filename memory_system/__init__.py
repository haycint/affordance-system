"""
Memory System for IAG Model
===========================

An external memory module that enables the IAG model to learn from past
interaction experiences.  It stores structured "memory entries" indexed by
scene-level feature vectors, retrieves relevant memories via approximate
nearest-neighbour search, aligns historical preferences onto the current
point cloud through cross-attention, and fuses them with reward-weighted
aggregation.

Architecture overview::

    ┌───────────────────────────────────────────────────────┐
    │                  MemoryManager                        │
    │  (top-level orchestrator)                             │
    │                                                       │
    │  ┌──────────────┐  ┌───────────────┐  ┌─────────────┐ │
    │  │ MemoryIndexer│  │MemoryRetriever│  │MemoryAligner│ │
    │  │  (F_a→v_idx) │  │  (K-NN FAISS) │  │(cross-attn) │ │
    │  └──────────────┘  └───────────────┘  └─────────────┘ │
    │                                                       │
    │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
    │  │ MemoryStore  │  │ MemoryFusion │  │  Eviction   │  │
    │  │(FAISS+SQLite)│  │(reward-weight│  │  (LRU+merge)│  │
    │  └──────────────┘  └──────────────┘  └─────────────┘  │
    └───────────────────────────────────────────────────────┘

Usage::

    from memory_system import MemoryManager

    manager = MemoryManager(embed_dim=512, index_dim=128)

    # During / after inference, form a memory
    manager.form_memory(
        arm_feature=F_a,           # [1, N_p+N_i, C]
        point_cloud=points,        # [N_raw, 3]
        point_features=F_p,        # [N_raw, D]
        preference_matrix=pref,    # [N_raw]
        reward=1.0,
        action_params={...},
        outcome="success",
        object_category="Mug",
        affordance_label="grasp",
        confidence=0.92,
    )

    # Before / during inference, retrieve and fuse memories
    pref_fused = manager.retrieve_and_fuse(
        arm_feature=F_a,           # [1, N_p+N_i, C]
        current_point_cloud=pts,   # [N_raw, 3]
        current_point_features=Fp, # [N_raw, D]
        top_k=5,
    )
    # pref_fused: [N_raw]  –  add as residual to model output
"""

from .memory_entry import MemoryEntry
from .memory_indexer import MemoryIndexer
from .memory_store import MemoryStore
from .memory_retriever import MemoryRetriever
from .memory_aligner import MemoryAligner
from .memory_fusion import MemoryFusion
from .memory_manager import MemoryManager
from .image_memory_store import ImageMemoryStore
from .image_memory_manager import ImageMemoryManager

__all__ = [
    "MemoryEntry",
    "MemoryIndexer",
    "MemoryStore",
    "MemoryRetriever",
    "MemoryAligner",
    "MemoryFusion",
    "MemoryManager",
    "ImageMemoryStore",
    "ImageMemoryManager",
]
