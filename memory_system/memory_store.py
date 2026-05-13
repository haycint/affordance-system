"""
MemoryStore – dual-path storage backed by FAISS (vector index) and
SQLite (structured data).

Design
------
* **Vector path** (FAISS): stores the ``index_vector`` of every memory for
  ultra-fast approximate nearest-neighbour (ANN) retrieval.
  Falls back to a pure-NumPy brute-force index when FAISS is not installed.
* **Structured path** (SQLite): stores the full :class:`MemoryEntry` as a
  serialised BLOB, keyed by ``entry.id``.  Both paths are linked by the
  same ``id``.

Thread safety
-------------
SQLite connections are created per-thread (``check_same_thread=False`` is
used but the caller should still avoid sharing a single connection across
threads).  FAISS / NumPy index operations hold an internal lock.
"""

from __future__ import annotations

import os
import json
import sqlite3
import threading
import time
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from .memory_entry import MemoryEntry

# ── Try importing FAISS ─────────────────────────────────────────────────
try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    faiss = None  # type: ignore


# ======================================================================
# NumPy fallback index (brute-force L2)
# ======================================================================

class _NumpyIndex:
    """Minimal brute-force L2 nearest-neighbour index.

    This is used when FAISS is not available.  It stores all vectors in
    a single matrix and computes distances with NumPy broadcasting.

    Not suitable for >100k vectors, but perfectly adequate for the
    typical scale of an affordance-memory system (hundreds to a few
    thousand entries).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: Optional[np.ndarray] = None  # [N, dim]
        self.ids: List[str] = []

    def add(self, vector: np.ndarray, entry_id: str):
        """添加单个向量到索引中。

        参数:
            vector: np.ndarray - 要添加的向量，形状任意，会被重塑为[1, -1]
            entry_id: str - 向量对应的条目ID
        """
        v = vector.reshape(1, -1).astype(np.float32)
        if self.vectors is None:
            self.vectors = v
        else:
            self.vectors = np.vstack([self.vectors, v])
        self.ids.append(entry_id)

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, List[str]]:
        """搜索最相似的top_k个向量。

        参数:
            query: np.ndarray - 查询向量，形状[D]或[1, D]
            top_k: int - 返回的最近邻数量

        返回:
            Tuple[np.ndarray, List[str]] - (距离数组[K], ID列表[K])
        """
        if self.vectors is None or len(self.ids) == 0:
            return np.array([]), []

        q = query.reshape(1, -1).astype(np.float32)
        # L2 distances
        dists = np.linalg.norm(self.vectors - q, axis=1)  # [N]
        k = min(top_k, len(self.ids))
        top_idx = np.argpartition(dists, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(dists[top_idx])]
        return dists[top_idx], [self.ids[i] for i in top_idx]

    def remove(self, entry_id: str) -> bool:
        """根据entry_id移除向量。如果找到则返回True。

        参数:
            entry_id: str - 要移除的条目ID

        返回:
            bool - 是否成功移除
        """
        if entry_id not in self.ids:
            return False
        idx = self.ids.index(entry_id)
        self.ids.pop(idx)
        if self.vectors is not None:
            self.vectors = np.delete(self.vectors, idx, axis=0)
        return True

    def __len__(self) -> int:
        return len(self.ids)


# ======================================================================
# MemoryStore
# ======================================================================

class MemoryStore:
    """Dual-path memory storage: FAISS (or NumPy) + SQLite.

    Parameters
    ----------
    store_dir : str
        Directory where the SQLite database and FAISS index file reside.
    index_dim : int
        Dimensionality of the index vectors (must match :class:`MemoryIndexer`).
    use_faiss : bool
        If True, try to use FAISS for ANN search.  Falls back to NumPy
        brute-force when FAISS is not installed.
    max_memories : int
        Soft cap on the number of stored memories.  When exceeded, the
        eviction policy (LRU) is triggered.
    """

    def __init__(
        self,
        store_dir: str = "./memory_store",
        index_dim: int = 128,
        use_faiss: bool = True,
        max_memories: int = 5000,
    ):
        self.store_dir = store_dir
        self.index_dim = index_dim
        self.max_memories = max_memories
        os.makedirs(store_dir, exist_ok=True)

        # ── Vector index ────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._use_faiss = use_faiss and _FAISS_AVAILABLE

        if self._use_faiss:
            self._faiss_index = faiss.IndexFlatIP(index_dim)  # inner product ≈ cosine after L2 norm
            self._faiss_ids: List[str] = []
            print(f"[MemoryStore] Using FAISS index (dim={index_dim})")
        else:
            self._numpy_index = _NumpyIndex(index_dim)
            print(f"[MemoryStore] Using NumPy brute-force index (dim={index_dim})")

        # ── SQLite database ─────────────────────────────────────────────
        self._db_path = os.path.join(store_dir, "memories.db")
        self._init_db()

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _init_db(self):
        """创建memories表（如果不存在）。初始化SQLite数据库结构，包括创建表和索引。"""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                data        BLOB NOT NULL,
                reward      REAL,
                outcome     TEXT,
                timestamp   REAL,
                object_category TEXT,
                affordance_label TEXT,
                confidence  REAL,
                access_count INTEGER
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_affordance
            ON memories(affordance_label)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcome
            ON memories(outcome)
        """)
        conn.commit()
        conn.close()

    def _db_insert(self, entry: MemoryEntry):
        """在SQLite中插入或替换记忆条目。

        参数:
            entry: MemoryEntry - 要插入的记忆条目
        """
        import base64

        serialised = json.dumps(entry.to_dict()).encode("utf-8")
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, data, reward, outcome, timestamp,
                 object_category, affordance_label, confidence, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                serialised,
                entry.reward,
                entry.outcome,
                entry.timestamp,
                entry.object_category,
                entry.affordance_label,
                entry.confidence,
                entry.access_count,
            ),
        )
        conn.commit()
        conn.close()

    def _db_get(self, entry_id: str) -> Optional[MemoryEntry]:
        """根据ID从SQLite检索记忆条目。

        参数:
            entry_id: str - 记忆条目的ID

        返回:
            Optional[MemoryEntry] - 检索到的记忆条目，如果不存在则返回None
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute(
            "SELECT data FROM memories WHERE id = ?", (entry_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None
        return MemoryEntry.from_dict(json.loads(row[0].decode("utf-8")))

    def _db_get_many(self, entry_ids: List[str]) -> List[MemoryEntry]:
        """从SQLite检索多个记忆条目。

        参数:
            entry_ids: List[str] - 要检索的记忆条目ID列表

        返回:
            List[MemoryEntry] - 检索到的记忆条目列表
        """
        if not entry_ids:
            return []
        conn = sqlite3.connect(self._db_path)
        placeholders = ",".join("?" for _ in entry_ids)
        cursor = conn.execute(
            f"SELECT data FROM memories WHERE id IN ({placeholders})",
            entry_ids,
        )
        rows = cursor.fetchall()
        conn.close()
        return [MemoryEntry.from_dict(json.loads(r[0].decode("utf-8"))) for r in rows]

    def _db_delete(self, entry_id: str):
        """从SQLite删除记忆条目。

        参数:
            entry_id: str - 要删除的记忆条目ID
        """
        conn = sqlite3.connect(self._db_path)
        conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        conn.commit()
        conn.close()

    def _db_increment_access(self, entry_id: str):
        """增加记忆条目的访问计数。

        参数:
            entry_id: str - 要增加访问计数的记忆条目ID
        """
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
            (entry_id,),
        )
        conn.commit()
        conn.close()

    def _db_count(self) -> int:
        """返回存储的记忆总数。

        返回:
            int - 存储的记忆条目数量
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def _db_get_all_timestamps(self) -> List[Tuple[str, float, int]]:
        """返回所有条目的(id, timestamp, access_count)。

        返回:
            List[Tuple[str, float, int]] - 包含(id, 时间戳, 访问计数)的元组列表
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT id, timestamp, access_count FROM memories")
        rows = cursor.fetchall()
        conn.close()
        return [(r[0], r[1], r[2]) for r in rows]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry):
        """在向量索引和SQLite中存储记忆条目。

        如果存储的记忆数量超过max_memories，则触发LRU（最近最少使用）驱逐策略。

        参数:
            entry: MemoryEntry - 要存储的记忆条目，必须有非空的index_vector
        """
        if entry.index_vector.size == 0:
            raise ValueError("MemoryEntry must have a non-empty index_vector")

        with self._lock:
            # Add to vector index
            if self._use_faiss:
                v = entry.index_vector.reshape(1, -1).astype(np.float32)
                self._faiss_index.add(v)
                self._faiss_ids.append(entry.id)
            else:
                self._numpy_index.add(entry.index_vector, entry.id)

        # Add to SQLite
        self._db_insert(entry)

        # Eviction check
        if self._db_count() > self.max_memories:
            self._evict_lru()

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """根据ID检索记忆条目。

        参数:
            entry_id: str - 记忆条目的ID

        返回:
            Optional[MemoryEntry] - 检索到的记忆条目，如果不存在则返回None
        """
        return self._db_get(entry_id)

    def get_many(self, entry_ids: List[str]) -> List[MemoryEntry]:
        """根据ID列表检索多个记忆条目。

        参数:
            entry_ids: List[str] - 要检索的记忆条目ID列表

        返回:
            List[MemoryEntry] - 检索到的记忆条目列表
        """
        return self._db_get_many(entry_ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[MemoryEntry]]:
        """搜索最相似的top_k个记忆。

        参数:
            query_vector: np.ndarray - L2归一化的查询向量，形状[D]
            top_k: int - 返回的最相似记忆数量

        返回:
            Tuple[np.ndarray, List[MemoryEntry]] - (相似度/距离分数[K], 按相似度降序排列的记忆条目列表)
        """
        with self._lock:
            if self._use_faiss:
                q = query_vector.reshape(1, -1).astype(np.float32)
                k = min(top_k, self._faiss_index.ntotal)
                if k == 0:
                    return np.array([]), []
                distances, indices = self._faiss_index.search(q, k)
                ids = [self._faiss_ids[i] for i in indices[0] if i >= 0]
                dists = distances[0][: len(ids)]
            else:
                dists, ids = self._numpy_index.search(query_vector, top_k)

        if not ids:
            return np.array([]), []

        entries = self._db_get_many(ids)

        # Increment access count for retrieved memories
        for entry_id in ids:
            self._db_increment_access(entry_id)

        return dists, entries

    def remove(self, entry_id: str):
        """从向量索引和SQLite中移除记忆条目。

        参数:
            entry_id: str - 要移除的记忆条目ID
        """
        with self._lock:
            if self._use_faiss:
                # FAISS does not support efficient removal from IndexFlatIP.
                # We rebuild the index without the target entry.
                if entry_id in self._faiss_ids:
                    idx = self._faiss_ids.index(entry_id)
                    self._faiss_ids.pop(idx)
                    self._rebuild_faiss_index()
            else:
                self._numpy_index.remove(entry_id)

        self._db_delete(entry_id)

    def count(self) -> int:
        """返回存储的记忆数量。

        返回:
            int - 存储的记忆条目数量
        """
        return self._db_count()

    def clear(self):
        """删除所有存储的记忆并重置索引。"""
        with self._lock:
            if self._use_faiss:
                self._faiss_index.reset()
                self._faiss_ids.clear()
            else:
                self._numpy_index = _NumpyIndex(self.index_dim)

        conn = sqlite3.connect(self._db_path)
        conn.execute("DELETE FROM memories")
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_lru(self, n: int = 1):
        """驱逐n个最近最少访问的记忆。

        平局通过时间戳打破（较旧的先驱逐）。

        参数:
            n: int - 要驱逐的记忆数量，默认1
        """
        records = self._db_get_all_timestamps()
        # Sort by access_count ascending, then timestamp ascending
        records.sort(key=lambda r: (r[2], r[1]))
        for i in range(min(n, len(records))):
            self.remove(records[i][0])

    # ------------------------------------------------------------------
    # FAISS rebuild (needed because IndexFlatIP lacks remove())
    # ------------------------------------------------------------------

    def _rebuild_faiss_index(self):
        """从SQLite中的所有剩余条目重建FAISS索引。因为IndexFlatIP不支持移除操作。"""
        if not self._use_faiss:
            return

        self._faiss_index.reset()
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT id, data FROM memories")
        new_ids = []
        vectors = []
        for row in cursor:
            entry_id = row[0]
            if entry_id in self._faiss_ids or entry_id in {
                x for x in self._faiss_ids
            }:
                entry = MemoryEntry.from_dict(json.loads(row[1].decode("utf-8")))
                if entry.index_vector.size > 0:
                    vectors.append(entry.index_vector.reshape(1, -1).astype(np.float32))
                    new_ids.append(entry_id)
        conn.close()

        if vectors:
            all_vecs = np.vstack(vectors)
            self._faiss_index.add(all_vecs)
        self._faiss_ids = new_ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: Optional[str] = None):
        """将FAISS索引保存到磁盘（SQLite始终是持久性的）。

        参数:
            path: Optional[str] - 保存路径，如果为None则使用默认路径
        """
        if not self._use_faiss:
            return
        path = path or os.path.join(self.store_dir, "faiss.index")
        faiss.write_index(self._faiss_index, path)
        # Also save the ID mapping
        id_path = path + ".ids"
        with open(id_path, "w") as f:
            for mid in self._faiss_ids:
                f.write(mid + "\n")

    def load_index(self, path: Optional[str] = None):
        """从磁盘加载FAISS索引。

        参数:
            path: Optional[str] - 加载路径，如果为None则使用默认路径
        """
        if not self._use_faiss:
            return
        path = path or os.path.join(self.store_dir, "faiss.index")
        if not os.path.exists(path):
            return
        self._faiss_index = faiss.read_index(path)
        id_path = path + ".ids"
        self._faiss_ids = []
        if os.path.exists(id_path):
            with open(id_path, "r") as f:
                self._faiss_ids = [line.strip() for line in f if line.strip()]
