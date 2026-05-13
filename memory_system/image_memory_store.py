"""
ImageMemoryStore -- dual-path storage for the image memory system.

Analogous to MemoryStore but specialised for *image* data:
  - SQLite stores structured metadata (object_category + affordance_label
    as composite index, plus image path and image feature BLOB)
  - On disk, raw image files are stored under ``store_dir/images/``
  - Image feature vectors (extracted by Img_Encoder) are stored as
    BLOBs for fast retrieval without re-running the encoder

Design
------
The composite key ``(object_category, affordance_label)`` is the primary
retrieval index.  When a query arrives with an object + action pair, all
matching images are returned from SQLite -- no FAISS needed because the
key space is discrete and small (hundreds of categories x 17 affordances).

Optionally, a FAISS / NumPy index on the *image feature vector* can be
used for soft similarity search (e.g. "images that look similar" beyond
exact category matching).
"""

from __future__ import annotations

import os
import json
import sqlite3
import threading
import time
import shutil
import uuid
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

# ── Try importing FAISS ─────────────────────────────────────────────────
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    faiss = None  # type: ignore


class ImageMemoryStore:
    """Dual-path storage for image memories.

    Parameters
    ----------
    store_dir : str
        Root directory for the SQLite database and image files.
    feature_dim : int
        Dimensionality of the image feature vectors (512 for ResNet18 layer4).
    use_faiss : bool
        If True, build a FAISS index on image features for soft similarity
        search.  Falls back to brute-force NumPy when FAISS is unavailable.
    max_images_per_key : int
        Maximum number of images to keep per (object, affordance) key.
        When exceeded, the oldest images are evicted (FIFO).
    """

    def __init__(
        self,
        store_dir: str = "./image_memory_store",
        feature_dim: int = 512,
        use_faiss: bool = True,
        max_images_per_key: int = 50,
    ):
        self.store_dir = store_dir
        self.feature_dim = feature_dim
        self.max_images_per_key = max_images_per_key
        self.images_dir = os.path.join(store_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        # ── Vector index (optional, for soft similarity search) ─────────
        self._lock = threading.Lock()
        self._use_faiss = use_faiss and _FAISS_AVAILABLE

        if self._use_faiss:
            self._faiss_index = faiss.IndexFlatIP(feature_dim)
            self._faiss_ids: List[str] = []
            print(f"[ImageMemoryStore] Using FAISS index (dim={feature_dim})")
        else:
            self._vectors: Optional[np.ndarray] = None
            self._vector_ids: List[str] = []
            print(f"[ImageMemoryStore] Using NumPy brute-force (dim={feature_dim})")

        # ── SQLite database ─────────────────────────────────────────────
        self._db_path = os.path.join(store_dir, "image_memories.db")
        self._init_db()

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create the image_memories table if it does not exist."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_memories (
                id              TEXT PRIMARY KEY,
                object_category TEXT NOT NULL,
                affordance_label TEXT NOT NULL,
                image_path      TEXT NOT NULL,
                image_feature   BLOB NOT NULL,
                sub_box         BLOB,
                obj_box         BLOB,
                confidence      REAL DEFAULT 0.0,
                timestamp       REAL,
                access_count    INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_obj_aff
            ON image_memories(object_category, affordance_label)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_affordance
            ON image_memories(affordance_label)
        """)
        conn.commit()
        conn.close()

    def _db_insert(self, entry_id: str, object_category: str,
                   affordance_label: str, image_path: str,
                   feature_blob: bytes, sub_box_blob: bytes = b'',
                   obj_box_blob: bytes = b'',
                   confidence: float = 0.0):
        """Insert or replace an image memory entry in SQLite."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO image_memories
                (id, object_category, affordance_label, image_path,
                 image_feature, sub_box, obj_box, confidence, timestamp, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                entry_id,
                object_category,
                affordance_label,
                image_path,
                feature_blob,
                sub_box_blob,
                obj_box_blob,
                confidence,
                time.time(),
            ),
        )
        conn.commit()
        conn.close()

    def _db_get_by_key(self, object_category: str,
                       affordance_label: str) -> List[Dict[str, Any]]:
        """Retrieve all entries matching (object_category, affordance_label)."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute(
            """
            SELECT id, object_category, affordance_label, image_path,
                   image_feature, sub_box, obj_box, confidence, timestamp, access_count
            FROM image_memories
            WHERE object_category = ? AND affordance_label = ?
            ORDER BY timestamp DESC
            """,
            (object_category, affordance_label),
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def _db_get_by_affordance(self, affordance_label: str) -> List[Dict[str, Any]]:
        """Retrieve all entries matching affordance_label."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute(
            """
            SELECT id, object_category, affordance_label, image_path,
                   image_feature, sub_box, obj_box, confidence, timestamp, access_count
            FROM image_memories
            WHERE affordance_label = ?
            ORDER BY timestamp DESC
            """,
            (affordance_label,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def _db_get_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single entry by ID."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute(
            """
            SELECT id, object_category, affordance_label, image_path,
                   image_feature, sub_box, obj_box, confidence, timestamp, access_count
            FROM image_memories
            WHERE id = ?
            """,
            (entry_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None
        return self._row_to_dict(row)

    def _db_count_by_key(self, object_category: str,
                         affordance_label: str) -> int:
        """Count entries for a given key."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM image_memories WHERE object_category = ? AND affordance_label = ?",
            (object_category, affordance_label),
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def _db_oldest_ids_by_key(self, object_category: str,
                              affordance_label: str, n: int) -> List[str]:
        """Return the n oldest entry IDs for a given key."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute(
            """
            SELECT id FROM image_memories
            WHERE object_category = ? AND affordance_label = ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (object_category, affordance_label, n),
        )
        ids = [r[0] for r in cursor.fetchall()]
        conn.close()
        return ids

    def _db_delete(self, entry_id: str):
        """Delete an entry from SQLite."""
        conn = sqlite3.connect(self._db_path)
        # First get the image_path to delete the file
        cursor = conn.execute(
            "SELECT image_path FROM image_memories WHERE id = ?", (entry_id,)
        )
        row = cursor.fetchone()
        if row and row[0] and os.path.exists(row[0]):
            try:
                os.remove(row[0])
            except OSError:
                pass
        conn.execute("DELETE FROM image_memories WHERE id = ?", (entry_id,))
        conn.commit()
        conn.close()

    def _db_increment_access(self, entry_id: str):
        """Increment access_count for an entry."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "UPDATE image_memories SET access_count = access_count + 1 WHERE id = ?",
            (entry_id,),
        )
        conn.commit()
        conn.close()

    def _db_count_all(self) -> int:
        """Total number of stored image memories."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM image_memories")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def _db_list_all(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """List all entries with pagination."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM image_memories")
        total = cursor.fetchone()[0]

        offset = (page - 1) * per_page
        cursor = conn.execute(
            """
            SELECT id, object_category, affordance_label, confidence, timestamp, access_count
            FROM image_memories
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (per_page, offset),
        )
        rows = cursor.fetchall()
        conn.close()

        entries = []
        for r in rows:
            entries.append({
                "id": r[0],
                "object_category": r[1],
                "affordance_label": r[2],
                "confidence": r[3],
                "timestamp": r[4],
                "access_count": r[5],
            })

        return {
            "entries": entries,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }

    def _db_list_categories(self) -> List[Dict[str, Any]]:
        """List distinct (object_category, affordance_label) pairs with counts."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute(
            """
            SELECT object_category, affordance_label, COUNT(*) as cnt
            FROM image_memories
            GROUP BY object_category, affordance_label
            ORDER BY object_category, affordance_label
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {"object_category": r[0], "affordance_label": r[1], "count": r[2]}
            for r in rows
        ]

    @staticmethod
    def _row_to_dict(row: tuple) -> Dict[str, Any]:
        """Convert a DB row tuple to a dictionary."""
        return {
            "id": row[0],
            "object_category": row[1],
            "affordance_label": row[2],
            "image_path": row[3],
            "image_feature": row[4],  # raw BLOB bytes
            "sub_box": row[5],
            "obj_box": row[6],
            "confidence": row[7],
            "timestamp": row[8],
            "access_count": row[9],
        }

    # ------------------------------------------------------------------
    # Feature vector encoding / decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_feature(feature: np.ndarray) -> bytes:
        """Encode a feature vector to BLOB bytes."""
        return feature.astype(np.float32).tobytes()

    @staticmethod
    def _decode_feature(blob: bytes, dim: int = 512) -> np.ndarray:
        """Decode BLOB bytes back to a feature vector."""
        return np.frombuffer(blob, dtype=np.float32).copy().reshape(dim)

    @staticmethod
    def _encode_box(box: np.ndarray) -> bytes:
        """Encode a bounding box to BLOB bytes."""
        if box is None or (isinstance(box, np.ndarray) and box.size == 0):
            return b''
        return box.astype(np.float32).tobytes()

    @staticmethod
    def _decode_box(blob: bytes) -> Optional[np.ndarray]:
        """Decode BLOB bytes back to a bounding box."""
        if not blob:
            return None
        return np.frombuffer(blob, dtype=np.float32).copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        image: np.ndarray,
        image_feature: np.ndarray,
        object_category: str,
        affordance_label: str,
        sub_box: Optional[np.ndarray] = None,
        obj_box: Optional[np.ndarray] = None,
        confidence: float = 0.0,
    ) -> str:
        """Store an image memory entry.

        Parameters
        ----------
        image : np.ndarray
            Raw image array, shape ``[H, W, 3]`` (uint8) or ``[3, H, W]`` (float).
        image_feature : np.ndarray
            Feature vector extracted by Img_Encoder, shape ``[C, h, w]`` or
            flattened ``[D]``.  If spatial, it will be average-pooled to ``[D]``.
        object_category : str
        affordance_label : str
        sub_box, obj_box : np.ndarray, optional
            Bounding boxes associated with this image.
        confidence : float

        Returns
        -------
        str
            The ID of the newly created entry.
        """
        entry_id = uuid.uuid4().hex

        # ── Pool feature if spatial ──────────────────────────────────────
        feat = image_feature.copy()
        if feat.ndim > 1:
            # e.g. [C, h, w] → average pool → [C]
            feat = feat.reshape(feat.shape[0], -1).mean(axis=1)
        feat = feat.flatten().astype(np.float32)

        # ── Save image to disk ───────────────────────────────────────────
        img_filename = f"{entry_id}.npy"
        img_path = os.path.join(self.images_dir, img_filename)
        np.save(img_path, image)

        # ── Encode BLOBs ─────────────────────────────────────────────────
        feature_blob = self._encode_feature(feat)
        sub_box_blob = self._encode_box(sub_box) if sub_box is not None else b''
        obj_box_blob = self._encode_box(obj_box) if obj_box is not None else b''

        # ── Insert into SQLite ───────────────────────────────────────────
        self._db_insert(
            entry_id=entry_id,
            object_category=object_category,
            affordance_label=affordance_label,
            image_path=img_path,
            feature_blob=feature_blob,
            sub_box_blob=sub_box_blob,
            obj_box_blob=obj_box_blob,
            confidence=confidence,
        )

        # ── Add to vector index ──────────────────────────────────────────
        with self._lock:
            if self._use_faiss:
                v = feat.reshape(1, -1).astype(np.float32)
                # L2 normalise for cosine similarity via inner product
                v = v / (np.linalg.norm(v) + 1e-8)
                self._faiss_index.add(v)
                self._faiss_ids.append(entry_id)
            else:
                v = feat.reshape(1, -1).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-8)
                if self._vectors is None:
                    self._vectors = v
                else:
                    self._vectors = np.vstack([self._vectors, v])
                self._vector_ids.append(entry_id)

        # ── Eviction check ───────────────────────────────────────────────
        count = self._db_count_by_key(object_category, affordance_label)
        if count > self.max_images_per_key:
            excess = count - self.max_images_per_key
            oldest_ids = self._db_oldest_ids_by_key(
                object_category, affordance_label, excess
            )
            for oid in oldest_ids:
                self.remove(oid)

        return entry_id

    def retrieve_by_key(
        self,
        object_category: str,
        affordance_label: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve image memories by (object_category, affordance_label).

        Parameters
        ----------
        object_category : str
        affordance_label : str
        top_k : int, optional
            If set, return at most this many entries (most recent first).

        Returns
        -------
        list of dict
            Each dict contains: id, image_path, image_feature (np.ndarray),
            sub_box, obj_box, confidence, timestamp.
        """
        entries = self._db_get_by_key(object_category, affordance_label)

        if top_k is not None:
            entries = entries[:top_k]

        # Decode features and boxes
        for entry in entries:
            entry["image_feature_decoded"] = self._decode_feature(
                entry["image_feature"], self.feature_dim
            )
            entry["sub_box_decoded"] = self._decode_box(entry.get("sub_box", b''))
            entry["obj_box_decoded"] = self._decode_box(entry.get("obj_box", b''))
            # Increment access count
            self._db_increment_access(entry["id"])

        return entries

    def retrieve_by_affordance(
        self,
        affordance_label: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve image memories by affordance_label only (all objects)."""
        entries = self._db_get_by_affordance(affordance_label)
        if top_k is not None:
            entries = entries[:top_k]
        for entry in entries:
            entry["image_feature_decoded"] = self._decode_feature(
                entry["image_feature"], self.feature_dim
            )
            entry["sub_box_decoded"] = self._decode_box(entry.get("sub_box", b''))
            entry["obj_box_decoded"] = self._decode_box(entry.get("obj_box", b''))
            self._db_increment_access(entry["id"])
        return entries

    def search_similar(
        self,
        query_feature: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for images with similar features (soft matching).

        Uses FAISS / NumPy index for approximate nearest-neighbour search.
        """
        feat = query_feature.copy().flatten().astype(np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-8)

        with self._lock:
            if self._use_faiss:
                q = feat.reshape(1, -1).astype(np.float32)
                k = min(top_k, self._faiss_index.ntotal)
                if k == 0:
                    return []
                distances, indices = self._faiss_index.search(q, k)
                ids = [self._faiss_ids[i] for i in indices[0] if i >= 0]
            else:
                if self._vectors is None or len(self._vector_ids) == 0:
                    return []
                q = feat.reshape(1, -1)
                dists = np.linalg.norm(self._vectors - q, axis=1)
                k = min(top_k, len(self._vector_ids))
                top_idx = np.argpartition(dists, kth=k - 1)[:k]
                top_idx = top_idx[np.argsort(dists[top_idx])]
                ids = [self._vector_ids[i] for i in top_idx]

        results = []
        for eid in ids:
            entry = self._db_get_by_id(eid)
            if entry:
                entry["image_feature_decoded"] = self._decode_feature(
                    entry["image_feature"], self.feature_dim
                )
                entry["sub_box_decoded"] = self._decode_box(entry.get("sub_box", b''))
                entry["obj_box_decoded"] = self._decode_box(entry.get("obj_box", b''))
                self._db_increment_access(eid)
                results.append(entry)

        return results

    def remove(self, entry_id: str):
        """Remove an image memory entry."""
        with self._lock:
            if self._use_faiss:
                if entry_id in self._faiss_ids:
                    idx = self._faiss_ids.index(entry_id)
                    self._faiss_ids.pop(idx)
                    self._rebuild_faiss_index()
            else:
                if entry_id in self._vector_ids:
                    idx = self._vector_ids.index(entry_id)
                    self._vector_ids.pop(idx)
                    if self._vectors is not None:
                        self._vectors = np.delete(self._vectors, idx, axis=0)
        self._db_delete(entry_id)

    def count(self) -> int:
        """Total number of stored image memories."""
        return self._db_count_all()

    def list_all(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """List all entries with pagination."""
        return self._db_list_all(page, per_page)

    def list_categories(self) -> List[Dict[str, Any]]:
        """List all (object_category, affordance_label) pairs with counts."""
        return self._db_list_categories()

    def clear(self):
        """Delete all image memories."""
        with self._lock:
            if self._use_faiss:
                self._faiss_index.reset()
                self._faiss_ids.clear()
            else:
                self._vectors = None
                self._vector_ids.clear()

        conn = sqlite3.connect(self._db_path)
        conn.execute("DELETE FROM image_memories")
        conn.commit()
        conn.close()

        # Remove all image files
        if os.path.exists(self.images_dir):
            shutil.rmtree(self.images_dir)
            os.makedirs(self.images_dir, exist_ok=True)

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the image memory store."""
        return {
            "total_images": self.count(),
            "max_per_key": self.max_images_per_key,
            "feature_dim": self.feature_dim,
            "use_faiss": self._use_faiss,
            "categories": self.list_categories(),
        }

    # ------------------------------------------------------------------
    # FAISS rebuild
    # ------------------------------------------------------------------

    def _rebuild_faiss_index(self):
        """Rebuild the FAISS index from all remaining entries in SQLite."""
        if not self._use_faiss:
            return
        self._faiss_index.reset()
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT id, image_feature FROM image_memories")
        new_ids = []
        vectors = []
        current_id_set = set(self._faiss_ids)
        for row in cursor:
            entry_id = row[0]
            if entry_id in current_id_set:
                feat = self._decode_feature(row[1], self.feature_dim)
                v = feat.reshape(1, -1).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-8)
                vectors.append(v)
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
        """Save the FAISS index to disk."""
        if not self._use_faiss:
            return
        path = path or os.path.join(self.store_dir, "image_faiss.index")
        faiss.write_index(self._faiss_index, path)
        id_path = path + ".ids"
        with open(id_path, "w") as f:
            for mid in self._faiss_ids:
                f.write(mid + "\n")

    def load_index(self, path: Optional[str] = None):
        """Load the FAISS index from disk."""
        if not self._use_faiss:
            return
        path = path or os.path.join(self.store_dir, "image_faiss.index")
        if not os.path.exists(path):
            return
        self._faiss_index = faiss.read_index(path)
        id_path = path + ".ids"
        self._faiss_ids = []
        if os.path.exists(id_path):
            with open(id_path, "r") as f:
                self._faiss_ids = [line.strip() for line in f if line.strip()]
