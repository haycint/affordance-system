"""
ImageMemoryManager -- high-level orchestrator for the image memory system.

This module provides the core functionality:

1. **During inference**, query the image memory by (object + affordance) to
   retrieve additional images, extract their features using the model's
   Img_Encoder, average them with the current image feature, and feed the
   averaged feature into the JRA module.

2. **After inference**, store the current image (along with its extracted
   feature, bounding boxes, and metadata) into the image memory for future
   retrieval.

Core equation (feature averaging before JRA)::

    F_I_current  = img_encoder(img_current)           # [B, C, h, w]
    F_I_memory_k = stored features from image memory   # [K, C]
    F_I_avg      = (F_I_current + mean(F_I_memory)) / 2  # averaged

The averaged feature F_I_avg is then used in place of F_I when computing
F_i (object ROI feature), F_s (subject ROI feature), and F_e (scene mask
feature) in the model's ``get_mask_feature()`` call.

Architecture overview::

    ┌───────────────────────────────────────────────────────────┐
    │                   ImageMemoryManager                      │
    │                                                           │
    │  ┌─────────────────┐    ┌──────────────────────────────┐ │
    │  │ ImageMemoryStore │    │  Feature Averaging Logic     │ │
    │  │ (SQLite + FAISS) │    │  F_avg = avg(F_cur, F_mem)  │ │
    │  └────────┬────────┘    └──────────────────────────────┘ │
    │           │                                               │
    │           │   ┌───────────────────────────────────────┐  │
    │           └──▶│  Img_Encoder (from IAG_TextEmb/MyNet) │  │
    │               │  Used for on-the-fly feature extract. │  │
    │               └───────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import time
import threading
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .image_memory_store import ImageMemoryStore


class ImageMemoryManager:
    """Top-level orchestrator for image memory enhanced inference.

    Parameters
    ----------
    store_dir : str
        Directory for the ImageMemoryStore.
    feature_dim : int
        Dimensionality of the image feature vector (default 512 for ResNet18).
    use_faiss : bool
        Whether to use FAISS for soft similarity search.
    max_images_per_key : int
        Maximum images per (object, affordance) key.
    averaging_strategy : str
        How to combine current and memory features:
        - ``"mean"``: simple average  (F_cur + F_mem_avg) / 2
        - ``"weighted"``: weighted average with learnable or fixed alpha
        - ``"attention"``: attention-weighted combination
    alpha : float
        Weight for the current image feature in weighted averaging.
        ``F_final = alpha * F_cur + (1 - alpha) * F_mem_avg``
        Only used when ``averaging_strategy="weighted"``.
    max_memory_images : int
        Maximum number of memory images to retrieve and average per query.
    """

    def __init__(
        self,
        store_dir: str = "./image_memory_store",
        feature_dim: int = 512,
        use_faiss: bool = True,
        max_images_per_key: int = 50,
        averaging_strategy: str = "mean",
        alpha: float = 0.5,
        max_memory_images: int = 3,
    ):
        self.store = ImageMemoryStore(
            store_dir=store_dir,
            feature_dim=feature_dim,
            use_faiss=use_faiss,
            max_images_per_key=max_images_per_key,
        )
        self.feature_dim = feature_dim
        self.averaging_strategy = averaging_strategy
        self.alpha = alpha
        self.max_memory_images = max_memory_images

        # ── Optional attention gate for "attention" strategy ─────────────
        if averaging_strategy == "attention":
            self._attn_gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, 1),
                nn.Sigmoid(),
            )

    # ==================================================================
    # Core: retrieve memory features and compute averaged feature
    # ==================================================================

    def retrieve_and_average_feature(
        self,
        current_feature: torch.Tensor,
        object_category: str,
        affordance_label: str,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Retrieve memory image features and compute averaged feature.

        This is the main method called during inference. It:
        1. Queries the image memory by (object_category, affordance_label)
        2. Retrieves stored feature vectors
        3. Averages them with the current feature
        4. Returns the averaged feature and metadata

        Parameters
        ----------
        current_feature : torch.Tensor
            Current image feature from Img_Encoder, shape ``[B, C, h, w]``
            or already pooled ``[B, C]`` or ``[C]``.
        object_category : str
        affordance_label : str
        device : torch.device, optional

        Returns
        -------
        averaged_feature : torch.Tensor
            Same shape as current_feature but with memory information
            blended in.
        info : dict
            Metadata about the retrieval (num_retrieved, averaging_method, etc.)
        """
        info = {
            "memory_retrieved": 0,
            "averaging_strategy": self.averaging_strategy,
            "memory_applied": False,
        }

        # ── 1. Retrieve memory entries ──────────────────────────────────
        entries = self.store.retrieve_by_key(
            object_category=object_category,
            affordance_label=affordance_label,
            top_k=self.max_memory_images,
        )

        if not entries:
            # No memory available, return current feature unchanged
            return current_feature, info

        # ── 2. Extract pooled feature vectors from memory ───────────────
        mem_features = []
        for entry in entries:
            feat = entry["image_feature_decoded"]  # [D] numpy array
            mem_features.append(feat)

        mem_features_np = np.stack(mem_features, axis=0)  # [K, D]
        mem_features_t = torch.from_numpy(mem_features_np).float()
        if device is not None:
            mem_features_t = mem_features_t.to(device)

        # ── 3. Pool current feature if spatial ──────────────────────────
        cur_feat = current_feature
        was_spatial = False
        spatial_shape = None

        if cur_feat.dim() == 4:
            # [B, C, h, w] → pool to [B, C]
            was_spatial = True
            spatial_shape = cur_feat.shape
            cur_pooled = cur_feat.mean(dim=[2, 3])  # [B, C]
        elif cur_feat.dim() == 3:
            # [B, C, N] → pool to [B, C]
            was_spatial = True
            spatial_shape = cur_feat.shape
            cur_pooled = cur_feat.mean(dim=2)  # [B, C]
        elif cur_feat.dim() == 2:
            cur_pooled = cur_feat  # [B, C]
        else:
            cur_pooled = cur_feat.unsqueeze(0)  # [1, C]

        # ── 4. Compute averaged pooled feature ──────────────────────────
        mem_avg = mem_features_t.mean(dim=0)  # [D]

        # Expand to batch size
        B = cur_pooled.shape[0]
        mem_avg_expanded = mem_avg.unsqueeze(0).expand(B, -1)  # [B, D]

        if self.averaging_strategy == "mean":
            # Simple mean: (F_cur + F_mem) / 2
            avg_pooled = (cur_pooled + mem_avg_expanded) / 2.0

        elif self.averaging_strategy == "weighted":
            # Weighted: alpha * F_cur + (1-alpha) * F_mem
            avg_pooled = self.alpha * cur_pooled + (1 - self.alpha) * mem_avg_expanded

        elif self.averaging_strategy == "attention":
            # Attention-weighted: concat → gate
            concat = torch.cat([cur_pooled, mem_avg_expanded], dim=-1)  # [B, 2D]
            if hasattr(self, '_attn_gate') and device is not None:
                self._attn_gate = self._attn_gate.to(device)
            if hasattr(self, '_attn_gate'):
                gate = self._attn_gate(concat)  # [B, 1]
                avg_pooled = gate * cur_pooled + (1 - gate) * mem_avg_expanded
            else:
                avg_pooled = (cur_pooled + mem_avg_expanded) / 2.0
        else:
            avg_pooled = (cur_pooled + mem_avg_expanded) / 2.0

        # ── 5. Restore spatial shape if needed ───────────────────────────
        if was_spatial and spatial_shape is not None:
            # Rescale the spatial feature map
            # cur_feat / cur_pooled * avg_pooled
            # This preserves the spatial structure while shifting the
            # feature distribution towards the memory-averaged direction
            cur_pooled_safe = cur_pooled.clone()
            cur_pooled_safe[cur_pooled_safe.abs() < 1e-8] = 1e-8

            # Scale factor per batch item
            scale = avg_pooled / cur_pooled_safe  # [B, C]

            if cur_feat.dim() == 4:
                # [B, C, h, w]
                avg_feature = cur_feat * scale.unsqueeze(-1).unsqueeze(-1)
            else:
                # [B, C, N]
                avg_feature = cur_feat * scale.unsqueeze(-1)
        else:
            avg_feature = avg_pooled
            if current_feature.dim() == 1:
                avg_feature = avg_feature.squeeze(0)

        info["memory_retrieved"] = len(entries)
        info["memory_applied"] = True

        return avg_feature, info

    # ==================================================================
    # Store image to memory
    # ==================================================================

    def store_image(
        self,
        image: np.ndarray,
        image_feature: np.ndarray,
        object_category: str,
        affordance_label: str,
        sub_box: Optional[np.ndarray] = None,
        obj_box: Optional[np.ndarray] = None,
        confidence: float = 0.0,
    ) -> str:
        """Store an image with its feature into the image memory.

        Parameters
        ----------
        image : np.ndarray
            Raw image array (for on-disk storage and later retrieval).
        image_feature : np.ndarray
            Feature vector from Img_Encoder, shape ``[C, h, w]`` or ``[D]``.
        object_category : str
        affordance_label : str
        sub_box, obj_box : np.ndarray, optional
        confidence : float

        Returns
        -------
        str
            Entry ID.
        """
        return self.store.add(
            image=image,
            image_feature=image_feature,
            object_category=object_category,
            affordance_label=affordance_label,
            sub_box=sub_box,
            obj_box=obj_box,
            confidence=confidence,
        )

    # ==================================================================
    # Batch pre-population from dataset
    # ==================================================================

    def prepopulate_from_dataset(
        self,
        model: torch.nn.Module,
        dataset,
        device: str = "cuda:0",
        setting: str = "Seen",
        max_samples: int = 500,
        use_text_emb: bool = False,
        affordance_emb_tensor: Optional[torch.Tensor] = None,
    ) -> int:
        """Pre-populate the image memory from a training dataset.

        For each sample, extract the image feature using the model's
        Img_Encoder and store it in the image memory.

        Parameters
        ----------
        model : nn.Module
            IAG model (MyNet or IAG_TextEmb).
        dataset
            PIAD training dataset.
        device : str
        setting : str
        max_samples : int
        use_text_emb : bool
        affordance_emb_tensor : torch.Tensor, optional

        Returns
        -------
        int
            Number of images stored.
        """
        from torch.utils.data import DataLoader

        model.eval()
        model.to(device)
        dev = torch.device(device)

        loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

        AFFORDANCE_LABELS = [
            'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
            'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
            'wear', 'press', 'cut', 'stab'
        ]

        # We need a transform to convert PIL image to tensor
        from torchvision import transforms
        img_transform = transforms.Compose([
            transforms.Resize((420, 420)),
            transforms.ToTensor(),
        ])

        count = 0
        seen_keys = set()  # Track (object, affordance) to avoid too many duplicates

        with torch.no_grad():
            for batch_idx, sample in enumerate(loader):
                if count >= max_samples:
                    break
                try:
                    img = sample[0].to(dev)
                    points_list = sample[1]
                    labels_list = sample[2]
                    indices_list = sample[3]
                    sub_box = sample[4]
                    obj_box = sample[5]

                    # Extract image feature using model's img_encoder
                    F_I = model.img_encoder(img)  # [1, C, h, w]
                    feature_np = F_I.cpu().numpy().squeeze()  # [C, h, w]

                    # Get image as numpy for storage
                    img_np = img.cpu().numpy().squeeze()  # [3, H, W]
                    # Transpose to [H, W, 3] for storage
                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        img_np = img_np.transpose(1, 2, 0)

                    sub_box_np = sub_box.cpu().numpy().squeeze() if sub_box.numel() > 0 else None
                    obj_box_np = obj_box.cpu().numpy().squeeze() if obj_box.numel() > 0 else None

                    # Store one entry per affordance label in this sample
                    for j, aff_idx in enumerate(indices_list):
                        if count >= max_samples:
                            break
                        aff_idx_val = aff_idx.item() if isinstance(aff_idx, torch.Tensor) else aff_idx
                        aff_name = AFFORDANCE_LABELS[aff_idx_val] if aff_idx_val < len(AFFORDANCE_LABELS) else "unknown"

                        # Derive object category from dataset path
                        # This is a simplified approach; ideally we'd get it from the dataset
                        object_cat = f"object_{batch_idx}"

                        key = (object_cat, aff_name)
                        # Skip if we already have enough for this key
                        if key in seen_keys and len(seen_keys) > self.max_images_per_key:
                            continue
                        seen_keys.add(key)

                        self.store_image(
                            image=img_np,
                            image_feature=feature_np,
                            object_category=object_cat,
                            affordance_label=aff_name,
                            sub_box=sub_box_np,
                            obj_box=obj_box_np,
                            confidence=1.0,
                        )
                        count += 1

                except Exception as e:
                    print(f"[ImageMemory prepopulate] Error at sample {batch_idx}: {e}")
                    continue

        print(f"[ImageMemory prepopulate] Stored {count} image memories")
        return count

    # ==================================================================
    # Store management
    # ==================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the image memory store."""
        return self.store.get_stats()

    def save(self):
        """Persist the image memory store to disk."""
        self.store.save_index()

    def load(self):
        """Load the image memory store from disk."""
        self.store.load_index()

    def clear(self):
        """Delete all image memories."""
        self.store.clear()

    def list_categories(self) -> List[Dict[str, Any]]:
        """List all (object_category, affordance_label) pairs."""
        return self.store.list_categories()

    def list_entries(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """List all entries with pagination."""
        return self.store.list_all(page, per_page)

    def retrieve_image(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single image memory entry by ID."""
        entry = self.store._db_get_by_id(entry_id)
        if entry:
            entry["image_feature_decoded"] = ImageMemoryStore._decode_feature(
                entry["image_feature"], self.feature_dim
            )
            # Load the actual image from disk
            img_path = entry.get("image_path", "")
            if img_path and os.path.exists(img_path):
                entry["image_data"] = np.load(img_path)
            else:
                entry["image_data"] = None
        return entry


# Import nn for attention gate
import torch.nn as nn
