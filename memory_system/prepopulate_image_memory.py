"""
Pre-populate image memory from training dataset using IAG_TextEmb model.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from memory_system.image_memory_manager import ImageMemoryManager
# Assuming your model file is importable
from model import IAG_TextEmb  # or wherever your model is defined


def prepopulate_image_memory_from_dataset(
    model: IAG_TextEmb,
    dataset,
    device: str = "cuda:0",
    store_dir: str = "./image_memory_store",
    max_samples: int = 500,
    batch_size: int = 1,
    object_category_extractor: Optional[Callable] = None,
    affordance_labels: Optional[list] = None,
    max_images_per_key: int = 50,
) -> ImageMemoryManager:
    """Pre-populate the image memory store with F_i, F_s, F_e features.

    Parameters
    ----------
    model : IAG_TextEmb
        The IAG_TextEmb model (must be on correct device).
    dataset
        Training dataset that yields samples: (img, points_list, labels_list,
        indices_list, sub_box, obj_box) or similar format.
    device : str
        Device to run inference on.
    store_dir : str
        Directory for the image memory store.
    max_samples : int
        Maximum number of samples to process.
    batch_size : int
        Batch size (should be 1 due to variable point lists).
    object_category_extractor : Callable, optional
        A function that takes a sample and returns object category string.
        If None, uses "object_{batch_idx}".
    affordance_labels : list, optional
        List of affordance label strings (e.g. ['grasp', 'contain', ...]).
        If None, uses default PIAD 17 classes.
    max_images_per_key : int
        Maximum images to store per (object, affordance) key.

    Returns
    -------
    ImageMemoryManager
        The populated image memory manager.
    """
    if affordance_labels is None:
        affordance_labels = [
            'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
            'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
            'wear', 'press', 'cut', 'stab'
        ]

    manager = ImageMemoryManager(
        store_dir=store_dir,
        feature_dim=512,
        use_faiss=True,
        max_images_per_key=max_images_per_key,
        averaging_strategy="mean",  # not used for retrieval
    )

    model.eval()
    model.to(device)
    dev = torch.device(device)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    count = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            if count >= max_samples:
                break

            try:
                # Adjust based on your dataset's output format.
                # Common format: (img, points_list, labels_list, indices_list, sub_box, obj_box)
                img = sample[0].to(dev)
                points_list = sample[1]  # list of tensors
                labels_list = sample[2]  # list of tensors (affordance labels per point)
                indices_list = sample[3]  # list of affordance indices
                sub_box = sample[4].to(dev)  # [B, 4]
                obj_box = sample[5].to(dev)  # [B, 4]

                # Extract image feature F_I
                F_I = model.img_encoder(img)  # [B, C, H', W']

                # For each paired point cloud (batch size is 1, but dataset may have multiple point clouds per image)
                for point, label, aff_idx in zip(points_list, labels_list, indices_list):
                    if count >= max_samples:
                        break

                    point = point.float().to(dev)  # [1, 3, N_p] or [3, N_p]? adjust
                    aff_idx_val = aff_idx.item() if isinstance(aff_idx, torch.Tensor) else aff_idx

                    # Get F_i, F_s, F_e using model's method
                    F_i, F_s, F_e = model.get_mask_feature(img, F_I, sub_box, obj_box, dev)

                    # Convert to numpy for storage
                    F_i_np = F_i.squeeze(0).cpu().numpy()  # [C, 4, 4]
                    F_s_np = F_s.squeeze(0).cpu().numpy()
                    F_e_np = F_e.squeeze(0).cpu().numpy()  # [C, H, W]

                    # Store image (convert to numpy)
                    img_np = img.squeeze(0).cpu().numpy()  # [3, H, W]
                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        img_np = img_np.transpose(1, 2, 0)  # HWC for storage

                    # Get affordance label string
                    aff_name = affordance_labels[aff_idx_val] if aff_idx_val < len(affordance_labels) else "unknown"

                    # Get object category
                    if object_category_extractor is not None:
                        obj_cat = object_category_extractor(sample)
                    else:
                        obj_cat = f"object_{batch_idx}"

                    # Store the memory
                    manager.store_image(
                        image=img_np,
                        image_feature=F_I.squeeze(0).cpu().numpy(),  # [C, H', W']
                        object_category=obj_cat,
                        affordance_label=aff_name,
                        sub_box=sub_box.squeeze(0).cpu().numpy() if sub_box.numel() > 0 else None,
                        obj_box=obj_box.squeeze(0).cpu().numpy() if obj_box.numel() > 0 else None,
                        confidence=1.0,  # ground truth
                        F_i=F_i_np,
                        F_s=F_s_np,
                        F_e=F_e_np,
                    )
                    count += 1

            except Exception as e:
                print(f"[prepopulate_image_memory] Error at sample {batch_idx}: {e}")
                continue

    print(f"[prepopulate_image_memory] Stored {count} image memories")
    return manager


def prepopulate_from_dataset_with_custom_extractor(
    model: IAG_TextEmb,
    dataset,
    device: str = "cuda:0",
    store_dir: str = "./image_memory_store",
    max_samples: int = 500,
    object_category_extractor: Optional[Callable] = None,
) -> ImageMemoryManager:
    """Wrapper with custom object category extractor."""
    return prepopulate_image_memory_from_dataset(
        model=model,
        dataset=dataset,
        device=device,
        store_dir=store_dir,
        max_samples=max_samples,
        object_category_extractor=object_category_extractor,
    )