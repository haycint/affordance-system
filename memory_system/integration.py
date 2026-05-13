"""
Integration – glue code that connects the memory system to the IAG model
(both :class:`MyNet` and :class:`IAG_TextEmb`).

Two integration patterns are provided:

1. **Offline pre-population** – build the memory store from the training
   dataset's ground-truth labels, so that the memory system is "warm"
   before any real interaction occurs.

2. **Online inference enhancement** – during inference, retrieve
   relevant memories and apply the fused preference as a residual to
   the model's raw output.

Usage example (offline)::

    from memory_system.integration import prepopulate_from_dataset
    manager = prepopulate_from_dataset(
        model=iag_model,
        dataset=train_dataset,
        device="cuda:0",
        setting="Seen",
    )

Usage example (online)::

    from memory_system.integration import MemoryEnhancedInference
    enhancer = MemoryEnhancedInference(manager, iag_model, device="cuda:0")
    final_pred = enhancer.predict(img, xyz, sub_box, obj_box, affordance_label="grasp")
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Add project root to path ────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from memory_system.memory_manager import MemoryManager
from memory_system.memory_entry import MemoryEntry


# ======================================================================
# Offline pre-population
# ======================================================================

def prepopulate_from_dataset(
    model: torch.nn.Module,
    dataset,
    device: str = "cuda:0",
    setting: str = "Seen",
    manager: Optional[MemoryManager] = None,
    max_samples: int = 500,
    batch_size: int = 4,
    reward: float = 1.0,
    use_text_emb: bool = False,
    affordance_emb_tensor: Optional[torch.Tensor] = None,
) -> MemoryManager:
    """Pre-populate the memory store from a training dataset.

    For each sample, we run the model forward pass, extract the ARM
    feature to generate an index vector, and store the ground-truth
    affordance label as a positive preference matrix.

    Parameters
    ----------
    model : nn.Module
        IAG model (:class:`MyNet` or :class:`IAG_TextEmb`).
    dataset
        PIAD training dataset instance.
    device : str
    setting : str
        ``"Seen"`` or ``"Unseen"``.
    manager : MemoryManager, optional
        Reuse an existing manager, or create a new one if None.
    max_samples : int
        Maximum number of samples to pre-populate.
    batch_size : int
        Not used (samples are processed one by one because the training
        dataset returns variable-length lists).
    reward : float
        Reward value for ground-truth positive preference.
    use_text_emb : bool
        If True, also store the text embedding (for IAG_TextEmb).
    affordance_emb_tensor : torch.Tensor, optional
        ``[num_affordance, text_dim]`` tensor on device.

    Returns
    -------
    MemoryManager
        The populated memory manager.
    """
    if manager is None:
        manager = MemoryManager(emb_dim=512, index_dim=128, feat_dim=512)

    model.eval()
    model.to(device)
    dev = torch.device(device)

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    AFFORDANCE_LABELS = [
        'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
        'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
        'wear', 'press', 'cut', 'stab'
    ]

    count = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            if count >= max_samples:
                break

            try:
                # Training dataset returns:
                # Img, Points_List, affordance_label_List,
                # affordance_index_List, sub_box, obj_box
                img = sample[0].to(dev)
                points_list = sample[1]
                labels_list = sample[2]
                indices_list = sample[3]
                sub_box = sample[4].to(dev)
                obj_box = sample[5].to(dev)

                # Process each paired point cloud
                for point, label, aff_idx in zip(points_list, labels_list, indices_list):
                    if count >= max_samples:
                        break

                    point = point.float().to(dev)
                    aff_idx_val = aff_idx.item() if isinstance(aff_idx, torch.Tensor) else aff_idx

                    # Forward pass (need ARM features)
                    # We use a hook to capture ARM output
                    arm_output = _capture_arm_feature(model, img, point, sub_box, obj_box,
                                                      use_text_emb=use_text_emb,
                                                      affordance_emb_tensor=affordance_emb_tensor,
                                                      aff_idx=aff_idx_val)

                    # Get point features (from the last layer of point encoder)
                    point_features = _capture_point_features(model, point)

                    # Generate preference from ground truth
                    gt_label = label.cpu().numpy()
                    if gt_label.ndim > 1:
                        gt_label = gt_label.squeeze()
                    pref = MemoryManager.generate_preference_from_ground_truth(
                        gt_label, reward=reward
                    )

                    # Derive object category and affordance label from dataset
                    aff_name = AFFORDANCE_LABELS[aff_idx_val] if aff_idx_val < len(AFFORDANCE_LABELS) else "unknown"

                    # Text embedding (if IAG_TextEmb)
                    text_emb = None
                    if use_text_emb and affordance_emb_tensor is not None:
                        text_emb = affordance_emb_tensor[aff_idx_val].cpu().numpy()

                    # Store the memory
                    point_cloud_np = point.cpu().numpy().squeeze().T  # [3, N] -> [N, 3]
                    point_feat_np = point_features.cpu().numpy()

                    manager.form_memory(
                        arm_feature=arm_output,
                        point_cloud=point_cloud_np,
                        point_features=point_feat_np,
                        preference_matrix=pref,
                        reward=reward,
                        outcome="success",  # ground truth is always "correct"
                        affordance_label=aff_name,
                        confidence=1.0,  # ground truth is fully confident
                        text_embedding=text_emb,
                    )

                    count += 1

            except Exception as e:
                print(f"[prepopulate] Error at sample {batch_idx}: {e}")
                continue

    print(f"[prepopulate] Stored {count} memories")
    return manager


# ======================================================================
# Helper: capture internal features via hooks
# ======================================================================

def _capture_arm_feature(
    model: torch.nn.Module,
    img: torch.Tensor,
    point: torch.Tensor,
    sub_box: torch.Tensor,
    obj_box: torch.Tensor,
    use_text_emb: bool = False,
    affordance_emb_tensor: Optional[torch.Tensor] = None,
    aff_idx: int = 0,
) -> torch.Tensor:
    """Run a forward pass and capture the ARM output feature.

    Returns the ARM output ``F_a`` of shape ``[1, N_p + N_i, C]``.
    """
    arm_output = [None]

    def hook_fn(module, input, output):
        arm_output[0] = output.detach()

    # Register a temporary hook on the ARM module
    handle = model.ARM.register_forward_hook(hook_fn)

    try:
        if use_text_emb and affordance_emb_tensor is not None:
            text_emb = affordance_emb_tensor[aff_idx].unsqueeze(0)  # [1, text_dim]
            _ = model(img, point, sub_box, obj_box, text_emb)
        else:
            _ = model(img, point, sub_box, obj_box)
    finally:
        handle.remove()

    return arm_output[0]


def _capture_point_features(
    model: torch.nn.Module,
    point: torch.Tensor,
) -> torch.Tensor:
    """Extract per-point features from the model's point encoder.

    Returns features of shape ``[N_raw, D]`` using the deepest
    PointNet++ set-abstraction output, propagated back to the original
    resolution via feature propagation.
    """
    # Use the point encoder to get hierarchical features
    encoder_p = model.point_encoder(point)

    # The last level contains the most abstract features
    # l3_points: [B, C, N_p]  — these are the most semantic
    l3_points = encoder_p[-1][1]  # [1, C, N_p]

    # Average pool to get a global feature, then expand to N_raw
    # For alignment purposes, we use the global point feature
    # In a more sophisticated version, we would use FP to get
    # per-N_raw features
    B, C, N_p = l3_points.shape
    N_raw = point.shape[-1]

    # Use feature propagation to get per-point features at N_raw resolution
    from model.MyNet import PointNetFeaturePropagation

    # Build a simple per-point feature by propagating l3 back
    # This is a simplified version; the full decoder does multi-level FP
    fp = PointNetFeaturePropagation(
        in_channel=C + 3,  # +3 for xyz coordinates
        mlp=[C, C]
    ).to(point.device)

    # l3_xyz: [1, 3, N_p]
    l3_xyz = encoder_p[-1][0]
    # l0_xyz: [1, 3, N_raw]
    l0_xyz = point[:, :3, :]

    # Concatenate xyz with features for FP
    up_sampled = fp(l0_xyz, l3_xyz, point, l3_points)  # [1, C, N_raw]

    # Transpose to [N_raw, C]
    point_features = up_sampled.squeeze(0).permute(1, 0)
    return point_features


# ======================================================================
# Online inference enhancement
# ======================================================================

class MemoryEnhancedInference:
    """Wrap an IAG model with the memory system for enhanced inference.

    Usage::

        enhancer = MemoryEnhancedInference(manager, model, device="cuda:0")
        result = enhancer.predict(img, xyz, sub_box, obj_box,
                                   affordance_label="grasp")
    """

    def __init__(
        self,
        manager: MemoryManager,
        model: torch.nn.Module,
        device: str = "cuda:0",
        use_text_emb: bool = False,
        affordance_emb_tensor: Optional[torch.Tensor] = None,
        alpha: float = 0.3,
        top_k: int = 5,
    ):
        self.manager = manager
        self.model = model
        self.device = torch.device(device)
        self.use_text_emb = use_text_emb
        self.affordance_emb_tensor = affordance_emb_tensor
        self.alpha = alpha
        self.top_k = top_k

        self.model.eval()
        self.model.to(self.device)

    def predict(
        self,
        img: torch.Tensor,
        xyz: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
        affordance_label: Optional[str] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Run memory-enhanced inference.

        Parameters
        ----------
        img : torch.Tensor
            Image tensor, ``[1, 3, H, W]``.
        xyz : torch.Tensor
            Point cloud, ``[1, 3, N_raw]``.
        sub_box, obj_box : torch.Tensor
            Bounding boxes, ``[1, 4]``.
        affordance_label : str, optional
            If provided, filter memories by this affordance.
        return_details : bool
            If True, also return the raw model output and the
            fused preference for debugging.

        Returns
        -------
        dict with keys:
            - ``"prediction"``: final affordance map after memory, ``[N_raw]``
            - ``"raw_prediction"``: model output before memory, ``[N_raw]``
            - ``"logits"``: classification logits, ``[num_affordance]``
            - ``"pref_fused"``: fused preference (if return_details)
            - ``"memory_applied"``: whether memory was applied
        """
        with torch.no_grad():
            # Forward pass
            if self.use_text_emb and self.affordance_emb_tensor is not None:
                # Use zero text embedding for initial prediction
                B = img.size(0)
                text_emb = torch.zeros(B, self.affordance_emb_tensor.size(1),
                                       device=self.device)
                _3d, logits, to_KL = self.model(img, xyz, sub_box, obj_box, text_emb)
            else:
                _3d, logits, to_KL = self.model(img, xyz, sub_box, obj_box)

        raw_output = _3d.squeeze().cpu().numpy()  # [N_raw]

        # Try memory enhancement
        try:
            arm_feature = _capture_arm_feature(
                self.model, img, xyz, sub_box, obj_box,
                use_text_emb=self.use_text_emb,
                affordance_emb_tensor=self.affordance_emb_tensor,
            )

            point_features = _capture_point_features(self.model, xyz)
            point_cloud_np = xyz.squeeze().cpu().numpy().T  # [N_raw, 3]

            pref_fused = self.manager.retrieve_and_fuse(
                arm_feature=arm_feature,
                current_point_cloud=point_cloud_np,
                current_point_features=point_features.cpu().numpy(),
                top_k=self.top_k,
                affordance_label=affordance_label,
            )

            if np.abs(pref_fused).sum() > 1e-6:
                # Memory has relevant content
                enhanced_raw = self.manager.apply_memory_to_output(
                    raw_output, pref_fused, alpha=self.alpha
                )
                prediction = 1.0 / (1.0 + np.exp(-enhanced_raw))  # sigmoid
                memory_applied = True
            else:
                prediction = raw_output
                memory_applied = False
        except Exception as e:
            print(f"[MemoryEnhancedInference] Memory retrieval failed: {e}")
            prediction = raw_output
            pref_fused = np.zeros_like(raw_output)
            memory_applied = False

        result = {
            "prediction": prediction,
            "raw_prediction": raw_output,
            "logits": logits.squeeze().cpu().numpy(),
            "memory_applied": memory_applied,
        }

        if return_details:
            result["pref_fused"] = pref_fused

        return result

    def predict_and_remember(
        self,
        img: torch.Tensor,
        xyz: torch.Tensor,
        sub_box: torch.Tensor,
        obj_box: torch.Tensor,
        gt_label: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        affordance_label: Optional[str] = None,
        object_category: str = "",
    ) -> Dict[str, Any]:
        """Predict with memory enhancement AND optionally form a new memory.

        This is the full **perceive → predict → remember** cycle.

        If ``gt_label`` is provided, a memory is formed from the
        ground-truth label.  If ``reward`` is also provided, it is used
        as the reward value; otherwise the reward is computed from the
        prediction quality.
        """
        result = self.predict(img, xyz, sub_box, obj_box, affordance_label)

        # Form a memory if ground truth is available
        if gt_label is not None:
            pred = result["prediction"]

            # Compute reward if not provided
            if reward is None:
                mae = np.mean(np.abs(pred - gt_label))
                reward = max(0.0, 1.0 - mae * 2)  # simple heuristic

            # Generate preference
            if reward >= 0.5:
                pref = MemoryManager.generate_preference_from_ground_truth(gt_label, reward=reward)
                outcome = "success" if reward > 0.7 else "partial"
            else:
                pref = MemoryManager.generate_preference_from_prediction(
                    pred, gt_label,
                    positive_reward=1.0,
                    negative_reward=-1.0,
                )
                outcome = "failure"

            # Capture features for memory
            with torch.no_grad():
                arm_feature = _capture_arm_feature(
                    self.model, img, xyz, sub_box, obj_box,
                    use_text_emb=self.use_text_emb,
                    affordance_emb_tensor=self.affordance_emb_tensor,
                )
                point_features = _capture_point_features(self.model, xyz)

            point_cloud_np = xyz.squeeze().cpu().numpy().T
            text_emb = None
            if self.use_text_emb and self.affordance_emb_tensor is not None:
                # Use predicted class for text embedding
                pred_class = np.argmax(result["logits"])
                text_emb = self.affordance_emb_tensor[pred_class].cpu().numpy()

            self.manager.form_memory(
                arm_feature=arm_feature,
                point_cloud=point_cloud_np,
                point_features=point_features.cpu().numpy(),
                preference_matrix=pref,
                reward=reward,
                outcome=outcome,
                object_category=object_category,
                affordance_label=affordance_label or "",
                confidence=float(np.max(result["logits"])),
                text_embedding=text_emb,
            )
            result["memory_formed"] = True
        else:
            result["memory_formed"] = False

        return result