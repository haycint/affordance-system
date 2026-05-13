"""
Train Annotation Model
======================

Training script for both annotation model schemes.

Usage:
    # Train Scheme 1 with pretrained backbone (default)
    python annotation/train_annotation.py --scheme 1 --pretrained

    # Train Scheme 1 from random initialization
    python train_annotation.py --scheme 1 --no-pretrained

    # Train Scheme 2 (multi-model collaboration)
    python train_annotation.py --scheme 2

    # Train Scheme 1 with frozen backbone (feature extraction only)
    python train_annotation.py --scheme 1 --pretrained --freeze-backbone

    # Resume training from checkpoint
    python train_annotation.py --scheme 1 --resume ./annotation/checkpoints/scheme1_epoch_10.pth

All arguments override values in config_annotation.yaml.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import importlib.util

# ---------------------------------------------------------------------------
# Robust module loading: load by exact file path to avoid any package
# namespace collision (e.g. 'annotation/' being a Python package).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name: str, filepath: str):
    """Load a Python module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ds_mod = _load_module("annotation_dataset", os.path.join(_HERE, "annotation_dataset.py"))
_model_mod = _load_module("annotation_model", os.path.join(_HERE, "annotation_model.py"))

AnnotationDataset = _ds_mod.AnnotationDataset
annotation_collate_fn = _ds_mod.annotation_collate_fn
build_annotation_datasets = _ds_mod.build_annotation_datasets

AnnotationModelScheme1 = _model_mod.AnnotationModelScheme1
AnnotationModelScheme2 = _model_mod.AnnotationModelScheme2
AnnotationLoss = _model_mod.AnnotationLoss
AnnotationMetrics = _model_mod.AnnotationMetrics
build_annotation_model = _model_mod.build_annotation_model
build_annotation_loss = _model_mod.build_annotation_loss


logger = logging.getLogger(__name__)


# ===========================================================================
# Argument parser
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Annotation Model (Scheme 1 or Scheme 2)"
    )

    # ---- Scheme selection ----
    parser.add_argument(
        "--scheme", type=int, choices=[1, 2], default=None,
        help="Model scheme: 1=end-to-end ImageNet, 2=multi-model collaboration. "
             "Overrides config_annotation.yaml."
    )

    # ---- Scheme 1 options ----
    parser.add_argument(
        "--pretrained", action="store_true", default=None,
        help="Load ImageNet pretrained weights for backbone (Scheme 1 only)."
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Train from random initialization (Scheme 1 only)."
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true", default=None,
        help="Freeze backbone and only train prediction heads (Scheme 1 only)."
    )
    parser.add_argument(
        "--backbone", type=str, choices=["resnet18", "resnet50"], default=None,
        help="Backbone architecture."
    )

    # ---- Training hyperparameters ----
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Image size (square)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # ---- Paths ----
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config_annotation.yaml"),
        help="Path to config_annotation.yaml"
    )
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint save directory")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    # ---- Device ----
    parser.add_argument("--device", type=str, default=None, help="Device: cuda / cpu")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


# ===========================================================================
# Helper functions
# ===========================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    save_path: str,
    scheme: int,
    config: dict,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "scheme": scheme,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "config": config,
        },
        save_path,
    )
    logger.info(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(f"Checkpoint loaded: {checkpoint_path} (epoch={ckpt.get('epoch', '?')})")
    return ckpt


# ===========================================================================
# Training loop
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: AnnotationLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    metrics: Optional[AnnotationMetrics] = None,
    log_interval: int = 20,
):
    """Train for one epoch. Returns average loss dict and average metrics dict."""
    model.train()
    total_losses = {}
    total_metrics = {}
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        imgs = batch["img"].to(device)
        targets = {
            "subject_box": batch["subject_box"].to(device),
            "object_box": batch["object_box"].to(device),
            "action_wv": batch["action_wv"].to(device),
            "object_wv": batch["object_wv"].to(device),
            "action_idx": batch["action_idx"].to(device),
            "object_idx": batch["object_idx"].to(device),
        }

        # Forward
        predictions = model(imgs)

        # Compute losses
        losses = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

        # Compute accuracy metrics (no grad, no effect on training)
        if metrics is not None:
            batch_metrics = metrics.compute(predictions, targets)
            for k, v in batch_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            loss_str = " | ".join(
                f"{k}: {v.item():.4f}" for k, v in losses.items()
            )
            acc_str = ""
            if metrics is not None and total_metrics:
                running_acc = {k: v / n_batches for k, v in total_metrics.items()}
                acc_str = " | " + " | ".join(
                    f"{k}: {v:.3f}" for k, v in running_acc.items()
                )
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] {loss_str}{acc_str}"
            )

    # Average
    avg_losses = {k: v / n_batches for k, v in total_losses.items()} if n_batches > 0 else {"total_loss": float("inf")}
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()} if total_metrics and n_batches > 0 else {}

    # TensorBoard
    global_step = epoch * max(len(dataloader), 1)
    for k, v in avg_losses.items():
        writer.add_scalar(f"train/{k}", v, global_step)
    for k, v in avg_metrics.items():
        writer.add_scalar(f"train/{k}", v, global_step)

    return avg_losses, avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: AnnotationLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    metrics: Optional[AnnotationMetrics] = None,
):
    """Validate. Returns average loss dict and average metrics dict."""
    model.eval()
    total_losses = {}
    total_metrics = {}
    n_batches = 0

    for batch in dataloader:
        imgs = batch["img"].to(device)
        targets = {
            "subject_box": batch["subject_box"].to(device),
            "object_box": batch["object_box"].to(device),
            "action_wv": batch["action_wv"].to(device),
            "object_wv": batch["object_wv"].to(device),
            "action_idx": batch["action_idx"].to(device),
            "object_idx": batch["object_idx"].to(device),
        }

        predictions = model(imgs)
        losses = loss_fn(predictions, targets)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()

        # Compute accuracy metrics
        if metrics is not None:
            batch_metrics = metrics.compute(predictions, targets)
            for k, v in batch_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

        n_batches += 1

    avg_losses = {k: v / n_batches for k, v in total_losses.items()} if n_batches > 0 else {"total_loss": float("inf")}
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()} if total_metrics and n_batches > 0 else {}

    # TensorBoard
    if n_batches > 0:
        for k, v in avg_losses.items():
            writer.add_scalar(f"val/{k}", v, epoch)
    for k, v in avg_metrics.items():
        writer.add_scalar(f"val/{k}", v, epoch)

    return avg_losses, avg_metrics


# ===========================================================================
# Main
# ===========================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with CLI args
    if args.scheme is not None:
        config["scheme"] = args.scheme
    if args.pretrained is not None:
        if args.no_pretrained:
            config.setdefault("scheme1", {})["pretrained"] = False
        elif args.pretrained:
            config.setdefault("scheme1", {})["pretrained"] = True
    if args.freeze_backbone is not None:
        config.setdefault("scheme1", {})["freeze_backbone"] = args.freeze_backbone
    if args.backbone is not None:
        config.setdefault("scheme1", {})["backbone"] = args.backbone
    if args.epochs is not None:
        config["epoch"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed
    if args.save_dir is not None:
        config["save_dir"] = args.save_dir
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir

    scheme = config["scheme"]
    assert scheme in (1, 2), f"Invalid scheme: {scheme}"

    seed = config.get("seed", 42)
    set_seed(seed)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Print training configuration
    # ------------------------------------------------------------------
    s1_cfg = config.get("scheme1", {})
    s2_cfg = config.get("scheme2", {})

    scheme_desc = {
        1: f"End-to-End ImageNet (backbone={s1_cfg.get('backbone', 'resnet18')}, "
           f"pretrained={s1_cfg.get('pretrained', True)}, "
           f"freeze_backbone={s1_cfg.get('freeze_backbone', False)})",
        2: f"Multi-Model Collaboration (box_hidden={s2_cfg.get('box_head_hidden_dim', 256)}, "
           f"action_cls={s2_cfg.get('action_cls_dim', 512)}, "
           f"object_cls={s2_cfg.get('object_cls_dim', 512)})",
    }

    logger.info("=" * 60)
    logger.info("  Annotation Model Training")
    logger.info("=" * 60)
    logger.info(f"  Scheme:       {scheme} - {scheme_desc[scheme]}")
    logger.info(f"  Epochs:       {config['epoch']}")
    logger.info(f"  Batch size:   {config['batch_size']}")
    logger.info(f"  Learning rate:{config['lr']}")
    logger.info(f"  Seed:         {seed}")
    logger.info(f"  Device:       {device}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    img_size = (args.img_size, args.img_size)
    train_ds, test_ds = build_annotation_datasets(
        config_path=args.config,
        img_size=img_size,
        augment_train=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=annotation_collate_fn,
        pin_memory=True,
        drop_last=True if len(train_ds) > config["batch_size"] else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=annotation_collate_fn,
        pin_memory=True,
    )

    # Validate dataset is not empty
    if len(train_ds) == 0:
        logger.error(
            "Training dataset is empty (0 samples)! This usually means:\n"
            "  1. The Data/ directory paths in config_annotation.yaml are incorrect\n"
            "  2. The image file naming convention doesn't match the parser\n"
            "  3. The action/object labels in config don't match the filenames\n"
            "Please check:\n"
            "  - Data paths exist: ./Data/Seen/Img_Train.txt, ./Data/Unseen/Img_Train.txt, etc.\n"
            "  - Image filenames follow the ObjectName_Action_Index.jpg convention\n"
            "  - All labels in data are listed in config affordance_labels/object_labels"
        )
        sys.exit(1)
    if len(test_ds) == 0:
        logger.warning("Test dataset is empty (0 samples). Validation will be skipped.")

    logger.info(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_annotation_model(config)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ------------------------------------------------------------------
    # Loss & Optimizer
    # ------------------------------------------------------------------
    loss_fn = build_annotation_loss(config)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=1e-5,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )

    # ------------------------------------------------------------------
    # TensorBoard writer
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scheme_tag = f"scheme{scheme}"
    pretrained_tag = ""
    if scheme == 1:
        pretrained_tag = "_pretrained" if s1_cfg.get("pretrained", True) else "_scratch"
        if s1_cfg.get("freeze_backbone", False):
            pretrained_tag += "_frozen"

    log_dir = config.get("log_dir", "./annotation/logs")
    log_dir = os.path.join(log_dir, f"{scheme_tag}{pretrained_tag}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard log: {log_dir}")

    # ------------------------------------------------------------------
    # Evaluation metrics (accuracy, NOT part of loss)
    # Use the pre-built reference embeddings from the dataset
    # ------------------------------------------------------------------
    metrics = None
    try:
        action_ref, object_ref = train_ds.get_reference_embeddings()
        from annotation_model import AnnotationMetrics
        metrics = AnnotationMetrics(
            action_ref_embeddings=torch.from_numpy(action_ref),
            object_ref_embeddings=torch.from_numpy(object_ref),
            device=device,
        )
        logger.info("AnnotationMetrics initialized (action_nn_acc, object_nn_acc, action_cls_acc, object_cls_acc)")
    except Exception as e:
        logger.warning(f"Could not build AnnotationMetrics: {e}. Accuracy will not be tracked.")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_loss", float("inf"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    save_dir = config.get("save_dir", "./annotation/checkpoints")
    num_epochs = config["epoch"]

    logger.info(f"\nStarting training from epoch {start_epoch} to {num_epochs} ...")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_losses, train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, writer,
            metrics=metrics, log_interval=20,
        )

        # Validate
        val_losses, val_metrics = validate(
            model, test_loader, loss_fn, device, epoch, writer,
            metrics=metrics,
        )

        # Scheduler step
        scheduler.step()

        # Epoch summary
        epoch_time = time.time() - epoch_start

        # Build accuracy string for logging
        acc_str = ""
        if train_metrics:
            acc_str += (
                f" | train_act_nn={train_metrics.get('action_nn_acc', 0):.3f}"
                f" train_obj_nn={train_metrics.get('object_nn_acc', 0):.3f}"
                f" train_act_cls={train_metrics.get('action_cls_acc', 0):.3f}"
                f" train_obj_cls={train_metrics.get('object_cls_acc', 0):.3f}"
            )
        if val_metrics:
            acc_str += (
                f" | val_act_nn={val_metrics.get('action_nn_acc', 0):.3f}"
                f" val_obj_nn={val_metrics.get('object_nn_acc', 0):.3f}"
                f" val_act_cls={val_metrics.get('action_cls_acc', 0):.3f}"
                f" val_obj_cls={val_metrics.get('object_cls_acc', 0):.3f}"
            )

        logger.info(
            f"Epoch [{epoch}/{num_epochs}] time={epoch_time:.1f}s | "
            f"train_loss={train_losses['total_loss']:.4f} | "
            f"val_loss={val_losses['total_loss']:.4f} | "
            f"train_box={train_losses['subject_box_loss'] + train_losses['object_box_loss']:.4f} | "
            f"val_box={val_losses['subject_box_loss'] + val_losses['object_box_loss']:.4f} | "
            f"train_embed={train_losses['action_embed_loss'] + train_losses['object_embed_loss']:.4f} | "
            f"val_embed={val_losses['action_embed_loss'] + val_losses['object_embed_loss']:.4f}"
            f"{acc_str} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save latest checkpoint every epoch
        '''
        ckpt_name = f"scheme{scheme}{pretrained_tag}_epoch_{epoch}.pth"
        save_checkpoint(
            model, optimizer, epoch, best_val_loss,
            os.path.join(save_dir, ckpt_name),
            scheme, config,
        )
        '''

        # Save best model
        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            best_name = f"scheme{scheme}{pretrained_tag}_best.pth"
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                os.path.join(save_dir, best_name),
                scheme, config,
            )
            logger.info(f"  >> New best model saved (val_loss={best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # Training complete
    # ------------------------------------------------------------------
    writer.close()
    logger.info("=" * 60)
    logger.info("  Training Complete!")
    logger.info(f"  Best validation loss: {best_val_loss:.4f}")
    logger.info(f"  Checkpoints saved in: {save_dir}")
    logger.info(f"  Logs saved in:        {log_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()