"""
Annotation Model - Two Schemes
================================

Scheme 1: End-to-End ImageNet Model
------------------------------------
A single backbone (ResNet-18/50) extracts visual features, followed by
task-specific prediction heads for:
  - subject_box (4-dim)
  - object_box  (4-dim)
  - action_embed (300-dim GloVe-aligned)
  - object_embed (300-dim GloVe-aligned)

Total output: 4 + 4 + 300 + 300 = 608 dimensions.

Supports two training modes:
  a) From random initialization
  b) From ImageNet pretrained weights (feature extraction / fine-tuning)


Scheme 2: Multi-Model Collaboration
------------------------------------
Separate sub-models handle different aspects:
  - BoxHead:    predicts subject_box and object_box (region proposal style)
  - ActionHead: classifies action label, then maps to 300-dim GloVe space
  - ObjectHead: classifies object name, then maps to 300-dim GloVe space

Each sub-model can use a shared or separate backbone.
Total output: 4 + 4 + 300 + 300 = 608 dimensions.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


# ===========================================================================
# Shared building blocks
# ===========================================================================

class MLPHead(nn.Module):
    """Multi-layer perceptron head with optional BatchNorm and Dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list,
        out_dim: int,
        dropout: float = 0.3,
        use_bn: bool = True,
    ):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BoxRegressionHead(nn.Module):
    """Predicts a single bounding box [x1, y1, x2, y2] from feature vector."""

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = MLPHead(
            in_dim=in_dim,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            out_dim=4,
            dropout=0.2,
            use_bn=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EmbeddingRegressionHead(nn.Module):
    """
    Predicts a continuous word embedding vector (e.g. 300-dim GloVe-aligned).
    Optionally first classifies into discrete labels, then maps to embedding space.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        classify_first: bool = True,
    ):
        """
        Args:
            in_dim:       Input feature dimension
            num_classes:  Number of discrete classes (for classification loss)
            embed_dim:    Output embedding dimension (300 for GloVe)
            hidden_dim:   Hidden layer dimension
            classify_first: If True, first classifies then maps to embedding;
                            If False, directly regresses the embedding.
        """
        super().__init__()
        self.classify_first = classify_first
        self.embed_dim = embed_dim

        if classify_first:
            # Classification head
            self.cls_head = MLPHead(
                in_dim=in_dim,
                hidden_dims=[hidden_dim],
                out_dim=num_classes,
                dropout=0.3,
            )
            # Mapping from class logits to embedding space
            self.map_head = MLPHead(
                in_dim=num_classes,
                hidden_dims=[hidden_dim],
                out_dim=embed_dim,
                dropout=0.2,
            )
        else:
            # Direct regression from features to embedding
            self.reg_head = MLPHead(
                in_dim=in_dim,
                hidden_dims=[hidden_dim, hidden_dim // 2],
                out_dim=embed_dim,
                dropout=0.3,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: (B, embed_dim) - predicted embedding vector
            logits:    (B, num_classes) - classification logits (zero if classify_first=False)
        """
        if self.classify_first:
            logits = self.cls_head(x)
            embedding = self.map_head(logits)
            return embedding, logits
        else:
            embedding = self.reg_head(x)
            logits = torch.zeros(x.size(0), 0, device=x.device)
            return embedding, logits


# ===========================================================================
# Scheme 1: End-to-End ImageNet Model
# ===========================================================================

class AnnotationModelScheme1(nn.Module):
    """
    End-to-end annotation model based on ImageNet-pretrained backbone.

    Architecture:
        Backbone (ResNet) -> Global pooled features ->
            -> subject_box_head -> [x1, y1, x2, y2]
            -> object_box_head  -> [x1, y1, x2, y2]
            -> action_embed_head -> (300-dim, action_cls_logits)
            -> object_embed_head -> (300-dim, object_cls_logits)

    Output: 4 + 4 + 300 + 300 = 608 dims total
    """

    def __init__(
        self,
        num_actions: int = 17,
        num_objects: int = 29,
        embed_dim: int = 300,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_actions:      Number of action/affordance categories
            num_objects:      Number of object categories
            embed_dim:        Word embedding dimension (300 for GloVe)
            backbone:         "resnet18" or "resnet50"
            pretrained:       Whether to load ImageNet pretrained weights
            freeze_backbone:  Whether to freeze backbone parameters
        """
        super().__init__()

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # ---- Backbone ----
        if backbone == "resnet18":
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the classification head, keep features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, feat_dim, 1, 1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen (feature extraction mode).")

        # ---- Task-specific heads ----
        self.subject_box_head = BoxRegressionHead(in_dim=feat_dim, hidden_dim=256)
        self.object_box_head = BoxRegressionHead(in_dim=feat_dim, hidden_dim=256)

        self.action_embed_head = EmbeddingRegressionHead(
            in_dim=feat_dim,
            num_classes=num_actions,
            embed_dim=embed_dim,
            hidden_dim=512,
            classify_first=True,
        )
        self.object_embed_head = EmbeddingRegressionHead(
            in_dim=feat_dim,
            num_classes=num_objects,
            embed_dim=embed_dim,
            hidden_dim=512,
            classify_first=True,
        )

        logger.info(
            f"Scheme1 initialized: backbone={backbone}, pretrained={pretrained}, "
            f"freeze_backbone={freeze_backbone}, feat_dim={feat_dim}"
        )

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            img: (B, 3, H, W) normalized image tensor

        Returns:
            Dictionary with:
                "subject_box":   (B, 4)   predicted subject bounding box
                "object_box":    (B, 4)   predicted object bounding box
                "action_embed":  (B, 300) predicted action word embedding
                "object_embed":  (B, 300) predicted object word embedding
                "action_logits": (B, num_actions)   action classification logits
                "object_logits": (B, num_objects)   object classification logits
        """
        # Backbone features
        feat = self.backbone(img)  # (B, feat_dim, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, feat_dim)

        # Predictions
        subject_box = self.subject_box_head(feat)
        object_box = self.object_box_head(feat)
        action_embed, action_logits = self.action_embed_head(feat)
        object_embed, object_logits = self.object_embed_head(feat)

        return {
            "subject_box": subject_box,
            "object_box": object_box,
            "action_embed": action_embed,
            "object_embed": object_embed,
            "action_logits": action_logits,
            "object_logits": object_logits,
        }


# ===========================================================================
# Scheme 2: Multi-Model Collaboration
# ===========================================================================

class BoxSubModel(nn.Module):
    """
    Sub-model for bounding box prediction.
    Uses its own backbone (or shared) to predict subject and object boxes.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        feat_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()

        if backbone == "resnet18":
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.subject_box_head = BoxRegressionHead(in_dim=feat_dim, hidden_dim=hidden_dim)
        self.object_box_head = BoxRegressionHead(in_dim=feat_dim, hidden_dim=hidden_dim)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            subject_box: (B, 4)
            object_box:  (B, 4)
        """
        feat = self.backbone(img)
        feat = feat.view(feat.size(0), -1)
        return self.subject_box_head(feat), self.object_box_head(feat)


class ActionSubModel(nn.Module):
    """
    Sub-model for action classification and embedding regression.
    """

    def __init__(
        self,
        num_actions: int = 17,
        embed_dim: int = 300,
        backbone: str = "resnet18",
        pretrained: bool = True,
        cls_dim: int = 512,
    ):
        super().__init__()

        if backbone == "resnet18":
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embed_head = EmbeddingRegressionHead(
            in_dim=feat_dim,
            num_classes=num_actions,
            embed_dim=embed_dim,
            hidden_dim=cls_dim,
            classify_first=True,
        )

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_embed:  (B, 300)
            action_logits: (B, num_actions)
        """
        feat = self.backbone(img)
        feat = feat.view(feat.size(0), -1)
        return self.embed_head(feat)


class ObjectSubModel(nn.Module):
    """
    Sub-model for object classification and embedding regression.
    """

    def __init__(
        self,
        num_objects: int = 29,
        embed_dim: int = 300,
        backbone: str = "resnet18",
        pretrained: bool = True,
        cls_dim: int = 512,
    ):
        super().__init__()

        if backbone == "resnet18":
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embed_head = EmbeddingRegressionHead(
            in_dim=feat_dim,
            num_classes=num_objects,
            embed_dim=embed_dim,
            hidden_dim=cls_dim,
            classify_first=True,
        )

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            object_embed:  (B, 300)
            object_logits: (B, num_objects)
        """
        feat = self.backbone(img)
        feat = feat.view(feat.size(0), -1)
        return self.embed_head(feat)


class AnnotationModelScheme2(nn.Module):
    """
    Multi-model collaboration for annotation.

    Three sub-models work together:
      1. BoxSubModel    -> subject_box, object_box
      2. ActionSubModel -> action_embed, action_logits
      3. ObjectSubModel -> object_embed, object_logits

    Total output: 4 + 4 + 300 + 300 = 608 dimensions.
    """

    def __init__(
        self,
        num_actions: int = 17,
        num_objects: int = 29,
        embed_dim: int = 300,
        backbone: str = "resnet18",
        pretrained: bool = True,
        box_head_hidden_dim: int = 256,
        action_cls_dim: int = 512,
        object_cls_dim: int = 512,
        share_backbone: bool = False,
    ):
        """
        Args:
            num_actions:         Number of action/affordance categories
            num_objects:         Number of object categories
            embed_dim:           Word embedding dimension (300)
            backbone:            Backbone architecture
            pretrained:          Load ImageNet pretrained weights
            box_head_hidden_dim: Hidden dim for box regression heads
            action_cls_dim:      Hidden dim for action classification
            object_cls_dim:      Hidden dim for object classification
            share_backbone:      If True, all sub-models share a single backbone
        """
        super().__init__()

        self.share_backbone = share_backbone

        if share_backbone:
            # Shared backbone
            if backbone == "resnet18":
                resnet = models.resnet18(
                    weights=models.ResNet18_Weights.DEFAULT if pretrained else None
                )
                feat_dim = 512
            elif backbone == "resnet50":
                resnet = models.resnet50(
                    weights=models.ResNet50_Weights.DEFAULT if pretrained else None
                )
                feat_dim = 2048
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

            self.shared_backbone = nn.Sequential(*list(resnet.children())[:-1])

            # Lightweight heads on top of shared features
            self.subject_box_head = BoxRegressionHead(feat_dim, box_head_hidden_dim)
            self.object_box_head = BoxRegressionHead(feat_dim, box_head_hidden_dim)
            self.action_embed_head = EmbeddingRegressionHead(
                feat_dim, num_actions, embed_dim, action_cls_dim, classify_first=True
            )
            self.object_embed_head = EmbeddingRegressionHead(
                feat_dim, num_objects, embed_dim, object_cls_dim, classify_first=True
            )
        else:
            # Separate sub-models with independent backbones
            self.box_sub_model = BoxSubModel(
                backbone=backbone,
                pretrained=pretrained,
                hidden_dim=box_head_hidden_dim,
            )
            self.action_sub_model = ActionSubModel(
                num_actions=num_actions,
                embed_dim=embed_dim,
                backbone=backbone,
                pretrained=pretrained,
                cls_dim=action_cls_dim,
            )
            self.object_sub_model = ObjectSubModel(
                num_objects=num_objects,
                embed_dim=embed_dim,
                backbone=backbone,
                pretrained=pretrained,
                cls_dim=object_cls_dim,
            )

        logger.info(
            f"Scheme2 initialized: backbone={backbone}, pretrained={pretrained}, "
            f"share_backbone={share_backbone}"
        )

    def _extract_shared_features(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.shared_backbone(img)
        return feat.view(feat.size(0), -1)

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            img: (B, 3, H, W) normalized image tensor

        Returns:
            Dictionary with same keys as Scheme1:
                "subject_box", "object_box", "action_embed", "object_embed",
                "action_logits", "object_logits"
        """
        if self.share_backbone:
            feat = self._extract_shared_features(img)
            subject_box = self.subject_box_head(feat)
            object_box = self.object_box_head(feat)
            action_embed, action_logits = self.action_embed_head(feat)
            object_embed, object_logits = self.object_embed_head(feat)
        else:
            subject_box, object_box = self.box_sub_model(img)
            action_embed, action_logits = self.action_sub_model(img)
            object_embed, object_logits = self.object_sub_model(img)

        return {
            "subject_box": subject_box,
            "object_box": object_box,
            "action_embed": action_embed,
            "object_embed": object_embed,
            "action_logits": action_logits,
            "object_logits": object_logits,
        }


# ===========================================================================
# Loss computation
# ===========================================================================

class AnnotationLoss(nn.Module):
    """
    Combined loss for the annotation model.

    Losses:
      1. subject_box_loss: Smooth L1 loss for subject bounding box
      2. object_box_loss:  Smooth L1 loss for object bounding box
      3. action_embed_loss: MSE loss between predicted and GT action embedding
      4. object_embed_loss: MSE loss between predicted and GT object embedding
      5. action_cls_loss:  Cross-entropy loss for action classification (optional)
      6. object_cls_loss:  Cross-entropy loss for object classification (optional)

    Each loss is weighted and computed independently.
    """

    def __init__(
        self,
        w_subject_box: float = 1.0,
        w_object_box: float = 1.0,
        w_action_embed: float = 1.0,
        w_object_embed: float = 1.0,
        w_action_cls: float = 0.5,
        w_object_cls: float = 0.5,
    ):
        super().__init__()
        self.w_subject_box = w_subject_box
        self.w_object_box = w_object_box
        self.w_action_embed = w_action_embed
        self.w_object_embed = w_object_embed
        self.w_action_cls = w_action_cls
        self.w_object_cls = w_object_cls

        self.box_loss_fn = nn.SmoothL1Loss()
        self.embed_loss_fn = nn.MSELoss()
        # ignore_index=-1: skip samples with unknown labels (idx=-1)
        self.cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model output dict with keys:
                "subject_box", "object_box", "action_embed", "object_embed",
                "action_logits", "object_logits"
            targets: Ground truth dict with keys:
                "subject_box", "object_box", "action_wv", "object_wv",
                "action_idx", "object_idx"

        Returns:
            Dict with:
                "total_loss":      weighted sum
                "subject_box_loss": scalar
                "object_box_loss":  scalar
                "action_embed_loss": scalar
                "object_embed_loss": scalar
                "action_cls_loss":  scalar (0 if no logits)
                "object_cls_loss":  scalar (0 if no logits)
        """
        # Box losses
        subject_box_loss = self.box_loss_fn(
            predictions["subject_box"], targets["subject_box"]
        )
        object_box_loss = self.box_loss_fn(
            predictions["object_box"], targets["object_box"]
        )

        # Embedding losses
        action_embed_loss = self.embed_loss_fn(
            predictions["action_embed"], targets["action_wv"]
        )
        object_embed_loss = self.embed_loss_fn(
            predictions["object_embed"], targets["object_wv"]
        )

        # Classification losses (if logits are available)
        action_cls_loss = torch.tensor(0.0, device=subject_box_loss.device)
        object_cls_loss = torch.tensor(0.0, device=subject_box_loss.device)

        if predictions.get("action_logits") is not None and predictions["action_logits"].numel() > 0:
            action_cls_loss = self.cls_loss_fn(
                predictions["action_logits"], targets["action_idx"]
            )
        if predictions.get("object_logits") is not None and predictions["object_logits"].numel() > 0:
            object_cls_loss = self.cls_loss_fn(
                predictions["object_logits"], targets["object_idx"]
            )

        # Weighted total
        total_loss = (
            self.w_subject_box * subject_box_loss
            + self.w_object_box * object_box_loss
            + self.w_action_embed * action_embed_loss
            + self.w_object_embed * object_embed_loss
            + self.w_action_cls * action_cls_loss
            + self.w_object_cls * object_cls_loss
        )

        return {
            "total_loss": total_loss,
            "subject_box_loss": subject_box_loss,
            "object_box_loss": object_box_loss,
            "action_embed_loss": action_embed_loss,
            "object_embed_loss": object_embed_loss,
            "action_cls_loss": action_cls_loss,
            "object_cls_loss": object_cls_loss,
        }


# ===========================================================================
# Evaluation metrics (NOT part of loss, for performance monitoring only)
# ===========================================================================

class AnnotationMetrics:
    """
    Evaluation metrics for the annotation model.

    Computes:
      - action_nn_acc:  Nearest-neighbor classification accuracy for action
                        embeddings (cosine similarity with GloVe reference)
      - object_nn_acc:  Nearest-neighbor classification accuracy for object
                        embeddings (cosine similarity with GloVe reference)
      - action_cls_acc: Direct classification accuracy from action_logits (argmax)
      - object_cls_acc: Direct classification accuracy from object_logits (argmax)

    These metrics do NOT participate in loss computation — they only indicate
    model performance.
    """

    def __init__(
        self,
        action_ref_embeddings: torch.Tensor,   # (num_actions, embed_dim)
        object_ref_embeddings: torch.Tensor,    # (num_objects, embed_dim)
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            action_ref_embeddings: L2-normalized GloVe embeddings for each
                                   action label, shape (num_actions, embed_dim).
            object_ref_embeddings: L2-normalized GloVe embeddings for each
                                   object label, shape (num_objects, embed_dim).
            device:                Torch device.
        """
        # Pre-normalize reference embeddings for cosine similarity
        self.action_ref = F.normalize(action_ref_embeddings.to(device), dim=1)
        self.object_ref = F.normalize(object_ref_embeddings.to(device), dim=1)
        self.device = device

    @torch.no_grad()
    def compute(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute accuracy metrics for one batch.

        Args:
            predictions: Model output dict (must contain action_embed, object_embed,
                         action_logits, object_logits).
            targets:     Ground-truth dict (must contain action_idx, object_idx).

        Returns:
            Dict of float scalars:
                action_nn_acc, object_nn_acc,
                action_cls_acc, object_cls_acc
        """
        # --- Action nearest-neighbor accuracy ---
        pred_action_norm = F.normalize(predictions["action_embed"], dim=1)
        action_sim = pred_action_norm @ self.action_ref.T   # (B, num_actions)
        action_nn_pred = action_sim.argmax(dim=1)
        action_nn_acc = (action_nn_pred == targets["action_idx"]).float().mean().item()

        # --- Object nearest-neighbor accuracy ---
        pred_object_norm = F.normalize(predictions["object_embed"], dim=1)
        object_sim = pred_object_norm @ self.object_ref.T   # (B, num_objects)
        object_nn_pred = object_sim.argmax(dim=1)
        object_nn_acc = (object_nn_pred == targets["object_idx"]).float().mean().item()

        # --- Direct classification accuracy from logits ---
        action_cls_acc = 0.0
        object_cls_acc = 0.0

        if predictions.get("action_logits") is not None and predictions["action_logits"].numel() > 0:
            action_cls_acc = (
                predictions["action_logits"].argmax(dim=1) == targets["action_idx"]
            ).float().mean().item()

        if predictions.get("object_logits") is not None and predictions["object_logits"].numel() > 0:
            object_cls_acc = (
                predictions["object_logits"].argmax(dim=1) == targets["object_idx"]
            ).float().mean().item()

        return {
            "action_nn_acc": action_nn_acc,
            "object_nn_acc": object_nn_acc,
            "action_cls_acc": action_cls_acc,
            "object_cls_acc": object_cls_acc,
        }



# ===========================================================================
# Model factory
# ===========================================================================

def build_annotation_model(config: dict) -> nn.Module:
    """
    Build annotation model based on config.

    Args:
        config: Parsed YAML config dict (from config_annotation.yaml)

    Returns:
        nn.Module: Either AnnotationModelScheme1 or AnnotationModelScheme2
    """
    scheme = config.get("scheme", 1)
    num_actions = len(config["affordance_labels"])
    num_objects = len(config["object_labels"])
    embed_dim = config.get("word_embed_dim", 300)

    if scheme == 1:
        s1_cfg = config.get("scheme1", {})
        model = AnnotationModelScheme1(
            num_actions=num_actions,
            num_objects=num_objects,
            embed_dim=embed_dim,
            backbone=s1_cfg.get("backbone", "resnet18"),
            pretrained=s1_cfg.get("pretrained", True),
            freeze_backbone=s1_cfg.get("freeze_backbone", False),
        )
    elif scheme == 2:
        s2_cfg = config.get("scheme2", {})
        model = AnnotationModelScheme2(
            num_actions=num_actions,
            num_objects=num_objects,
            embed_dim=embed_dim,
            backbone=config.get("scheme1", {}).get("backbone", "resnet18"),
            pretrained=config.get("scheme1", {}).get("pretrained", True),
            box_head_hidden_dim=s2_cfg.get("box_head_hidden_dim", 256),
            action_cls_dim=s2_cfg.get("action_cls_dim", 512),
            object_cls_dim=s2_cfg.get("object_cls_dim", 512),
            share_backbone=False,
        )
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Must be 1 or 2.")

    return model


def build_annotation_loss(config: dict) -> AnnotationLoss:
    """Build the loss module from config weights."""
    lw = config.get("loss_weights", {})
    return AnnotationLoss(
        w_subject_box=lw.get("subject_box", 1.0),
        w_object_box=lw.get("object_box", 1.0),
        w_action_embed=lw.get("action_embed", 1.0),
        w_object_embed=lw.get("object_embed", 1.0),
    )


# ===========================================================================
# Main: quick architecture test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing Scheme 1: End-to-End ImageNet Model")
    print("=" * 60)

    model1 = AnnotationModelScheme1(
        num_actions=17, num_objects=29,
        backbone="resnet18", pretrained=False
    )
    dummy_img = torch.randn(2, 3, 224, 224)
    out1 = model1(dummy_img)
    for k, v in out1.items():
        print(f"  {k}: {v.shape}")

    total_params1 = sum(p.numel() for p in model1.parameters())
    trainable1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"  Total params: {total_params1:,}  Trainable: {trainable1:,}")

    print()
    print("=" * 60)
    print("Testing Scheme 2: Multi-Model Collaboration")
    print("=" * 60)

    model2 = AnnotationModelScheme2(
        num_actions=17, num_objects=29,
        backbone="resnet18", pretrained=False, share_backbone=False
    )
    out2 = model2(dummy_img)
    for k, v in out2.items():
        print(f"  {k}: {v.shape}")

    total_params2 = sum(p.numel() for p in model2.parameters())
    trainable2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"  Total params: {total_params2:,}  Trainable: {trainable2:,}")

    print()
    print("=" * 60)
    print("Testing Loss Module")
    print("=" * 60)

    loss_fn = AnnotationLoss()
    targets = {
        "subject_box": torch.rand(2, 4),
        "object_box": torch.rand(2, 4),
        "action_wv": torch.rand(2, 300),
        "object_wv": torch.rand(2, 300),
        "action_idx": torch.randint(0, 17, (2,)),
        "object_idx": torch.randint(0, 29, (2,)),
    }
    losses = loss_fn(out1, targets)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")
