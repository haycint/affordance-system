"""
Annotation Dataset for Image Annotation Model
==============================================

Collects all image and annotation data from Data/ directory (Seen + Unseen, train + test).
Processes bounding boxes, action labels, and object names.
Loads GloVe word embeddings for action labels and object names.
Splits data into 85% train / 15% test.

Design:
    - During __init__: load config → load GloVe → resolve every label to its
      GloVe vector and class index → build lookup dictionaries.
    - During __getitem__: only dict lookups, no resolution logic.

Each sample returns:
    - img:          normalized image tensor (C, H, W)
    - subject_box:  [x1, y1, x2, y2] absolute coordinates (resized to img_size)
    - object_box:   [x1, y1, x2, y2] absolute coordinates (resized to img_size)
    - action_wv:    GloVe word vector for action label (300-dim)
    - object_wv:    GloVe word vector for object name   (300-dim)
    - action:       action label string
    - object_name:  object name string
    - action_idx:   int  index into affordance_labels
    - object_idx:   int  index into object_labels
"""

import os
import json
import random
import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GloVe loader
# ---------------------------------------------------------------------------

def load_glove_embeddings(glove_path: str, vocab: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from a text file.

    Args:
        glove_path: Path to glove.6B.300d.txt (or similar).
        vocab: If provided, only load embeddings for words in this list.
               Also loads sub-words from camelCase splits (e.g. StorageFurniture → storage, furniture).

    Returns:
        Dictionary mapping word (lowercase) -> 300-dim numpy array.
    """
    embeddings: Dict[str, np.ndarray] = {}

    # Build expanded vocab set: include sub-words from camelCase/compound labels
    expanded_vocab: Optional[set] = None
    if vocab is not None:
        expanded_vocab = set()
        for w in vocab:
            expanded_vocab.add(w.lower())
            # Split camelCase: StorageFurniture → storage, furniture
            sub_words = re.sub(r'([a-z])([A-Z])', r'\1 \2', w).split()
            for sw in sub_words:
                expanded_vocab.add(sw.lower())

    logger.info(f"Loading GloVe embeddings from {glove_path} ...")
    if not os.path.isfile(glove_path):
        raise FileNotFoundError(
            f"GloVe file not found at {glove_path}. "
            "Please download glove.6B.300d.txt and place it under ./glove/"
        )

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if expanded_vocab is not None and word.lower() not in expanded_vocab:
                continue
            vec = np.array(parts[1:], dtype=np.float32)
            if vec.shape[0] == 300:
                embeddings[word.lower()] = vec

    logger.info(f"Loaded {len(embeddings)} word vectors from GloVe.")
    return embeddings


# ---------------------------------------------------------------------------
# Helper: resolve a word to its GloVe vector (with fallbacks)
# ---------------------------------------------------------------------------

def resolve_glove_vector(
    word: str,
    glove_embeddings: Dict[str, np.ndarray],
    aliases: Dict[str, str],
    embed_dim: int = 300,
) -> np.ndarray:
    """
    Resolve a label word to its GloVe vector with fallback strategies:
      1. Exact match (lowercase) in glove_embeddings
      2. Alias mapping from config (glove_aliases)
      3. CamelCase / compound splitting → sub-word average
      4. Zero vector (last resort, with warning)

    This function is only called during __init__ to pre-build the dictionaries.
    """
    key = word.lower()

    # 1. Exact match
    if key in glove_embeddings:
        return glove_embeddings[key]
    if word in glove_embeddings:
        return glove_embeddings[word]

    # 2. Alias from config
    if key in aliases:
        alias = aliases[key]
        if alias in glove_embeddings:
            return glove_embeddings[alias]
        if alias.lower() in glove_embeddings:
            return glove_embeddings[alias.lower()]

    # 3. CamelCase / compound splitting → average sub-word vectors
    sub_words = re.sub(r'([a-z])([A-Z])', r'\1 \2', word).split()
    if len(sub_words) > 1:
        vecs = []
        for sw in sub_words:
            sw_key = sw.lower()
            if sw_key in glove_embeddings:
                vecs.append(glove_embeddings[sw_key])
        if vecs:
            logger.info(f"GloVe: '{word}' resolved via sub-word average of {sub_words}")
            return np.mean(vecs, axis=0)

    # 4. Zero vector
    logger.warning(f"GloVe vector not found for '{word}', using zero vector")
    return np.zeros(embed_dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def img_normalize_val():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ---------------------------------------------------------------------------
# Helper: read file list
# ---------------------------------------------------------------------------

def read_file_list(path: str) -> List[str]:
    """Read a text file containing one path per line."""
    file_list = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip("\n").strip()
            if line:
                file_list.append(line)
    return file_list


# ---------------------------------------------------------------------------
# Helper: parse annotation metadata from image path
# ---------------------------------------------------------------------------

def parse_img_path(img_path: str) -> Tuple[str, str]:
    """
    Parse object_name and action from image path.
    Naming convention: ObjectName_action_XXXX.jpg
    Steps:
        1. Extract just the filename (basename) to avoid directory underscores
        2. Remove file extension
        3. Split by underscore: parts[0]=object_name, parts[1]=action
    """
    basename = os.path.basename(img_path)                # e.g. "Dishwasher_grasp_0001.jpg"
    name_without_ext = os.path.splitext(basename)[0]     # e.g. "Dishwasher_grasp_0001"
    parts = name_without_ext.split("_")
    # Format: ObjectName_Action_Index
    if len(parts) >= 3:
        object_name = parts[-3]
        action = parts[-2]
    elif len(parts) == 2:
        object_name = parts[0]
        action = parts[1]
    else:
        object_name = name_without_ext
        action = ""
    return object_name, action


# ---------------------------------------------------------------------------
# Helper: load box annotation from JSON
# ---------------------------------------------------------------------------

def load_box_annotation(json_path: str) -> Tuple[List[float], List[float]]:
    """
    Load subject and object bounding boxes from a LabelMe-style JSON file.

    Returns:
        subject_box: [x1, y1, x2, y2] in original image coordinates
        object_box:  [x1, y1, x2, y2] in original image coordinates
    """
    with open(json_path, "r") as f:
        json_data = json.load(f)

    sub_points: List = []
    obj_points: List = []

    for shape in json_data.get("shapes", []):
        if shape["label"] == "subject":
            sub_points = shape["points"]
        elif shape["label"] == "object":
            obj_points = shape["points"]

    # Default empty box if missing
    if len(sub_points) == 0:
        sub_points = [[0.0, 0.0], [0.0, 0.0]]
    if len(obj_points) == 0:
        obj_points = [[0.0, 0.0], [0.0, 0.0]]

    subject_box = [sub_points[0][0], sub_points[0][1],
                   sub_points[1][0], sub_points[1][1]]
    object_box = [obj_points[0][0], obj_points[0][1],
                  obj_points[1][0], obj_points[1][1]]

    return subject_box, object_box


# ---------------------------------------------------------------------------
# Helper: resize boxes to target image size
# ---------------------------------------------------------------------------

def resize_box(box: List[float], orig_size: Tuple[int, int],
               target_size: Tuple[int, int]) -> List[float]:
    """
    Resize bounding box coordinates when image is resized.
    """
    orig_h, orig_w = orig_size
    target_h, target_w = target_size
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w

    return [
        box[0] * scale_w,
        box[1] * scale_h,
        box[2] * scale_w,
        box[3] * scale_h,
    ]


# ---------------------------------------------------------------------------
# Helper: resolve a raw label string to a registered label (case-insensitive)
# ---------------------------------------------------------------------------

def resolve_label(raw: str, exact_map: Dict[str, int], lower_map: Dict[str, int],
                  labels_list: List[str]) -> Optional[str]:
    """
    Try exact match first, then case-insensitive match.
    Returns the registered label string if found, else None.
    """
    if raw in exact_map:
        return raw
    raw_lower = raw.lower()
    if raw_lower in lower_map:
        idx = lower_map[raw_lower]
        return labels_list[idx]
    return None


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class AnnotationDataset(Dataset):
    """
    Annotation Dataset that collects ALL data from Data/Seen and Data/Unseen
    (both train and test splits) into a single pool, then splits 85/15.

    All GloVe vectors and class indices are resolved during __init__ and
    stored in pre-built dictionaries. __getitem__ only performs simple
    dict lookups.
    """

    def __init__(
        self,
        config_path: str,
        split: str = "train",           # "train" or "test"
        img_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        glove_cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            config_path: Path to config_annotation.yaml
            split: "train" or "test" (85% / 15% split)
            img_size: Target image size (H, W)
            augment: Whether to apply data augmentation (train only)
            glove_cache: Pre-loaded GloVe dict to avoid double-loading
        """
        super().__init__()

        assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"

        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")

        # ==============================================================
        # 1. Load config
        # ==============================================================
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.affordance_labels: List[str] = config["affordance_labels"]
        self.object_labels: List[str] = config["object_labels"]
        self.word_embed_dim: int = config.get("word_embed_dim", 300)

        # Label → index dictionaries (exact-case)
        self.affordance2idx: Dict[str, int] = {label: i for i, label in enumerate(self.affordance_labels)}
        self.object2idx: Dict[str, int] = {label: i for i, label in enumerate(self.object_labels)}
        # Case-insensitive label → index (for filename matching)
        affordance2idx_lower: Dict[str, int] = {label.lower(): i for i, label in enumerate(self.affordance_labels)}
        object2idx_lower: Dict[str, int] = {label.lower(): i for i, label in enumerate(self.object_labels)}

        logger.info(f"Registered {len(self.affordance_labels)} affordance labels: {self.affordance_labels}")
        logger.info(f"Registered {len(self.object_labels)} object labels: {self.object_labels}")

        glove_aliases: Dict[str, str] = config.get("glove_aliases", {})
        train_ratio = config.get("train_ratio", 0.85)
        seed = config.get("seed", 42)

        # ==============================================================
        # 2. Load GloVe embeddings (once, shared via cache)
        # ==============================================================
        glove_path = config["glove_path"]
        if glove_cache is not None:
            glove_embeddings = glove_cache
        else:
            vocab = [w.lower() for w in self.affordance_labels + self.object_labels]
            glove_embeddings = load_glove_embeddings(glove_path, vocab=vocab)

        # Keep a reference for get_glove_dict()
        self.glove_embeddings = glove_embeddings

        # ==============================================================
        # 3. Build label → GloVe vector dictionaries (resolved at init)
        #    action_wv_dict: registered_label → np.ndarray (300,)
        #    object_wv_dict: registered_label → np.ndarray (300,)
        # ==============================================================
        self.action_wv_dict: Dict[str, np.ndarray] = {}
        for label in self.affordance_labels:
            self.action_wv_dict[label] = resolve_glove_vector(
                label, glove_embeddings, glove_aliases, self.word_embed_dim
            )

        self.object_wv_dict: Dict[str, np.ndarray] = {}
        for label in self.object_labels:
            self.object_wv_dict[label] = resolve_glove_vector(
                label, glove_embeddings, glove_aliases, self.word_embed_dim
            )

        logger.info(f"Built GloVe dictionaries: {len(self.action_wv_dict)} actions, {len(self.object_wv_dict)} objects")

        # ==============================================================
        # 4. Collect all data entries from Seen and Unseen (train + test)
        #    Each entry stores the resolved (registered) labels.
        # ==============================================================
        all_entries: List[Dict] = []
        skipped_unknown = 0

        for setting in ("seen", "unseen"):
            cfg = config[setting]
            for phase in ("train", "test"):
                img_txt = cfg[f"img_{phase}"]
                box_txt = cfg[f"box_{phase}"]

                if not os.path.isfile(img_txt):
                    logger.warning(f"File not found, skipping: {img_txt}")
                    continue
                if not os.path.isfile(box_txt):
                    logger.warning(f"File not found, skipping: {box_txt}")
                    continue

                img_files = read_file_list(img_txt)
                box_files = read_file_list(box_txt)

                logger.info(f"{setting}/{phase}: Found {len(img_files)} images, {len(box_files)} boxes")

                # Log first image path for debugging
                if img_files:
                    logger.info(f"  First image path example: {img_files[0]}")
                    parsed_obj, parsed_act = parse_img_path(img_files[0])
                    logger.info(f"  Parsed from path: object={parsed_obj}, action={parsed_act}")

                assert len(img_files) == len(box_files), (
                    f"Mismatch: {len(img_files)} images vs {len(box_files)} boxes "
                    f"in {setting}/{phase}"
                )

                # Track first few skipped entries for diagnostics
                first_skipped_examples = []

                for img_path, box_path in zip(img_files, box_files):
                    raw_obj, raw_act = parse_img_path(img_path)

                    # Resolve raw parsed strings to registered labels
                    resolved_action = resolve_label(
                        raw_act, self.affordance2idx, affordance2idx_lower, self.affordance_labels
                    )
                    resolved_object = resolve_label(
                        raw_obj, self.object2idx, object2idx_lower, self.object_labels
                    )

                    if resolved_action is None or resolved_object is None:
                        skipped_unknown += 1
                        if len(first_skipped_examples) < 5:
                            first_skipped_examples.append(
                                f"  img={img_path} -> parsed(object={raw_obj}, action={raw_act}) "
                                f"resolved(action={resolved_action}, object={resolved_object})"
                            )
                        continue

                    all_entries.append({
                        "img_path": img_path,
                        "box_path": box_path,
                        "action": resolved_action,
                        "object_name": resolved_object,
                    })

                if first_skipped_examples:
                    logger.warning(
                        f"{setting}/{phase}: Skipped entries due to unknown labels. "
                        f"First examples:\n" + "\n".join(first_skipped_examples)
                    )

        if skipped_unknown > 0:
            logger.warning(
                f"Skipped {skipped_unknown} entries with unknown action/object labels. "
                "Ensure all data labels are listed in config affordance_labels/object_labels."
            )

        logger.info(f"Total collected entries: {len(all_entries)}")

        # ==============================================================
        # 5. 85% / 15% split (deterministic)
        # ==============================================================
        rng = random.Random(seed)
        indices = list(range(len(all_entries)))
        rng.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        if split == "train":
            selected_indices = indices[:n_train]
        else:
            selected_indices = indices[n_train:]

        self.entries = [all_entries[i] for i in selected_indices]
        logger.info(f"AnnotationDataset split='{split}': {len(self.entries)} samples")

        # ==============================================================
        # 6. Image transforms
        # ==============================================================
        if self.augment:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = img_normalize_val()

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with all sample data.
        All label-derived values (action_wv, object_wv, action_idx, object_idx)
        are obtained via simple dict lookups — no resolution logic here.
        """
        entry = self.entries[index]
        img_path = entry["img_path"]
        box_path = entry["box_path"]
        action = entry["action"]
        object_name = entry["object_name"]

        # --- Load image ---
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError:
            logger.warning(f"Cannot open image: {img_path}, returning next sample")
            return self.__getitem__((index + 1) % len(self))

        orig_w, orig_h = img.size  # PIL returns (W, H)

        # --- Load bounding boxes (original coordinates) ---
        subject_box, object_box = load_box_annotation(box_path)

        # --- Resize boxes to target image size ---
        subject_box = resize_box(subject_box, (orig_h, orig_w), self.img_size)
        object_box = resize_box(object_box, (orig_h, orig_w), self.img_size)

        # --- Resize and normalize image ---
        img = img.resize((self.img_size[1], self.img_size[0]))  # (W, H)
        img_tensor = self.transform(img)

        # --- Dict lookups (all resolved at init time) ---
        action_wv = self.action_wv_dict[action]
        object_wv = self.object_wv_dict[object_name]
        action_idx = self.affordance2idx[action]
        object_idx = self.object2idx[object_name]

        return {
            "img": img_tensor,
            "subject_box": torch.tensor(subject_box, dtype=torch.float32),
            "object_box": torch.tensor(object_box, dtype=torch.float32),
            "action_wv": torch.tensor(action_wv, dtype=torch.float32),
            "object_wv": torch.tensor(object_wv, dtype=torch.float32),
            "action": action,
            "object_name": object_name,
            "action_idx": action_idx,
            "object_idx": object_idx,
        }

    # ------------------------------------------------------------------
    # Public utility
    # ------------------------------------------------------------------

    def get_glove_dict(self) -> Dict[str, np.ndarray]:
        """Return the loaded GloVe dictionary for external use."""
        return self.glove_embeddings

    def get_reference_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build reference embedding matrices for action labels and object labels
        using the pre-built dictionaries (no re-resolution needed).

        Returns:
            action_ref: (num_actions, embed_dim) numpy array
            object_ref: (num_objects, embed_dim) numpy array
        """
        action_ref = np.stack([self.action_wv_dict[label] for label in self.affordance_labels])
        object_ref = np.stack([self.object_wv_dict[label] for label in self.object_labels])
        return action_ref, object_ref


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def annotation_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate that stacks tensor fields and gathers string fields into lists.
    """
    tensor_keys = ["img", "subject_box", "object_box", "action_wv", "object_wv"]
    string_keys = ["action", "object_name"]
    int_keys = ["action_idx", "object_idx"]

    result = {}
    for k in tensor_keys:
        result[k] = torch.stack([item[k] for item in batch])
    for k in string_keys:
        result[k] = [item[k] for item in batch]
    for k in int_keys:
        result[k] = torch.tensor([item[k] for item in batch], dtype=torch.long)

    return result


# ---------------------------------------------------------------------------
# Convenience: build train / test datasets
# ---------------------------------------------------------------------------

def build_annotation_datasets(
    config_path: str,
    img_size: Tuple[int, int] = (224, 224),
    augment_train: bool = True,
) -> Tuple[AnnotationDataset, AnnotationDataset]:
    """
    Build train and test AnnotationDataset instances sharing the same GloVe cache.

    Returns:
        (train_dataset, test_dataset)
    """
    # Pre-load GloVe once
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vocab = [w.lower() for w in config["affordance_labels"] + config["object_labels"]]
    glove_cache = load_glove_embeddings(config["glove_path"], vocab=vocab)

    train_ds = AnnotationDataset(
        config_path=config_path,
        split="train",
        img_size=img_size,
        augment=augment_train,
        glove_cache=glove_cache,
    )

    test_ds = AnnotationDataset(
        config_path=config_path,
        split="test",
        img_size=img_size,
        augment=False,
        glove_cache=glove_cache,
    )

    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Main: quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config_path = os.path.join(os.path.dirname(__file__), "config_annotation.yaml")
    train_ds, test_ds = build_annotation_datasets(config_path)

    print(f"Train: {len(train_ds)} samples")
    print(f"Test:  {len(test_ds)} samples")

    if len(train_ds) > 0:
        sample = train_ds[0]
        print("\n--- Sample ---")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")

        # Verify reference embeddings
        action_ref, object_ref = train_ds.get_reference_embeddings()
        print(f"\n--- Reference Embeddings ---")
        print(f"  action_ref: {action_ref.shape}")
        print(f"  object_ref: {object_ref.shape}")
