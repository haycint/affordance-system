"""
Standalone Training Script for IAG_TextEmb Model
=================================================

This script trains the IAG_TextEmb model, which extends IAG_Net by
incorporating pre-trained GloVe word embeddings for affordance labels
as additional semantic input to the model.

Key Features:
- Automatically downloads GloVe 6B 300d embeddings from Stanford NLP
- Converts affordance label text to GloVe vectors as model input
- Trains on the PIAD dataset (Seen setting by default)
- Implements complete evaluation with AUC, IOU, SIM, MAE metrics
- Supports checkpoint saving/resuming
- Follows original app1.py/backend.py training logic and config parameters

Usage:
    python train_textemb.py                                  # Train with defaults
    python train_textemb.py --epochs 80 --batch_size 8       # Custom params
    python train_textemb.py --resume ckpt/model.pt           # Resume training
    python train_textemb.py --eval_only ckpt/model.pt        # Evaluation only
"""

import os
import sys
import argparse
import time
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model.MyNet import get_IAG_TextEmb
from data_utils.dataset import PIAD
from utils.loss import HM_Loss, kl_div
from utils.eval import SIM

# ============================================================================
# Constants (consistent with backend.py and config_seen.yaml)
# ============================================================================

AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]

CKPT_DIR = os.path.join(PROJECT_ROOT, 'ckpt')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
GLOVE_DIR = os.path.join(PROJECT_ROOT, 'glove')

# Default config from config_seen.yaml
DEFAULT_CONFIG = {
    'Setting': 'Seen',
    'batch_size': 8,
    'lr': 0.0001,
    'Epoch': 80,
    'loss_cls': 0.3,
    'loss_kl': 0.5,
    'N_p': 64,
    'emb_dim': 512,
    'proj_dim': 512,
    'num_heads': 4,
    'N_raw': 2048,
    'num_affordance': 17,
    'pairing_num': 2,
    'text_dim': 300,   # GloVe 300d embedding
}


# ============================================================================
# GloVe Word Embedding Utilities
# ============================================================================

GLOVE_URLS = [
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://mirrors.tuna.tsinghua.edu.cn/stanford-nlp/data/glove.6B.zip",
    "https://mirror.sjtu.edu.cn/stanford-nlp/data/glove.6B.zip",
    "https://mirrors.bfsu.edu.cn/stanford-nlp/data/glove.6B.zip",
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    "https://hf-mirror.com/stanfordnlp/glove/resolve/main/glove.6B.zip"
]
GLOVE_6B_300D_FILE = "glove.6B.300d.txt"

def get_affordance_from_path(img_path, affordance_labels):
    """
    从图像路径中解析 affordance 名称。
    逻辑与 dataset.py 中的 get_affordance_label 完全一致。
    
    Args:
        img_path: 图像文件路径字符串
        affordance_labels: affordance 名称列表
    
    Returns:
        aff_name: 字符串，如果未找到则返回 None
    """
    filename = os.path.basename(img_path)
    parts = filename.split('_')
    if len(parts) >= 2:
        candidate = parts[-2]        # 与 dataset.py 中 cut_str[-2] 相同
        if candidate in affordance_labels:
            return candidate
    # 备选：扫描整个文件名，但通常不需要
    for aff in affordance_labels:
        if aff in filename:
            return aff
    return None

def download_glove(force=False, urls=None):
    """
    Download GloVe 6B embeddings from one of the available URLs.
    The zip file contains 50d, 100d, 200d, and 300d versions.
    We use 300d for the richest semantic representation.

    Args:
        force: If True, re-download even if files exist.
        urls: Optional list of download URLs to try in order.

    Returns:
        Path to the GloVe 300d text file.
    """
    os.makedirs(GLOVE_DIR, exist_ok=True)
    glove_300d_path = os.path.join(GLOVE_DIR, GLOVE_6B_300D_FILE)

    if os.path.exists(glove_300d_path) and not force:
        print(f"[GloVe] Found existing GloVe 300d at: {glove_300d_path}")
        return glove_300d_path

    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
    urls = urls or GLOVE_URLS

    # Download
    if not os.path.exists(zip_path) or force:
        print(f"[GloVe] Trying to download GloVe 6B embeddings from available mirrors...")
        print("[GloVe] This is a ~862MB download, please wait...")
        last_error = None
        for url in urls:
            try:
                print(f"[GloVe] Attempting: {url}")
                import urllib.request
                urllib.request.urlretrieve(url, zip_path)
                print(f"[GloVe] Download complete: {zip_path}")
                last_error = None
                break
            except Exception as e:
                last_error = e
                print(f"[GloVe] Download failed from {url}: {e}")
                continue

        if last_error is not None:
            print(f"[GloVe] All download attempts failed.")
            print("[GloVe] Please download manually from one of the following URLs:")
            for url in urls:
                print(f"        {url}")
            print(f"        Extract to: {GLOVE_DIR}")
            sys.exit(1)

    # Extract
    if not os.path.exists(glove_300d_path):
        print(f"[GloVe] Extracting {GLOVE_6B_300D_FILE} from zip...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(GLOVE_6B_300D_FILE, GLOVE_DIR)
        print(f"[GloVe] Extraction complete: {glove_300d_path}")

    return glove_300d_path


def load_glove_embeddings(glove_path, target_words=None):
    """
    Load GloVe word embeddings from text file.
    Only loads the words we need to save memory (if target_words specified).

    Args:
        glove_path: Path to GloVe text file (e.g. glove.6B.300d.txt)
        target_words: Set/list of words to load. If None, load all.

    Returns:
        dict: word -> numpy array (embedding vector)
    """
    print(f"[GloVe] Loading embeddings from: {glove_path}")
    embeddings = {}
    target_set = set(target_words) if target_words else None

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]

            # Skip words we don't need
            if target_set is not None and word not in target_set:
                continue

            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector

    print(f"[GloVe] Loaded {len(embeddings)} word vectors")
    return embeddings


def build_affordance_embeddings(glove_embeddings, affordance_labels):
    """
    Build affordance label embedding matrix from GloVe vectors.
    For compound words like 'wrapgrasp', we split and average the component
    word vectors (e.g. 'wrap' + 'grasp' -> average of their GloVe vectors).

    Args:
        glove_embeddings: dict from load_glove_embeddings()
        affordance_labels: list of affordance label strings

    Returns:
        emb_matrix: numpy array [num_affordance, text_dim]
        missing_words: list of words not found in GloVe
    """
    text_dim = next(iter(glove_embeddings.values())).shape[0]
    emb_matrix = np.zeros((len(affordance_labels), text_dim))
    missing_words = []

    for i, label in enumerate(affordance_labels):
        # Try direct lookup first
        if label in glove_embeddings:
            emb_matrix[i] = glove_embeddings[label]
        else:
            # Try splitting compound words (e.g. 'wrapgrasp' -> 'wrap' + 'grasp')
            # Common splitting patterns for PIAD affordance labels
            sub_words = _split_compound_word(label)
            vectors = []
            all_found = True
            for w in sub_words:
                if w in glove_embeddings:
                    vectors.append(glove_embeddings[w])
                else:
                    missing_words.append(w)
                    all_found = False

            if vectors:
                # Average the component word vectors
                emb_matrix[i] = np.mean(vectors, axis=0)
            else:
                # Fallback: random initialization for missing words
                print(f"[Warning] No GloVe vector for '{label}', using random init")
                emb_matrix[i] = np.random.randn(text_dim) * 0.1

    return emb_matrix, missing_words


def _split_compound_word(word):
    """
    Split compound affordance labels into component words.
    Uses a predefined mapping for known compounds and heuristics for others.

    Args:
        word: compound word string (e.g. 'wrapgrasp')

    Returns:
        list of component words
    """
    # Known compound word mappings for PIAD affordance labels
    compound_map = {
        'wrapgrasp': ['wrap', 'grasp'],
    }

    if word in compound_map:
        return compound_map[word]

    # Heuristic: try to split at common boundaries
    # For words that concatenate two known English words
    sub_words = []
    remaining = word

    # Try to find known English words by greedy prefix matching
    common_words = {
        'wrap', 'grasp', 'contain', 'lift', 'open', 'lay', 'sit',
        'support', 'pour', 'move', 'display', 'push', 'listen',
        'wear', 'press', 'cut', 'stab', 'hold', 'place', 'put',
        'turn', 'rotate', 'slide', 'pull', 'pick', 'hand', 'arm',
        'finger', 'grip', 'handle', 'carry', 'lift', 'drop',
    }

    while remaining:
        found = False
        for end in range(len(remaining), 0, -1):
            prefix = remaining[:end]
            if prefix in common_words:
                sub_words.append(prefix)
                remaining = remaining[end:]
                found = True
                break
        if not found:
            # Can't split further; keep the rest as one word
            sub_words.append(remaining)
            break

    return sub_words if len(sub_words) > 1 else [word]


# ============================================================================
# Training Utilities
# ============================================================================

def ensure_dir(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_device(use_gpu=True):
    """Get the appropriate compute device."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


def save_checkpoint(model, optimizer, scheduler, epoch, history, config,
                    model_name, setting, log_file_path, save_path):
    """Save a training checkpoint."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'history': history,
        'config': config,
        'model_name': model_name,
        'setting': setting,
        'log_file_path': log_file_path,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, criterion_hm, criterion_ce,
                    optimizer, device, config, affordance_emb_tensor, scaler=None):
    """
    训练一个 epoch
    - 支持批量处理多个点云（pair_num）
    - 支持混合精度训练（scaler）
    """
    model.train()
    num_batches = len(train_loader)
    loss_sum = 0.0
    pair_num = config['pairing_num']          # 默认为 2

    for i, (img, points, labels, logits_labels, sub_box, obj_box) in enumerate(train_loader):
        # ------------------------------------------------------------
        # 1. 拼接多个点云（在 batch 维度）
        # ------------------------------------------------------------
        # points: list of [B, 3, N] 长度 = pair_num
        points_concat = torch.cat(points, dim=0).to(device).float()          # [B*pair_num, 3, N]
        labels_concat = torch.cat(labels, dim=0).to(device).float().unsqueeze(-1)  # [B*pair_num, N, 1]
        logits_labels_concat = torch.cat(logits_labels, dim=0).to(device).long()    # [B*pair_num]

        # 扩展图像、边界框、文本嵌入
        B = img.size(0)
        img_expand = img.repeat_interleave(pair_num, dim=0).to(device).float()      # [B*pair_num, 3, H, W]
        sub_box_expand = sub_box.repeat_interleave(pair_num, dim=0).to(device).float()
        obj_box_expand = obj_box.repeat_interleave(pair_num, dim=0).to(device).float()
        text_emb = affordance_emb_tensor[logits_labels_concat].float()             # [B*pair_num, text_dim]

        # ------------------------------------------------------------
        # 2. 前向传播（可选混合精度）
        # ------------------------------------------------------------
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                _3d, logits, to_KL = model(img_expand, points_concat,
                                           sub_box_expand, obj_box_expand, text_emb)
                loss_hm = criterion_hm(_3d, labels_concat)
                loss_ce = criterion_ce(logits, logits_labels_concat)
                loss_kl = kl_div(to_KL[0], to_KL[1])
                total_loss = loss_hm + config['loss_cls'] * loss_ce + config['loss_kl'] * loss_kl
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _3d, logits, to_KL = model(img_expand, points_concat,
                                       sub_box_expand, obj_box_expand, text_emb)
            loss_hm = criterion_hm(_3d, labels_concat)
            loss_ce = criterion_ce(logits, logits_labels_concat)
            loss_kl = kl_div(to_KL[0], to_KL[1])
            total_loss = loss_hm + config['loss_cls'] * loss_ce + config['loss_kl'] * loss_kl
            total_loss.backward()
            optimizer.step()

        loss_sum += total_loss.item()
        if i % 10 == 0:
            print(f"  Batch {i}/{num_batches} | Loss: {total_loss.item():.4f}")

    return loss_sum / num_batches

def validate(model, val_loader, criterion_hm, criterion_ce,
             device, config, affordance_emb_tensor):
    """
    Validate the model and compute evaluation metrics.
    现在正确使用真实 affordance 对应的 GloVe 向量。
    """
    model.eval()
    val_dataset = val_loader.dataset
    results = torch.zeros((len(val_dataset), config['N_raw'], 1))
    targets = torch.zeros((len(val_dataset), config['N_raw'], 1))

    val_loss_sum = 0.0
    total_mae = 0.0
    total_points = 0
    num = 0

    with torch.no_grad():
        # 重要：val_loader 返回 7 个元素，顺序必须匹配 dataset.py
        for i, (img, point, label, img_path, point_path, sub_box, obj_box) in enumerate(val_loader):
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)

            batch_size = img_dev.size(0)
            text_emb = torch.zeros(batch_size, config['text_dim']).to(device)

            # 为 batch 中每个样本构建正确的 text embedding
            for b_idx, path in enumerate(img_path):
                aff_name = get_affordance_from_path(path, AFFORDANCE_LABELS)
                if aff_name is not None:
                    aff_idx = AFFORDANCE_LABELS.index(aff_name)
                    text_emb[b_idx] = affordance_emb_tensor[aff_idx]
                else:
                    # 首次出现时警告
                    if i == 0 and b_idx == 0:
                        print(f"[Warning] Cannot parse affordance from {path}. Using zero embedding.")

            # 前向传播
            _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)

            val_loss_hm = criterion_hm(_3d, label)
            val_loss_kl = kl_div(to_KL[0], to_KL[1])
            val_loss = val_loss_hm + config['loss_kl'] * val_loss_kl

            mae = torch.sum(torch.abs(_3d - label), dim=(0, 1))
            point_nums = _3d.shape[0] * _3d.shape[1]
            total_mae += mae.item()
            total_points += point_nums
            val_loss_sum += val_loss.item()

            pred_num = _3d.shape[0]
            results[num:num + pred_num, :, :] = _3d.cpu()
            targets[num:num + pred_num, :, :] = label.cpu()
            num += pred_num

    if total_points == 0:
        return 0, 0, 0, 0, 0

    val_mean_loss = val_loss_sum / len(val_loader)
    mean_mae = total_mae / total_points

    results_np = results.numpy()
    targets_np = targets.numpy()

    # 计算 SIM
    sim_values = np.zeros(targets_np.shape[0])
    for i in range(targets_np.shape[0]):
        sim_values[i] = SIM(results_np[i], targets_np[i])
    sim = np.mean(sim_values)

    # 计算 AUC 和 IOU
    auc_values = np.zeros(targets_np.shape[0])
    iou_values = np.zeros(targets_np.shape[0])
    iou_thres = np.linspace(0, 1, 20)
    targets_binary = (targets_np >= 0.5).astype(int)

    for i in range(targets_np.shape[0]):
        t_true = targets_binary[i]
        p_score = results_np[i]

        if np.sum(t_true) == 0:
            auc_values[i] = np.nan
            iou_values[i] = np.nan
        else:
            try:
                auc_values[i] = roc_auc_score(t_true.flatten(), p_score.flatten())
            except ValueError:
                auc_values[i] = np.nan

            temp_iou = []
            for thre in iou_thres:
                p_mask = (p_score >= thre).astype(int)
                intersect = np.sum(p_mask & t_true)
                union = np.sum(p_mask | t_true)
                temp_iou.append(1. * intersect / union if union > 0 else 0)
            iou_values[i] = np.mean(temp_iou)

    auc = np.nanmean(auc_values)
    iou = np.nanmean(iou_values)

    return val_mean_loss, auc, iou, sim, mean_mae

def validate1(model, val_loader, criterion_hm, criterion_ce,
             device, config, affordance_emb_tensor):
    """
    Validate the model and compute evaluation metrics.

    Metrics computed:
    - AUC (Area Under ROC Curve): measures discrimination ability
    - IOU (Intersection over Union): measures segmentation overlap
    - SIM (Similarity): measures histogram intersection
    - MAE (Mean Absolute Error): measures prediction accuracy
    - Val Loss: validation loss (HM + KL)

    Args:
        model: IAG_TextEmb model
        val_loader: DataLoader for validation data
        criterion_hm: HM_Loss
        criterion_ce: CrossEntropyLoss
        device: compute device
        config: training configuration dict
        affordance_emb_tensor: [num_affordance, text_dim] tensor on device

    Returns:
        val_loss, auc, iou, sim, mae
    """
    model.eval()
    val_dataset = val_loader.dataset
    results = torch.zeros((len(val_dataset), config['N_raw'], 1))
    targets = torch.zeros((len(val_dataset), config['N_raw'], 1))

    val_loss_sum = 0.0
    total_mae = 0.0
    total_points = 0
    num = 0

    with torch.no_grad():
        for i, (img, point, label, _, _, sub_box, obj_box) in enumerate(val_loader):
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)

            # For validation, we need the affordance index from the image path
            # The 4th return value from PIAD val is img_path which contains
            # the affordance name. We derive the index from it.
            # Since the val_loader returns: Img, Point, affordance_label, img_path, point_path, sub_box, obj_box
            # The 'label' already has the correct affordance column selected.
            # For text embedding, we need the affordance index.
            # We can get it from the label shape: the full affordance_label has 17 columns
            # But in val mode, label is already selected to one column.
            # We'll use a default text embedding (zeros) for val since we don't know
            # which affordance is being tested - this tests generalization.
            # Actually, we should use the correct affordance index.
            # Let's derive it from the original dataset.

            # Use zero text embedding for fair evaluation (model shouldn't
            # rely solely on text embedding at inference time).
            # Alternatively, we can provide the correct text embedding.
            # For consistency with the training regime, we provide the GT text emb.
            # The val dataset returns img_path at index 3, which contains affordance name.

            # Get affordance index from img_path (encoded in the batch)
            # Since DataLoader stacks strings as tuples, we need to handle this
            # We'll reconstruct the text embedding from the label dimensions
            # For simplicity, use a learned average text embedding
            batch_size = img_dev.size(0)
            text_emb = torch.zeros(batch_size, config['text_dim']).to(device)

            # Try to get affordance index from img_path if available
            try:
                # val_loader returns: Img, Point, affordance_label, img_path, point_path, sub_box, obj_box
                img_paths = _  # This is the 4th element (img_path) from the tuple
                if isinstance(_, (list, tuple)):
                    for b_idx, path in enumerate(_):
                        if isinstance(path, str):
                            aff_name = path.split('_')[-2]
                            if aff_name in AFFORDANCE_LABELS:
                                aff_idx = AFFORDANCE_LABELS.index(aff_name)
                                text_emb[b_idx] = affordance_emb_tensor[aff_idx]
            except Exception:
                pass  # Use zero text embedding as fallback

            _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)

            val_loss_hm = criterion_hm(_3d, label)
            val_loss_kl = kl_div(to_KL[0], to_KL[1])
            val_loss = val_loss_hm + config['loss_kl'] * val_loss_kl

            mae = torch.sum(torch.abs(_3d - label), dim=(0, 1))
            point_nums = _3d.shape[0] * _3d.shape[1]
            total_mae += mae.item()
            total_points += point_nums
            val_loss_sum += val_loss.item()

            pred_num = _3d.shape[0]
            results[num:num + pred_num, :, :] = _3d.cpu()
            targets[num:num + pred_num, :, :] = label.cpu()
            num += pred_num

    if total_points == 0:
        return 0, 0, 0, 0, 0

    val_mean_loss = val_loss_sum / len(val_loader)
    mean_mae = total_mae / total_points

    results_np = results.numpy()
    targets_np = targets.numpy()

    # Compute SIM
    sim_values = np.zeros(targets_np.shape[0])
    for i in range(targets_np.shape[0]):
        sim_values[i] = SIM(results_np[i], targets_np[i])
    sim = np.mean(sim_values)

    # Compute AUC and IOU
    auc_values = np.zeros(targets_np.shape[0])
    iou_values = np.zeros(targets_np.shape[0])
    iou_thres = np.linspace(0, 1, 20)
    targets_binary = (targets_np >= 0.5).astype(int)

    for i in range(targets_np.shape[0]):
        t_true = targets_binary[i]
        p_score = results_np[i]

        if np.sum(t_true) == 0:
            auc_values[i] = np.nan
            iou_values[i] = np.nan
        else:
            try:
                auc_values[i] = roc_auc_score(t_true.flatten(), p_score.flatten())
            except ValueError:
                auc_values[i] = np.nan

            temp_iou = []
            for thre in iou_thres:
                p_mask = (p_score >= thre).astype(int)
                intersect = np.sum(p_mask & t_true)
                union = np.sum(p_mask | t_true)
                temp_iou.append(1. * intersect / union if union > 0 else 0)
            iou_values[i] = np.mean(temp_iou)

    auc = np.nanmean(auc_values)
    iou = np.nanmean(iou_values)

    return val_mean_loss, auc, iou, sim, mean_mae

def evaluate_model(model, val_loader, device, config, affordance_emb_tensor):
    """
    完整的模型评估，支持每个 affordance 的详细指标。
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_aff_indices = []

    print("\n" + "=" * 60)
    print("EVALUATION: Running full evaluation...")
    print("=" * 60)

    with torch.no_grad():
        for i, (img, point, label, img_path, point_path, sub_box, obj_box) in enumerate(val_loader):
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)

            batch_size = img_dev.size(0)
            text_emb = torch.zeros(batch_size, config['text_dim']).to(device)
            batch_aff_indices = []

            for b_idx, path in enumerate(img_path):
                aff_name = get_affordance_from_path(path, AFFORDANCE_LABELS)
                if aff_name is not None:
                    aff_idx = AFFORDANCE_LABELS.index(aff_name)
                    text_emb[b_idx] = affordance_emb_tensor[aff_idx]
                    batch_aff_indices.append(aff_idx)
                else:
                    batch_aff_indices.append(-1)
                    if i == 0 and b_idx == 0:
                        print(f"[Warning] Cannot parse affordance from {path}. Using zero embedding.")

            all_aff_indices.extend(batch_aff_indices)

            _3d, logits, _ = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)

            all_preds.append(_3d.cpu())
            all_targets.append(label.cpu())

    # 合并所有 batch
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # 总体指标
    results = {}
    sim_values = np.zeros(all_preds.shape[0])
    mae_values = np.zeros(all_preds.shape[0])
    auc_values = np.zeros(all_preds.shape[0])
    iou_values = np.zeros(all_preds.shape[0])
    iou_thres = np.linspace(0, 1, 20)
    targets_binary = (all_targets >= 0.5).astype(int)

    for i in range(all_preds.shape[0]):
        sim_values[i] = SIM(all_preds[i], all_targets[i])
        mae_values[i] = np.sum(np.abs(all_preds[i] - all_targets[i])) / all_preds.shape[1]

        t_true = targets_binary[i]
        p_score = all_preds[i]
        if np.sum(t_true) == 0:
            auc_values[i] = np.nan
            iou_values[i] = np.nan
        else:
            try:
                auc_values[i] = roc_auc_score(t_true.flatten(), p_score.flatten())
            except ValueError:
                auc_values[i] = np.nan
            temp_iou = []
            for thre in iou_thres:
                p_mask = (p_score >= thre).astype(int)
                intersect = np.sum(p_mask & t_true)
                union = np.sum(p_mask | t_true)
                temp_iou.append(1. * intersect / union if union > 0 else 0)
            iou_values[i] = np.mean(temp_iou)

    results['overall'] = {
        'AUC': np.nanmean(auc_values),
        'IOU': np.nanmean(iou_values),
        'SIM': np.mean(sim_values),
        'MAE': np.mean(mae_values),
    }

    # 每个 affordance 的指标
    per_affordance = {}
    for aff_idx, aff_name in enumerate(AFFORDANCE_LABELS):
        indices = [i for i, idx in enumerate(all_aff_indices) if idx == aff_idx]
        if len(indices) == 0:
            continue
        aff_auc = np.nanmean(auc_values[indices]) if len(indices) > 0 else float('nan')
        aff_iou = np.nanmean(iou_values[indices]) if len(indices) > 0 else float('nan')
        aff_sim = np.mean(sim_values[indices]) if len(indices) > 0 else float('nan')
        aff_mae = np.mean(mae_values[indices]) if len(indices) > 0 else float('nan')
        per_affordance[aff_name] = {
            'AUC': aff_auc,
            'IOU': aff_iou,
            'SIM': aff_sim,
            'MAE': aff_mae,
            'count': len(indices),
        }

    results['per_affordance'] = per_affordance
    return results

def evaluate_model1(model, val_loader, device, config, affordance_emb_tensor):
    """
    Comprehensive model evaluation with per-affordance metrics.

    Args:
        model: trained IAG_TextEmb model
        val_loader: validation DataLoader
        device: compute device
        config: configuration dict
        affordance_emb_tensor: affordance embedding tensor

    Returns:
        dict with overall and per-affordance metrics
    """
    model.eval()
    val_dataset = val_loader.dataset

    all_preds = []
    all_targets = []
    all_aff_indices = []

    print("\n" + "=" * 60)
    print("EVALUATION: Running full evaluation...")
    print("=" * 60)

    with torch.no_grad():
        for i, (img, point, label, img_path, point_path, sub_box, obj_box) in enumerate(val_loader):
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)

            batch_size = img_dev.size(0)
            text_emb = torch.zeros(batch_size, config['text_dim']).to(device)

            # Get affordance index from img_path
            batch_aff_indices = []
            for b_idx, path in enumerate(img_path):
                aff_name = path.split('_')[-2]
                if aff_name in AFFORDANCE_LABELS:
                    aff_idx = AFFORDANCE_LABELS.index(aff_name)
                    text_emb[b_idx] = affordance_emb_tensor[aff_idx]
                    batch_aff_indices.append(aff_idx)
                else:
                    batch_aff_indices.append(-1)
            all_aff_indices.extend(batch_aff_indices)

            _3d, logits, _ = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)

            all_preds.append(_3d.cpu())
            all_targets.append(label.cpu())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Overall metrics
    results = {}
    sim_values = np.zeros(all_preds.shape[0])
    mae_values = np.zeros(all_preds.shape[0])
    auc_values = np.zeros(all_preds.shape[0])
    iou_values = np.zeros(all_preds.shape[0])
    iou_thres = np.linspace(0, 1, 20)
    targets_binary = (all_targets >= 0.5).astype(int)

    for i in range(all_preds.shape[0]):
        sim_values[i] = SIM(all_preds[i], all_targets[i])
        mae_values[i] = np.sum(np.abs(all_preds[i] - all_targets[i])) / all_preds.shape[1]

        t_true = targets_binary[i]
        p_score = all_preds[i]
        if np.sum(t_true) == 0:
            auc_values[i] = np.nan
            iou_values[i] = np.nan
        else:
            try:
                auc_values[i] = roc_auc_score(t_true.flatten(), p_score.flatten())
            except ValueError:
                auc_values[i] = np.nan
            temp_iou = []
            for thre in iou_thres:
                p_mask = (p_score >= thre).astype(int)
                intersect = np.sum(p_mask & t_true)
                union = np.sum(p_mask | t_true)
                temp_iou.append(1. * intersect / union if union > 0 else 0)
            iou_values[i] = np.mean(temp_iou)

    results['overall'] = {
        'AUC': np.nanmean(auc_values),
        'IOU': np.nanmean(iou_values),
        'SIM': np.mean(sim_values),
        'MAE': np.mean(mae_values),
    }

    # Per-affordance metrics
    per_affordance = {}
    for aff_idx, aff_name in enumerate(AFFORDANCE_LABELS):
        indices = [i for i, idx in enumerate(all_aff_indices) if idx == aff_idx]
        if len(indices) == 0:
            continue

        aff_auc = np.nanmean(auc_values[indices]) if len(indices) > 0 else float('nan')
        aff_iou = np.nanmean(iou_values[indices]) if len(indices) > 0 else float('nan')
        aff_sim = np.mean(sim_values[indices]) if len(indices) > 0 else float('nan')
        aff_mae = np.mean(mae_values[indices]) if len(indices) > 0 else float('nan')

        per_affordance[aff_name] = {
            'AUC': aff_auc,
            'IOU': aff_iou,
            'SIM': aff_sim,
            'MAE': aff_mae,
            'count': len(indices),
        }

    results['per_affordance'] = per_affordance

    return results


def print_evaluation_results(results):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    overall = results['overall']
    print(f"\n{'Overall Metrics':^70}")
    print("-" * 70)
    print(f"  AUC:  {overall['AUC']:.4f}")
    print(f"  IOU:  {overall['IOU']:.4f}")
    print(f"  SIM:  {overall['SIM']:.4f}")
    print(f"  MAE:  {overall['MAE']:.4f}")

    if 'per_affordance' in results and results['per_affordance']:
        print(f"\n{'Per-Affordance Metrics':^70}")
        print("-" * 70)
        print(f"{'Affordance':<15} {'AUC':>8} {'IOU':>8} {'SIM':>8} {'MAE':>8} {'Count':>6}")
        print("-" * 70)

        for aff_name, metrics in results['per_affordance'].items():
            print(f"{aff_name:<15} {metrics['AUC']:>8.4f} {metrics['IOU']:>8.4f} "
                  f"{metrics['SIM']:>8.4f} {metrics['MAE']:>8.4f} {metrics['count']:>6}")

    print("=" * 70)


def save_training_curves(history, model_name):
    """Save training curves as PNG image."""
    ensure_dir(CKPT_DIR)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if len(history['train_loss']) > 0:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        if len(history['val_loss']) > 0:
            axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curve (IAG_TextEmb)')
        axes[0, 0].legend(loc='best')
        axes[0, 0].grid(True)

    if len(history['val_auc']) > 0:
        axes[0, 1].plot(history['val_auc'], label='Val AUC', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('AUC Curve (IAG_TextEmb)')
        axes[0, 1].legend(loc='best')
        axes[0, 1].grid(True)

    if len(history['val_iou']) > 0:
        axes[1, 0].plot(history['val_iou'], label='Val IOU', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IOU')
        axes[1, 0].set_title('IOU Curve (IAG_TextEmb)')
        axes[1, 0].legend(loc='best')
        axes[1, 0].grid(True)

    if len(history['val_sim']) > 0:
        axes[1, 1].plot(history['val_sim'], label='Val SIM', color='purple')
        if len(history['val_mae']) > 0:
            ax2 = axes[1, 1].twinx()
            ax2.plot(history['val_mae'], label='Val MAE', color='brown', linestyle='--')
            ax2.set_ylabel('MAE', color='brown')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SIM')
        axes[1, 1].set_title('SIM and MAE Curves (IAG_TextEmb)')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True)

    plt.tight_layout()
    curve_path = os.path.join(CKPT_DIR, f"{model_name}-loss.png")
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Training] Curves saved: {curve_path}")


# ============================================================================
# Main Training Function
# ============================================================================

def train(args):
    """
    Main training function for IAG_TextEmb model.

    Steps:
    1. Download and load GloVe word embeddings
    2. Build affordance label embedding matrix
    3. Load PIAD dataset (Seen setting)
    4. Initialize IAG_TextEmb model
    5. Train with HM_Loss + CrossEntropy + KL divergence
    6. Validate with AUC, IOU, SIM, MAE metrics
    7. Save checkpoints and training curves
    """
    # ------------------------------------------------------------------
    # 1. Setup configuration
    # ------------------------------------------------------------------
    config = DEFAULT_CONFIG.copy()
    config['Setting'] = args.setting
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['Epoch'] = args.epochs
    config['text_dim'] = args.text_dim

    print("\n" + "=" * 60)
    print("IAG_TextEmb Training")
    print("=" * 60)
    print(f"  Setting:      {config['Setting']}")
    print(f"  Epochs:       {config['Epoch']}")
    print(f"  Batch size:   {config['batch_size']}")
    print(f"  Learning rate:{config['lr']}")
    print(f"  Text dim:     {config['text_dim']} (GloVe)")
    print(f"  Emb dim:      {config['emb_dim']}")
    print(f"  Data dir:     {args.data_dir}")
    print("=" * 60)

    device = get_device(args.use_gpu)

    # ------------------------------------------------------------------
    # 2. Download and load GloVe embeddings
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading pre-trained word embeddings...")
    glove_path = download_glove(urls=[args.glove_url] if args.glove_url else None)
    glove_embeddings = load_glove_embeddings(glove_path, target_words=AFFORDANCE_LABELS)

    # Also add component words for compound labels
    for label in AFFORDANCE_LABELS:
        sub_words = _split_compound_word(label)
        for w in sub_words:
            if w not in glove_embeddings:
                glove_embeddings.update(
                    load_glove_embeddings(glove_path, target_words=[w])
                )

    emb_matrix, missing = build_affordance_embeddings(glove_embeddings, AFFORDANCE_LABELS)
    if missing:
        print(f"[Warning] Missing GloVe vectors for: {missing}")

    # Convert to tensor on device
    affordance_emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32).to(device)
    print(f"[GloVe] Affordance embedding matrix shape: {affordance_emb_tensor.shape}")

    # Print some embedding statistics
    for i, label in enumerate(AFFORDANCE_LABELS):
        norm = np.linalg.norm(emb_matrix[i])
        print(f"  {label:<15} norm={norm:.4f}")

    # ------------------------------------------------------------------
    # 3. Load PIAD dataset
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading PIAD dataset...")
    data_path = os.path.join(args.data_dir, config['Setting'])

    if not os.path.exists(data_path):
        print(f"[Error] Data directory not found: {data_path}")
        print("[Info]  Please ensure the PIAD dataset is placed at:")
        print(f"       {data_path}/")
        print("       Required files:")
        print("         Point_Train.txt, Img_Train.txt, Box_Train.txt")
        print("         Point_Test.txt,  Img_Test.txt,  Box_Test.txt")
        sys.exit(1)

    required_files = ['Point_Train.txt', 'Img_Train.txt', 'Box_Train.txt',
                      'Point_Test.txt', 'Img_Test.txt', 'Box_Test.txt']
    for f in required_files:
        fpath = os.path.join(data_path, f)
        if not os.path.exists(fpath):
            print(f"[Error] Required data file not found: {fpath}")
            sys.exit(1)

    train_dataset = PIAD(
        'train', config['Setting'],
        os.path.join(data_path, 'Point_Train.txt'),
        os.path.join(data_path, 'Img_Train.txt'),
        os.path.join(data_path, 'Box_Train.txt'),
        config['pairing_num']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              num_workers=4, shuffle=True, drop_last=True)
    print(f"  Training samples: {len(train_dataset)}")

    val_dataset = PIAD(
        'val', config['Setting'],
        os.path.join(data_path, 'Point_Test.txt'),
        os.path.join(data_path, 'Img_Test.txt'),
        os.path.join(data_path, 'Box_Test.txt')
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            num_workers=4, shuffle=True)
    print(f"  Validation samples: {len(val_dataset)}")

    # ------------------------------------------------------------------
    # 4. Initialize model
    # ------------------------------------------------------------------
    print("\n[Step 3] Initializing IAG_TextEmb model...")
    model = get_IAG_TextEmb(
        pre_train=False,
        N_p=config['N_p'],
        emb_dim=config['emb_dim'],
        proj_dim=config['proj_dim'],
        num_heads=config['num_heads'],
        N_raw=config['N_raw'],
        num_affordance=config['num_affordance'],
        text_dim=config['text_dim']
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss functions
    criterion_hm = HM_Loss().to(device)
    criterion_ce = nn.CrossEntropyLoss().to(device)

    # Optimizer and scheduler (same as backend.py)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['Epoch'], eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # 5. Resume from checkpoint if specified
    # ------------------------------------------------------------------
    start_epoch = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_auc': [], 'val_iou': [],
        'val_sim': [], 'val_mae': []
    }

    if args.resume:
        print(f"\n[Resume] Loading checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint.get('epoch', 0) + 1
        history = checkpoint.get('history', history)
        config_loaded = checkpoint.get('config', {})
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Previous config: {config_loaded}")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    model_name = args.model_name or f"IAG_TextEmb-{config['Setting']}"
    ensure_dir(CKPT_DIR)
    ensure_dir(LOG_DIR)

    # Log file
    from datetime import datetime
    log_name = f"{datetime.now().year}-{datetime.now().month}-{datetime.now().day}-" \
               f"{datetime.now().hour}-{datetime.now().minute}-IAG_TextEmb-{config['Setting']}"
    log_file_path = os.path.join(LOG_DIR, f"{log_name}.txt")

    def log_print(msg):
        print(msg)
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

    log_print(f"Training started: IAG_TextEmb on {config['Setting']} dataset")
    log_print(f"Model: {model_name}")
    log_print(f"Config: {config}")
    log_print(f"Device: {device}")

    best_auc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, config['Epoch']):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        log_print(f"\n=== Epoch {epoch + 1}/{config['Epoch']} | LR: {lr:.6f} ===")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion_hm, criterion_ce,
            optimizer, device, config, affordance_emb_tensor
        )
        history['train_loss'].append(train_loss)
        log_print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, auc, iou, sim, mae = validate(
            model, val_loader, criterion_hm, criterion_ce,
            device, config, affordance_emb_tensor
        )
        history['val_loss'].append(val_loss)
        history['val_auc'].append(auc)
        history['val_iou'].append(iou)
        history['val_sim'].append(sim)
        history['val_mae'].append(mae)

        log_print(f"Val Loss: {val_loss:.4f} | AUC: {auc:.4f} | "
                  f"IOU: {iou:.4f} | SIM: {sim:.4f} | MAE: {mae:.4f}")

        # Save best model
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            import time
            best_path = os.path.join(CKPT_DIR, f"{model_name}-best.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, history,
                            config, model_name, config['Setting'],
                            log_file_path, best_path)
            log_print(f"New best model! AUC={auc:.4f} saved to {best_path}")

        # Save breakpoint every 5 epochs
        if (epoch + 1) % 5 == 0 and (epoch + 1) < config['Epoch']:
            bp_path = os.path.join(CKPT_DIR, f"{model_name}-epoch{epoch + 1}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, history,
                            config, model_name, config['Setting'],
                            log_file_path, bp_path)
            log_print(f"Breakpoint saved: epoch {epoch + 1}")

        scheduler.step()

    # ------------------------------------------------------------------
    # 7. Save final model and curves
    # ------------------------------------------------------------------
    final_path = os.path.join(CKPT_DIR, f"{model_name}-final.pt")
    save_checkpoint(model, optimizer, scheduler, config['Epoch'] - 1, history,
                    config, model_name, config['Setting'],
                    log_file_path, final_path)
    log_print(f"\nTraining completed! Final model saved: {final_path}")
    log_print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")

    # Save training curves
    save_training_curves(history, model_name)

    # ------------------------------------------------------------------
    # 8. Final evaluation
    # ------------------------------------------------------------------
    log_print("\nRunning final evaluation...")
    eval_results = evaluate_model(model, val_loader, device, config,
                                  affordance_emb_tensor)
    print_evaluation_results(eval_results)

    # Write evaluation results to log
    log_print("\nFinal Evaluation Results:")
    overall = eval_results['overall']
    log_print(f"  AUC: {overall['AUC']:.4f}")
    log_print(f"  IOU: {overall['IOU']:.4f}")
    log_print(f"  SIM: {overall['SIM']:.4f}")
    log_print(f"  MAE: {overall['MAE']:.4f}")

    return model, history, eval_results


# ============================================================================
# Evaluation-Only Mode
# ============================================================================

def evaluate_only(args):
    """
    Load a trained model and run comprehensive evaluation.
    """
    device = get_device(args.use_gpu)

    # Load GloVe
    glove_path = download_glove(urls=[args.glove_url] if args.glove_url else None)
    glove_embeddings = load_glove_embeddings(glove_path, target_words=AFFORDANCE_LABELS)
    for label in AFFORDANCE_LABELS:
        sub_words = _split_compound_word(label)
        for w in sub_words:
            if w not in glove_embeddings:
                glove_embeddings.update(
                    load_glove_embeddings(glove_path, target_words=[w])
                )

    emb_matrix, _ = build_affordance_embeddings(glove_embeddings, AFFORDANCE_LABELS)
    affordance_emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32).to(device)

    # Load checkpoint
    print(f"\n[Evaluation] Loading model from: {args.eval_only}")
    checkpoint = torch.load(args.eval_only, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', DEFAULT_CONFIG)
    config['text_dim'] = args.text_dim

    # Initialize model
    model = get_IAG_TextEmb(
        pre_train=False,
        N_p=config.get('N_p', 64),
        emb_dim=config.get('emb_dim', 512),
        proj_dim=config.get('proj_dim', 512),
        num_heads=config.get('num_heads', 4),
        N_raw=config.get('N_raw', 2048),
        num_affordance=config.get('num_affordance', 17),
        text_dim=config['text_dim']
    )
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    # Load dataset
    setting = config.get('Setting', args.setting)
    data_path = os.path.join(args.data_dir, setting)

    val_dataset = PIAD(
        'val', setting,
        os.path.join(data_path, 'Point_Test.txt'),
        os.path.join(data_path, 'Img_Test.txt'),
        os.path.join(data_path, 'Box_Test.txt')
    )
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 8),
                            num_workers=4, shuffle=False)

    # Run evaluation
    results = evaluate_model(model, val_loader, device, config, affordance_emb_tensor)
    print_evaluation_results(results)

    return results


# ============================================================================
# Argument Parser & Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="IAG_TextEmb Training Script - Train with GloVe text embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['Epoch'],
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                        help='Learning rate')
    parser.add_argument('--setting', type=str, default='Seen',
                        choices=['Seen', 'Unseen'],
                        help='Dataset setting (Seen or Unseen)')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Root data directory')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU usage')

    # Model parameters
    parser.add_argument('--text_dim', type=int, default=300,
                        help='GloVe embedding dimension (50/100/200/300)')
    parser.add_argument('--emb_dim', type=int, default=DEFAULT_CONFIG['emb_dim'],
                        help='Model embedding dimension')
    parser.add_argument('--num_heads', type=int, default=DEFAULT_CONFIG['num_heads'],
                        help='Number of attention heads')

    # Checkpoint / Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Custom model name (auto-generated if not set)')

    # Evaluation only
    parser.add_argument('--glove_url', type=str, default=None,
                        help='Optional custom URL for downloading GloVe zip file')
    parser.add_argument('--eval_only', type=str, default=None,
                        help='Path to trained model for evaluation-only mode')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.no_gpu:
        args.use_gpu = False

    if args.eval_only:
        evaluate_only(args)
    else:
        train(args)