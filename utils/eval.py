"""
Evaluation metrics for IAGNet
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def evaluating(pred, label):
    """Calculate MAE"""
    mae = torch.sum(torch.abs(pred - label), dim=(0, 1))
    points_num = pred.shape[0] * pred.shape[1]
    return mae, points_num


def KLD(map1, map2, eps=1e-12):
    """KL Divergence between two maps"""
    map1, map2 = map1 / (map1.sum() + eps), map2 / (map2.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def SIM(map1, map2, eps=1e-12):
    """Similarity metric"""
    map1, map2 = map1 / (map1.sum() + eps), map2 / (map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


def calculate_metrics(results, targets):
    """
    Calculate all evaluation metrics

    Args:
        results: predicted affordance scores [N, 2048, 1]
        targets: ground truth labels [N, 2048, 1]

    Returns:
        dict with AUC, IOU, SIM, MAE metrics
    """
    results = results.detach().cpu().numpy() if torch.is_tensor(results) else results
    targets = targets.detach().cpu().numpy() if torch.is_tensor(targets) else targets

    # SIM and MAE
    SIM_matrix = np.zeros(targets.shape[0])
    MAE_matrix = np.zeros(targets.shape[0])

    for i in range(targets.shape[0]):
        SIM_matrix[i] = SIM(results[i], targets[i])
        MAE_matrix[i] = np.sum(np.absolute(results[i] - targets[i])) / 2048

    sim = np.mean(SIM_matrix)
    mean_mae = np.mean(MAE_matrix)

    # AUC and IOU
    AUC = np.zeros((targets.shape[0], targets.shape[2]))
    IOU = np.zeros((targets.shape[0], targets.shape[2]))
    IOU_thres = np.linspace(0, 1, 20)
    targets_binary = targets >= 0.5
    targets_binary = targets_binary.astype(int)

    for i in range(AUC.shape[0]):
        t_true = targets_binary[i]
        p_score = results[i]

        if np.sum(t_true) == 0:
            AUC[i] = np.nan
            IOU[i] = np.nan
        else:
            auc = roc_auc_score(t_true.flatten(), p_score.flatten())
            AUC[i] = auc

            temp_iou = []
            for thre in IOU_thres:
                p_mask = (p_score >= thre).astype(int)
                intersect = np.sum(p_mask & t_true)
                union = np.sum(p_mask | t_true)
                if union > 0:
                    temp_iou.append(1. * intersect / union)
                else:
                    temp_iou.append(0)
            temp_iou = np.array(temp_iou)
            aiou = np.mean(temp_iou)
            IOU[i] = aiou

    auc = np.nanmean(AUC)
    iou = np.nanmean(IOU)

    return {
        'AUC': auc,
        'IOU': iou,
        'SIM': sim,
        'MAE': mean_mae,
        'SIM_matrix': SIM_matrix,
        'MAE_matrix': MAE_matrix,
        'AUC_matrix': AUC,
        'IOU_matrix': IOU
    }
