"""
Visualization utilities for IAGNet
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Affordance labels and colors
AFFORDANCE_LABEL_LIST = ['grasp', 'contain', 'lift', 'open',
                         'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                         'push', 'listen', 'wear', 'press', 'cut', 'stab']

COLOR_LIST = [
    [252, 19, 19], [249, 113, 45], [247, 183, 55], [251, 251, 11], [178, 244, 44],
    [255, 0, 0], [0, 0, 255], [25, 248, 99], [46, 253, 184], [40, 253, 253],
    [27, 178, 253], [28, 100, 243], [46, 46, 125], [105, 33, 247], [172, 10, 253],
    [249, 47, 249], [253, 51, 186], [250, 18, 95]
]
COLOR_LIST = np.array(COLOR_LIST)


def get_affordance_label(str_path, label):
    """Get affordance label from image path"""
    cut_str = str_path.split('_')
    affordance = cut_str[-2]
    index = AFFORDANCE_LABEL_LIST.index(affordance)
    label = label[:, index]
    return label, index


def get_point_colors(affordance_scores, reference_color=None, back_color=None):
    """
    Generate colors for point cloud visualization based on affordance scores

    Args:
        affordance_scores: [N,] array of scores between 0 and 1
        reference_color: RGB color for high affordance
        back_color: RGB color for low affordance

    Returns:
        colors: [N, 3] array of RGB colors
    """
    if reference_color is None:
        reference_color = np.array([255, 0, 0])
    if back_color is None:
        back_color = np.array([190, 190, 190])

    affordance_scores = np.array(affordance_scores).flatten()
    colors = np.zeros((len(affordance_scores), 3))
    for i, score in enumerate(affordance_scores):
        colors[i] = (reference_color - back_color) * score + back_color

    return colors / 255.0


def visualize_point_cloud_matplotlib(points, affordance_pred, affordance_gt=None,
                                      title="Point Cloud Affordance", save_path=None):
    """
    Visualize point cloud with affordance prediction using matplotlib

    Args:
        points: [N, 3] array of point coordinates
        affordance_pred: [N,] array of predicted affordance scores
        affordance_gt: [N,] array of ground truth affordance scores (optional)
        title: plot title
        save_path: path to save the figure
    """
    fig = plt.figure(figsize=(12, 6))

    if affordance_gt is not None:
        # Show both prediction and ground truth
        ax1 = fig.add_subplot(121, projection='3d')
        colors_pred = get_point_colors(affordance_pred)
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors_pred, s=1)
        ax1.set_title('Prediction')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(122, projection='3d')
        colors_gt = get_point_colors(affordance_gt)
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors_gt, s=1)
        ax2.set_title('Ground Truth')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
    else:
        # Show only prediction
        ax = fig.add_subplot(111, projection='3d')
        colors_pred = get_point_colors(affordance_pred)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors_pred, s=1)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def plot_training_curves(history, save_dir, model_name):
    """
    Plot training curves (loss, metrics) and save to files

    Args:
        history: dict containing training history
        save_dir: directory to save plots
        model_name: name for the saved files
    """
    os.makedirs(save_dir, exist_ok=True)

    # Plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total Loss
    if 'train_loss' in history and len(history['train_loss']) > 0:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in history and len(history['val_loss']) > 0:
            axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    # AUC
    if 'val_auc' in history and len(history['val_auc']) > 0:
        axes[0, 1].plot(history['val_auc'], label='Val AUC', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('AUC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # IOU
    if 'val_iou' in history and len(history['val_iou']) > 0:
        axes[1, 0].plot(history['val_iou'], label='Val IOU', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IOU')
        axes[1, 0].set_title('IOU Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # SIM and MAE
    if 'val_sim' in history and len(history['val_sim']) > 0:
        axes[1, 1].plot(history['val_sim'], label='Val SIM', color='purple')
    if 'val_mae' in history and len(history['val_mae']) > 0:
        ax2 = axes[1, 1].twinx()
        ax2.plot(history['val_mae'], label='Val MAE', color='brown', linestyle='--')
        ax2.set_ylabel('MAE', color='brown')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('SIM')
    axes[1, 1].set_title('SIM and MAE Curves')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    loss_path = os.path.join(save_dir, f'{model_name}-loss.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()

    return loss_path


def create_affordance_visualization_image(points, pred_scores, gt_scores=None,
                                           img_path=None, affordance_name=None):
    """
    Create a visualization image for affordance grounding results

    Args:
        points: [N, 3] point coordinates
        pred_scores: [N,] predicted affordance scores
        gt_scores: [N,] ground truth scores (optional)
        img_path: original image path for title
        affordance_name: name of affordance type

    Returns:
        fig: matplotlib figure
    """
    if gt_scores is not None:
        fig = plt.figure(figsize=(16, 6))

        # Prediction
        ax1 = fig.add_subplot(131, projection='3d')
        colors_pred = get_point_colors(pred_scores)
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors_pred, s=2)
        ax1.set_title(f'Prediction - {affordance_name}')
        ax1.view_init(elev=30, azim=45)

        # Ground Truth
        ax2 = fig.add_subplot(132, projection='3d')
        colors_gt = get_point_colors(gt_scores)
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors_gt, s=2)
        ax2.set_title(f'Ground Truth - {affordance_name}')
        ax2.view_init(elev=30, azim=45)

        # Difference
        ax3 = fig.add_subplot(133, projection='3d')
        diff = np.abs(pred_scores.flatten() - gt_scores.flatten())
        diff_colors = plt.cm.RdYlGn_r(diff)
        ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=diff_colors, s=2)
        ax3.set_title('Absolute Difference')
        ax3.view_init(elev=30, azim=45)
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        colors_pred = get_point_colors(pred_scores)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors_pred, s=2)
        ax.set_title(f'Prediction - {affordance_name}')
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    return fig
