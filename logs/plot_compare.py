import re
import matplotlib.pyplot as plt
import numpy as np

# 解析日志文件的函数
def parse_log_file(filename):
    epochs = []
    train_losses = []
    val_losses = []
    aucs = []
    ious = []
    sims = []
    maes = []
    
    with open(filename, 'r') as file:
        content = file.read()
        
    # 查找epoch信息
    epoch_pattern = r"=== Epoch (\d+)/\d+"
    epochs_found = re.findall(epoch_pattern, content)
    
    # 查找训练损失
    train_loss_pattern = r"Train Loss: ([\d.]+)"
    train_losses_found = re.findall(train_loss_pattern, content)
    
    # 查找验证损失和其他指标
    val_metrics_pattern = r"Val Loss: ([\d.]+) \| AUC: ([\d.]+) \| IOU: ([\d.]+) \| SIM: ([\d.]+) \| MAE: ([\d.]+)"
    val_metrics_found = re.findall(val_metrics_pattern, content)
    
    # 提取数据
    num_epochs = min(len(epochs_found), len(train_losses_found), len(val_metrics_found))
    
    for i in range(num_epochs):
        epochs.append(int(epochs_found[i]))
        train_losses.append(float(train_losses_found[i]))
        
        val_metrics = val_metrics_found[i]
        val_losses.append(float(val_metrics[0]))
        aucs.append(float(val_metrics[1]))
        ious.append(float(val_metrics[2]))
        sims.append(float(val_metrics[3]))
        maes.append(float(val_metrics[4]))
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'aucs': aucs,
        'ious': ious,
        'sims': sims,
        'maes': maes
    }

# 解析所有日志文件
try:
    seen_log1 = parse_log_file("2026-3-17-2-39-Seen-log.txt")
except Exception as e:
    print(f"解析 2026-3-17-2-39-Seen-log.txt 时出错: {e}")
    seen_log1 = None

try:
    unseen_log1 = parse_log_file("2026-3-17-2-39-Unseen-log.txt")
except Exception as e:
    print(f"解析 2026-3-17-2-39-Unseen-log.txt 时出错: {e}")
    unseen_log1 = None

try:
    seen_log2 = parse_log_file("2026-5-16-1-31-IAG_TextEmb-Seen.txt")
except Exception as e:
    print(f"解析 2026-5-16-1-31-IAG_TextEmb-Seen.txt 时出错: {e}")
    seen_log2 = None

try:
    unseen_log2 = parse_log_file("2026-5-12-20-32-IAG_TextEmb-Unseen.txt")
except Exception as e:
    print(f"解析 2026-5-12-20-32-IAG_TextEmb-Unseen.txt 时出错: {e}")
    unseen_log2 = None

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 创建Seen数据图表
if seen_log1 or seen_log2:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Seen', fontsize=16)
    
    # 绘制训练损失
    ax = axes[0, 0]
    if seen_log1 and seen_log1['epochs']:
        ax.plot(seen_log1['epochs'], seen_log1['train_losses'], label='IAGNet-Seen', marker='o', color='tab:blue')
    if seen_log2 and seen_log2['epochs']:
        ax.plot(seen_log2['epochs'], seen_log2['train_losses'], label='IAG-Textemb-Seen', marker='s', color='tab:orange')
    ax.set_title('Train-loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # 绘制验证损失
    ax = axes[0, 1]
    if seen_log1 and seen_log1['epochs']:
        ax.plot(seen_log1['epochs'], seen_log1['val_losses'], label='IAGNet-Seen', marker='o', color='tab:blue')
    if seen_log2 and seen_log2['epochs']:
        ax.plot(seen_log2['epochs'], seen_log2['val_losses'], label='IAG-Textemb-Seen', marker='s', color='tab:orange')
    ax.set_title('Val-loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # 绘制AUC
    ax = axes[0, 2]
    if seen_log1 and seen_log1['epochs']:
        ax.plot(seen_log1['epochs'], seen_log1['aucs'], label='IAGNet-Seen', marker='o', color='tab:blue')
    if seen_log2 and seen_log2['epochs']:
        ax.plot(seen_log2['epochs'], seen_log2['aucs'], label='IAG-Textemb-Seen', marker='s', color='tab:orange')
    ax.set_title('AUC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.legend()
    ax.grid(True)
    
    # 绘制IOU
    ax = axes[1, 0]
    if seen_log1 and seen_log1['epochs']:
        ax.plot(seen_log1['epochs'], seen_log1['ious'], label='IAGNet-Seen', marker='o', color='tab:blue')
    if seen_log2 and seen_log2['epochs']:
        ax.plot(seen_log2['epochs'], seen_log2['ious'], label='IAG-Textemb-Seen', marker='s', color='tab:orange')
    ax.set_title('IOU')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IOU')
    ax.legend()
    ax.grid(True)
    
    # 绘制SIM
    ax = axes[1, 1]
    if seen_log1 and seen_log1['epochs']:
        ax.plot(seen_log1['epochs'], seen_log1['sims'], label='IAGNet-Seen', marker='o', color='tab:blue')
    if seen_log2 and seen_log2['epochs']:
        ax.plot(seen_log2['epochs'], seen_log2['sims'], label='IAG-Textemb-Seen', marker='s', color='tab:orange')
    ax.set_title('SIM')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SIM')
    ax.legend()
    ax.grid(True)
    
    # 绘制MAE
    ax = axes[1, 2]
    if seen_log1 and seen_log1['epochs']:
        ax.plot(seen_log1['epochs'], seen_log1['maes'], label='IAGNet-Seen', marker='o', color='tab:blue')
    if seen_log2 and seen_log2['epochs']:
        ax.plot(seen_log2['epochs'], seen_log2['maes'], label='IAG-Textemb-Seen', marker='s', color='tab:orange')
    ax.set_title('MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    seen_fig_name = 'seen_metrics_comparison_updated.png'
    plt.savefig(seen_fig_name, dpi=300, bbox_inches='tight')
    print(seen_fig_name)

# 创建Unseen数据图表
if unseen_log1 or unseen_log2:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Unseen', fontsize=16)
    
    # 绘制训练损失
    ax = axes[0, 0]
    if unseen_log1 and unseen_log1['epochs']:
        ax.plot(unseen_log1['epochs'], unseen_log1['train_losses'], label='IAGNet-Unseen', marker='o', color='tab:blue')
    if unseen_log2 and unseen_log2['epochs']:
        ax.plot(unseen_log2['epochs'], unseen_log2['train_losses'], label='IAG-Textemb-Unseen', marker='s', color='tab:orange')
    ax.set_title('Train-Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # 绘制验证损失
    ax = axes[0, 1]
    if unseen_log1 and unseen_log1['epochs']:
        ax.plot(unseen_log1['epochs'], unseen_log1['val_losses'], label='IAGNet-Unseen', marker='o', color='tab:blue')
    if unseen_log2 and unseen_log2['epochs']:
        ax.plot(unseen_log2['epochs'], unseen_log2['val_losses'], label='IAG-Textemb-Unseen', marker='s', color='tab:orange')
    ax.set_title('Val-Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # 绘制AUC
    ax = axes[0, 2]
    if unseen_log1 and unseen_log1['epochs']:
        ax.plot(unseen_log1['epochs'], unseen_log1['aucs'], label='IAGNet-Unseen', marker='o', color='tab:blue')
    if unseen_log2 and unseen_log2['epochs']:
        ax.plot(unseen_log2['epochs'], unseen_log2['aucs'], label='IAG-Textemb-Unseen', marker='s', color='tab:orange')
    ax.set_title('AUC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.legend()
    ax.grid(True)
    
    # 绘制IOU
    ax = axes[1, 0]
    if unseen_log1 and unseen_log1['epochs']:
        ax.plot(unseen_log1['epochs'], unseen_log1['ious'], label='IAGNet-Unseen', marker='o', color='tab:blue')
    if unseen_log2 and unseen_log2['epochs']:
        ax.plot(unseen_log2['epochs'], unseen_log2['ious'], label='IAG-Textemb-Unseen', marker='s', color='tab:orange')
    ax.set_title('IOU')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IOU')
    ax.legend()
    ax.grid(True)
    
    # 绘制SIM
    ax = axes[1, 1]
    if unseen_log1 and unseen_log1['epochs']:
        ax.plot(unseen_log1['epochs'], unseen_log1['sims'], label='IAGNet-Unseen', marker='o', color='tab:blue')
    if unseen_log2 and unseen_log2['epochs']:
        ax.plot(unseen_log2['epochs'], unseen_log2['sims'], label='IAG-Textemb-Unseen', marker='s', color='tab:orange')
    ax.set_title('SIM')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SIM')
    ax.legend()
    ax.grid(True)
    
    # 绘制MAE
    ax = axes[1, 2]
    if unseen_log1 and unseen_log1['epochs']:
        ax.plot(unseen_log1['epochs'], unseen_log1['maes'], label='IAGNet-Unseen', marker='o', color='tab:blue')
    if unseen_log2 and unseen_log2['epochs']:
        ax.plot(unseen_log2['epochs'], unseen_log2['maes'], label='IAG-Textemb-Unseen', marker='s', color='tab:orange')
    ax.set_title('MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    unseen_fig_name = 'unseen_metrics_comparison_updated.png'
    plt.savefig(unseen_fig_name, dpi=300, bbox_inches='tight')
    print(unseen_fig_name)