import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(file_path):
    """解析单个日志文件，提取训练指标"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 提取epoch信息
    epochs = []
    train_losses = []
    val_losses = []
    auc_scores = []
    iou_scores = []
    sim_scores = []
    mae_scores = []
    lrs = []
    
    # 正则表达式匹配epoch行
    epoch_pattern = r'=== Epoch (\d+)/\d+ \| LR: ([\d\.]+) ==='
    train_pattern = r'Train Loss: ([\d\.]+)'
    val_pattern = r'Val Loss: ([\d\.]+) \| AUC: ([\d\.]+) \| IOU: ([\d\.]+) \| SIM: ([\d\.]+) \| MAE: ([\d\.]+)'
    
    # 分割内容为行
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # 匹配epoch开始
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            lr = float(epoch_match.group(2))
            epochs.append(epoch)
            lrs.append(lr)
            
            # 在后续行中查找Train Loss和Val Loss
            for j in range(i+1, min(i+10, len(lines))):  # 在后续10行内查找
                train_line = lines[j]
                val_line = lines[j+1] if j+1 < len(lines) else ""
                
                train_match = re.search(train_pattern, train_line)
                if train_match and 'Train Loss' in train_line:
                    train_losses.append(float(train_match.group(1)))
                    
                    # 在下一行查找验证指标
                    val_match = re.search(val_pattern, val_line)
                    if val_match and 'Val Loss' in val_line:
                        val_losses.append(float(val_match.group(1)))
                        auc_scores.append(float(val_match.group(2)))
                        iou_scores.append(float(val_match.group(3)))
                        sim_scores.append(float(val_match.group(4)))
                        mae_scores.append(float(val_match.group(5)))
                    else:
                        # 如果没有找到验证指标，用NaN填充
                        val_losses.append(np.nan)
                        auc_scores.append(np.nan)
                        iou_scores.append(np.nan)
                        sim_scores.append(np.nan)
                        mae_scores.append(np.nan)
                    break
    
    # 创建DataFrame
    data = {
        'Epoch': epochs,
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'AUC': auc_scores,
        'IOU': iou_scores,
        'SIM': sim_scores,
        'MAE': mae_scores,
        'LR': lrs
    }
    
    df = pd.DataFrame(data)
    return df

def plot_comparison(files_dict, dataset_type, output_dir='./plots'):
    """绘制相同数据集的对比图"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_type} Dataset - Training Curves Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('Train_Loss', 'Train Loss'),
        ('Val_Loss', 'Validation Loss'),
        ('AUC', 'AUC Score'),
        ('IOU', 'IOU Score'),
        ('SIM', 'SIM Score'),
        ('MAE', 'MAE')
    ]
    
    colors = plt.cm.tab10.colors
    
    for idx, (metric_col, metric_name) in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        
        for i, (exp_name, df) in enumerate(files_dict.items()):
            if metric_col in df.columns and len(df) > 0:
                epochs = df['Epoch']
                values = df[metric_col]
                
                # 过滤掉NaN值
                valid_mask = ~np.isnan(values)
                if np.any(valid_mask):
                    ax.plot(epochs[valid_mask], values[valid_mask], 
                           label=exp_name, color=colors[i % len(colors)], 
                           linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # 为AUC、IOU、SIM设置合适的y轴范围
        if metric_col in ['AUC', 'IOU', 'SIM']:
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_type}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独绘制学习率曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (exp_name, df) in enumerate(files_dict.items()):
        if 'LR' in df.columns and len(df) > 0:
            epochs = df['Epoch']
            lrs = df['LR']
            valid_mask = ~np.isnan(lrs)
            if np.any(valid_mask):
                ax.plot(epochs[valid_mask], lrs[valid_mask], 
                       label=exp_name, color=colors[i % len(colors)], 
                       linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(f'{dataset_type} Dataset - Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # 对数尺度显示学习率变化
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_type}_learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置日志文件路径
    log_dir = './logs'
    
    # 定义要分析的文件模式
    unseen_files = [
        f'{log_dir}/2026-5-12-20-32-IAG_TextEmb-Unseen.txt',
        f'{log_dir}/2026-3-17-2-39-Unseen-log.txt'
    ]
    
    seen_files = [
        f'{log_dir}/2026-5-16-1-31-IAG_TextEmb-Seen.txt',
        f'{log_dir}/2026-3-17-2-39-Seen-log.txt'
    ]
    
    # 解析文件
    unseen_data = {}
    seen_data = {}
    
    print("正在解析Unseen数据集文件...")
    for file_path in unseen_files:
        try:
            df = parse_log_file(file_path)
            exp_name = Path(file_path).stem
            unseen_data[exp_name] = df
            print(f"  已解析: {exp_name}, 找到 {len(df)} 个epoch")
        except Exception as e:
            print(f"  解析 {file_path} 时出错: {e}")
    
    print("\n正在解析Seen数据集文件...")
    for file_path in seen_files:
        try:
            df = parse_log_file(file_path)
            exp_name = Path(file_path).stem
            seen_data[exp_name] = df
            print(f"  已解析: {exp_name}, 找到 {len(df)} 个epoch")
        except Exception as e:
            print(f"  解析 {file_path} 时出错: {e}")
    
    # 绘制对比图
    print("\n正在生成对比图...")
    if unseen_data:
        plot_comparison(unseen_data, 'Unseen')
        print("  Unseen数据集对比图已保存到 ./plots/Unseen_comparison.png")
        print("  学习率曲线已保存到 ./plots/Unseen_learning_rate.png")
    
    if seen_data:
        plot_comparison(seen_data, 'Seen')
        print("  Seen数据集对比图已保存到 ./plots/Seen_comparison.png")
        print("  学习率曲线已保存到 ./plots/Seen_learning_rate.png")
    
    # 打印最终评估结果摘要
    print("\n" + "="*60)
    print("最终评估结果摘要:")
    print("="*60)
    
    for dataset_name, data_dict in [("Unseen", unseen_data), ("Seen", seen_data)]:
        if data_dict:
            print(f"\n{dataset_name}数据集:")
            for exp_name, df in data_dict.items():
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    print(f"  {exp_name}:")
                    print(f"    最终epoch: {int(last_row['Epoch'])}, Train Loss: {last_row['Train_Loss']:.4f}, Val Loss: {last_row['Val_Loss']:.4f}")
                    if not pd.isna(last_row.get('AUC')):
                        print(f"    AUC: {last_row['AUC']:.4f}, IOU: {last_row['IOU']:.4f}, SIM: {last_row['SIM']:.4f}, MAE: {last_row['MAE']:.4f}")

if __name__ == "__main__":
    main()