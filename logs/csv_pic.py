import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体（如果图表中需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_csv_data(file_path):
    """加载CSV文件数据"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def plot_all_losses(csv_files, save_path='all_losses_plot.png'):
    """
    将所有loss指标绘制在一个图像中
    
    参数:
        csv_files: 字典，包含指标名称和对应的CSV文件路径
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(12, 8))
    
    # 定义颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # 用于存储所有数据的字典
    all_data = {}
    
    # 加载并绘制每个loss指标
    for i, (metric_name, file_path) in enumerate(csv_files.items()):
        df = load_csv_data(file_path)
        if df is not None and not df.empty:
            # 使用Step列作为x轴（epoch）
            x = df['Step'].values
            y = df['Value'].values
            
            # 存储数据
            all_data[metric_name] = {'x': x, 'y': y}
            
            # 绘制曲线
            plt.plot(x, y, 
                    label=metric_name.replace('_', ' ').title(),
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=2,
                    alpha=0.8)
    
    # 设置图表属性
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Loss Curves (All Loss Components)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(1, 40)  # 设置x轴范围
    
    # 添加网格
    plt.minorticks_on()
    plt.grid(True, which='major', alpha=0.5)
    plt.grid(True, which='minor', alpha=0.2, linestyle=':')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"所有loss指标图表已保存至: {save_path}")
    
    return all_data

def plot_action_metrics(acc_file, loss_file, save_path='action_metrics_plot.png'):
    """
    将action_cls_acc和action_cls_loss绘制在一个图像中（使用双y轴）
    
    参数:
        acc_file: action_cls_acc的CSV文件路径
        loss_file: action_cls_loss的CSV文件路径
        save_path: 保存图像的路径
    """
    # 加载数据
    df_acc = load_csv_data(acc_file)
    df_loss = load_csv_data(loss_file)
    
    if df_acc is None or df_loss is None:
        print("无法加载action指标数据")
        return
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 设置x轴（epoch）
    x = df_acc['Step'].values
    
    # 绘制action_cls_acc（左y轴）
    color_acc = '#1f77b4'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', color=color_acc, fontsize=12)
    line_acc = ax1.plot(x, df_acc['Value'].values, 
                       label='Action Cls Accuracy',
                       color=color_acc,
                       linewidth=3,
                       alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0.8, 1.0)  # 设置准确率的y轴范围
    
    # 创建第二个y轴用于action_cls_loss
    ax2 = ax1.twinx()
    color_loss = '#d62728'
    ax2.set_ylabel('Loss', color=color_loss, fontsize=12)
    line_loss = ax2.plot(x, df_loss['Value'].values, 
                         label='Action Cls Loss',
                         color=color_loss,
                         linewidth=2,
                         linestyle='--',
                         alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_loss)
    ax2.set_ylim(0.6, 1.0)  # 设置损失的y轴范围
    
    # 设置标题和图例
    plt.title('Action Classification: Accuracy vs Loss', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines = line_acc + line_loss
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    # 添加网格
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(1, 40)  # 设置x轴范围
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Action指标图表已保存至: {save_path}")
    
    # 返回统计数据
    stats = {
        'final_acc': df_acc['Value'].iloc[-1],
        'final_loss': df_loss['Value'].iloc[-1],
        'max_acc': df_acc['Value'].max(),
        'min_loss': df_loss['Value'].min()
    }
    
    return stats

def create_detailed_loss_comparison(loss_files_dict, save_path='detailed_loss_comparison.png'):
    """
    创建更详细的loss对比图，将所有loss分成多个子图显示
    """
    # 过滤出所有的loss文件
    loss_files = {k: v for k, v in loss_files_dict.items() if 'loss' in k.lower()}
    
    n_losses = len(loss_files)
    n_cols = 2
    n_rows = (n_losses + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    # 如果只有一行，确保axes是二维数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(loss_files)))
    
    for idx, (metric_name, file_path) in enumerate(loss_files.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        df = load_csv_data(file_path)
        if df is not None and not df.empty:
            ax = axes[row, col]
            
            x = df['Step'].values
            y = df['Value'].values
            
            # 绘制曲线
            ax.plot(x, y, color=colors[idx], linewidth=2, alpha=0.8)
            
            # 添加最后值标签
            final_value = y[-1]
            ax.text(0.95, 0.95, f'Final: {final_value:.4f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 设置子图属性
            ax.set_title(metric_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Value')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(1, 40)
            
            # 如果是最小值在右上角的情况，添加最小值标记
            min_val = y.min()
            min_idx = np.argmin(y)
            if min_idx > len(x) * 0.7:  # 如果最小值出现在训练后期
                ax.plot(x[min_idx], min_val, 'r*', markersize=10)
                ax.annotate(f'Min: {min_val:.4f}', 
                          xy=(x[min_idx], min_val),
                          xytext=(5, 5),
                          textcoords='offset points',
                          fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(len(loss_files), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Detailed Loss Components Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"详细loss对比图已保存至: {save_path}")

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 假设CSV文件都在当前目录下
    # 定义CSV文件路径（根据您的实际文件路径修改）
    csv_files = {
        'action_cls_acc': 'action_cls_acc.csv',
        'total_loss': 'total_loss.csv',
        'sub_box_loss': 'sub_box_loss.csv',
        'size_loss': 'size_loss.csv',
        'obj_box_loss': 'obj_box_loss.csv',
        'mean_iou': 'mean_iou.csv',
        'giou_loss': 'giou_loss.csv',
        'action_cls_loss': 'action_cls_loss.csv'
    }
    
    # 检查文件是否存在
    missing_files = []
    for metric, file_path in csv_files.items():
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"警告: 以下文件不存在: {missing_files}")
        print("请确保所有CSV文件都在当前目录下，或修改文件路径。")
    else:
        print("所有CSV文件都存在，开始绘制图表...")
        
        # 1. 绘制所有loss指标在一个图中
        loss_files = {k: v for k, v in csv_files.items() if 'loss' in k.lower()}
        all_loss_data = plot_all_losses(loss_files, save_path='all_losses.png')
        
        # 2. 绘制action指标（准确率和损失）在一个图中
        action_stats = plot_action_metrics(
            csv_files['action_cls_acc'], 
            csv_files['action_cls_loss'], 
            save_path='action_metrics.png'
        )
        
        # 打印action指标的统计数据
        print("\n" + "="*50)
        print("Action指标统计数据:")
        print(f"最终准确率: {action_stats['final_acc']:.4f}")
        print(f"最终损失: {action_stats['final_loss']:.4f}")
        print(f"最高准确率: {action_stats['max_acc']:.4f}")
        print(f"最低损失: {action_stats['min_loss']:.4f}")
        print("="*50)
        
        # 3. 创建详细的loss对比图（每个loss一个子图）
        create_detailed_loss_comparison(csv_files, save_path='detailed_loss_comparison.png')
        
        # 4. 可选：绘制训练过程总结图（包含准确率和主要损失）
        plt.figure(figsize=(14, 10))
        
        # 子图1: 总损失
        plt.subplot(2, 2, 1)
        if 'total_loss' in csv_files:
            df_total_loss = load_csv_data(csv_files['total_loss'])
            if df_total_loss is not None:
                plt.plot(df_total_loss['Step'], df_total_loss['Value'], 'b-', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Total Loss')
                plt.title('Total Loss Curve')
                plt.grid(True, alpha=0.3)
                plt.xlim(1, 40)
        
        # 子图2: Action准确率
        plt.subplot(2, 2, 2)
        if 'action_cls_acc' in csv_files:
            df_acc = load_csv_data(csv_files['action_cls_acc'])
            if df_acc is not None:
                plt.plot(df_acc['Step'], df_acc['Value'], 'g-', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Action Classification Accuracy')
                plt.grid(True, alpha=0.3)
                plt.xlim(1, 40)
                plt.ylim(0.8, 1.0)
        
        # 子图3: Mean IoU
        plt.subplot(2, 2, 3)
        if 'mean_iou' in csv_files:
            df_iou = load_csv_data(csv_files['mean_iou'])
            if df_iou is not None:
                plt.plot(df_iou['Step'], df_iou['Value'], 'r-', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Mean IoU')
                plt.title('Mean IoU (Object Detection)')
                plt.grid(True, alpha=0.3)
                plt.xlim(1, 40)
        
        # 子图4: 主要损失分量对比
        plt.subplot(2, 2, 4)
        loss_components = ['giou_loss', 'action_cls_loss', 'obj_box_loss']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, loss_name in enumerate(loss_components):
            if loss_name in csv_files:
                df_loss = load_csv_data(csv_files[loss_name])
                if df_loss is not None:
                    plt.plot(df_loss['Step'], df_loss['Value'], 
                            label=loss_name.replace('_', ' ').title(),
                            color=colors[i],
                            linewidth=1.5,
                            alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Main Loss Components')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, 40)
        
        plt.suptitle('Training Summary (40 Epochs)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('training_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n所有图表已生成完毕！")
        print("生成的文件:")
        print("1. all_losses.png - 所有loss指标曲线")
        print("2. action_metrics.png - Action准确率和损失对比")
        print("3. detailed_loss_comparison.png - 详细loss分量对比")
        print("4. training_summary.png - 训练过程总结")