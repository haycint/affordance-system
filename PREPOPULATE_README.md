# 点云记忆库预填充脚本使用说明

## 概述

`prepopulate_point_memory.py` 用于从 PIAD 数据集预填充点云记忆库。该脚本会：

1. 加载预训练的 IAG_TextEmb 模型
2. 从 PIAD 数据集（Seen + Unseen）加载所有样本
3. 提取 ARM 特征和点云特征
4. 根据 Ground Truth 标签生成偏好矩阵
5. 将记忆存储到 MemoryManager

## 依赖项

确保已安装以下依赖：

```bash
pip install torch torchvision numpy scipy tqdm
# 可选：安装 FAISS 以加速索引
pip install faiss-cpu  # 或 faiss-gpu
```

## 使用方法

### 基本用法

```bash
python prepopulate_point_memory.py \
    --weights ./model_list/IAG_textemb_seen.pth \
    --data_root ./data/PIAD \
    --memory_dir ./point_memory
```

### 完整参数说明

```bash
python prepopulate_point_memory.py \
    --weights <模型权重路径> \
    --data_root <PIAD数据集根目录> \
    --memory_dir <记忆库存储目录> \
    --device cuda:0 \
    --batch_size 1 \
    --max_samples 1000 \
    --neighbor_k 30 \
    --max_memories 5000 \
    --no_faiss
```

### 参数详解

- `--weights`: **(必需)** 预训练模型权重文件路径（例如 `IAG_textemb_seen.pth`）
- `--data_root`: **(必需)** PIAD 数据集根目录，应包含 `Seen` 和 `Unseen` 子目录
- `--memory_dir`: 记忆库存储目录（默认：`./point_memory`）
- `--device`: 计算设备（默认：`cuda:0`，可选 `cpu`）
- `--batch_size`: 批次大小（默认：`1`，建议保持为 1）
- `--max_samples`: 最大处理样本数，用于调试（默认：`None`，处理全部）
- `--neighbor_k`: 每个正样本点的邻居数量（默认：`30`）
- `--max_memories`: 记忆库最大容量（默认：`5000`）
- `--no_faiss`: 禁用 FAISS 索引，使用 NumPy 暴力搜索

## 数据集结构

PIAD 数据集应具有以下结构：

```
data_root/
├── Seen/
│   ├── point/
│   ├── img/
│   └── box/
└── Unseen/
    ├── point/
    ├── img/
    └── box/
```

## 示例

### 1. 快速测试（处理 100 个样本）

```bash
python prepopulate_point_memory.py \
    --weights ./model_list/IAG_textemb_seen.pth \
    --data_root ./data/PIAD \
    --memory_dir ./point_memory_test \
    --max_samples 100 \
    --device cuda:0
```

### 2. 完整预填充（使用 CPU）

```bash
python prepopulate_point_memory.py \
    --weights ./model_list/IAG_textemb_seen.pth \
    --data_root ./data/PIAD \
    --memory_dir ./point_memory_full \
    --device cpu \
    --max_memories 10000
```

### 3. 不使用 FAISS（兼容模式）

```bash
python prepopulate_point_memory.py \
    --weights ./model_list/IAG_textemb_seen.pth \
    --data_root ./data/PIAD \
    --memory_dir ./point_memory \
    --no_faiss
```

## 输出

脚本运行后会在 `memory_dir` 目录下生成以下文件：

- `memories.pkl`: 记忆数据
- `index.faiss` 或 `index.npy`: 索引文件
- `metadata.json`: 元数据

## 故障排除

### 1. CUDA 内存不足

```bash
# 使用 CPU
python prepopulate_point_memory.py --device cpu ...
```

### 2. scipy 未安装

脚本会自动降级到简单模式（不使用 KDTree），但建议安装：

```bash
pip install scipy
```

### 3. 模型加载失败

确保模型权重文件路径正确，且模型参数与权重文件匹配。

### 4. 数据集路径错误

检查 `data_root` 是否包含 `Seen` 和 `Unseen` 子目录。

## 性能优化

1. **使用 GPU**: 设置 `--device cuda:0` 可显著加速特征提取
2. **安装 FAISS**: 使用 FAISS 可加速记忆检索
3. **调整 batch_size**: 如果内存充足，可尝试增加 batch_size（但需要修改数据集代码）

## 注意事项

1. 预填充过程可能需要较长时间，取决于数据集大小和硬件配置
2. 确保有足够的磁盘空间存储记忆库
3. 建议先用 `--max_samples` 进行小规模测试
4. 如果遇到错误，脚本会继续处理其他样本，最后会显示成功率

## 集成到训练流程

预填充完成后，可以在训练脚本中加载记忆库：

```python
from memory_system.memory_manager import MemoryManager

memory_manager = MemoryManager(
    emb_dim=512,
    index_dim=128,
    feat_dim=512,
    store_dir="./point_memory",
    max_memories=5000,
    use_faiss=True
)

# 记忆库会自动加载已存储的记忆
```
