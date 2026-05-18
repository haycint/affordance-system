# 标注模型架构问题分析与改进文档

**项目路径**: `annotation/`  
**涉及文件**: `annotation_model.py` · `annotation_dataset.py` · `train_annotation.py` · `config_annotation.yaml`

---

## 一、原始架构概述

标注模型（Annotation Model）的任务是：给定一张 224×224 的 RGB 图像，同时输出：

- `subject_box`：主体（操作者/机器人）的边界框 `[x1, y1, x2, y2]`
- `object_box`：客体（被操作物）的边界框 `[x1, y1, x2, y2]`
- `action_embed`：动作的 300 维 GloVe 词向量
- `object_embed`：客体类别的 300 维 GloVe 词向量
- `action_logits`：动作分类 logit（17 类）
- `object_logits`：客体分类 logit（23 类）

原始设计提供了两种方案（Scheme）：

- **Scheme 1**：单一 ResNet 主干，所有头共享特征
- **Scheme 2**：三个独立子模型（BoxSubModel、ActionSubModel、ObjectSubModel），各有独立主干

---

## 二、原始设计中的问题分析

### 问题 1：边界框输出无激活函数，与训练目标尺度严重不匹配

**原始代码（`BoxRegressionHead.forward`）：**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.mlp(x)   # 原始线性输出，范围 (-∞, +∞)
```

**训练目标（`annotation_dataset.py`，原始 `resize_box`）：**

```python
# 返回的是像素坐标，范围 [0, 224]
return [box[0] * scale_w, box[1] * scale_h,
        box[2] * scale_w, box[3] * scale_h]
```

**问题分析：**

ResNet 全局平均池化后接 MLP，输出默认在 `[-1, 1]` 附近（参数初始化决定），而训练目标是绝对像素坐标 `[0, 224]`。两者尺度相差约 100 倍。

Smooth L1 Loss 在误差 > β（默认 β=1）时退化为 L1，梯度恒为 ±1，此时一次反传更新量为 `lr × 1 ≈ 1e-4`，而模型需要从 `[-1, 1]` 跨越到 `[0, 224]`，需要大量 epoch 才能使输出进入合理范围。在这段时间内，loss 数值极大，梯度方差也大，训练极不稳定。

推理时若使用未充分训练的模型，所有框值都接近 0，导致前端观察到的边界框总是压缩在图像左上角附近。

**根本原因**：输出头没有激活函数约束，坐标系未归一化。

---

### 问题 2：EmbeddingRegressionHead 中的信息瓶颈

**原始代码（`EmbeddingRegressionHead.forward`，`classify_first=True` 时）：**

```python
def forward(self, x):
    logits = self.cls_head(x)        # 512-dim feature → 17/23-dim logits
    embedding = self.map_head(logits) # 17/23-dim logits → 300-dim embedding
    return embedding, logits
```

**问题分析：**

词向量回归路径是一个严格的串联结构：

```
backbone feature (512-dim)
    ↓
cls_head (MLP: 512 → 512 → 17)
    ↓ 【信息瓶颈：512-dim 特征被压缩为 17-dim】
map_head (MLP: 17 → 512 → 300)
    ↓
embedding (300-dim)
```

将 512 维的丰富视觉特征压缩到 17 维分类 logit 之后，再试图从中恢复出 300 维的语义词向量，是一个信息严重受限的操作。

- `cls_head` 只保留了"哪一类"的信息，丢弃了视觉细节
- `map_head` 实际上只是学了一个从类别 ID 到词向量的查找表，而非真正的视觉-语义对齐
- 这两个损失（分类交叉熵 + embedding MSE）相互冲突：cls_head 希望特征尽量可分，而 embedding 回归需要更连续的特征空间

此外，`map_head` 的输入维度只有 17 或 23（类别数），MLP 的中间层却是 512 维，形成了"漏斗-膨胀"的反常结构，第一层线性变换几乎无意义。

---

### 问题 3：损失函数设计不合理

**原始损失权重：**

```python
w_action_cls = 0.5
w_object_cls = 0.5
w_action_embed = 1.0
w_object_embed = 1.0
```

**问题分析：**

分类任务（交叉熵）权重仅为 embedding 回归任务（MSE）的 0.5 倍。但实际推理时最终用的是 `action_logits.argmax()` 来判断类别，embedding 回归只是辅助。这种权重设置让主要任务反而受到更弱的监督。

**embedding 使用 MSE Loss 的问题：**

MSE loss 在欧氏空间中定义距离，而 GloVe 词向量通常通过余弦相似度来度量语义相似性，两者度量空间不一致。对于 `[0.1, 0.2, ...]` 和 `[-0.1, -0.2, ...]` 这样的向量，MSE 很大但余弦距离只是方向相反，模型会错误地惩罚方向相反的正确语义方向。

**缺少 IoU 相关损失：**

SmoothL1 对坐标的每个维度独立计算，不能直接优化框重叠程度（IoU）。当预测框与 GT 框完全不重叠时，SmoothL1 的梯度方向是"靠近 GT 坐标"，而真正需要的是"增大重叠面积"——这两个目标方向有时是不一致的。

**缺少 label smoothing：**

标准交叉熵在 one-hot 目标上训练，会驱使 logit 趋向 ±∞（过拟合到训练标签分布），在类别不平衡时问题更严重。

---

### 问题 4：学习率调度策略不适合当前配置

**原始调度器：**

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

总训练 epoch 为 80，调度节点如下：

| epoch 区间 | 学习率 |
|-----------|--------|
| 1–30 | 1e-4 |
| 31–60 | 1e-5 |
| 61–80 | 1e-6 |

第 60 epoch 之后学习率降至 `1e-6`，而标准 Adam 的有效更新量约为 `lr × 梯度`，`1e-6` 量级下模型几乎停止学习，最后 20 epoch 的训练时间被浪费。

`StepLR` 的阶梯式下降会在衰减点处造成 loss 的突变，可能使已收敛的局部结构被破坏。

---

### 问题 5：数据增强不足

**原始 `train` 阶段的变换：**

```python
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
transforms.ToTensor()
transforms.Normalize(...)
```

仅包含颜色扰动，**没有任何几何变换**。

对于边界框回归任务，模型必须学会在不同的空间位置和尺度下定位物体。没有几何增强意味着：

- 模型可能记住训练集中物体的固定位置偏好（数据集偏差）
- 对测试集中位置、尺度略有变化的物体泛化能力差

最基础的几何增强——随机水平翻转——同样适用于边界框任务，只需要镜像 x 坐标（`x → 1 - x`）即可。

---

## 三、改进方案与实现

### 改进 1：边界框坐标归一化 + Sigmoid 输出

**`annotation_dataset.py`，`resize_box` 函数：**

```python
# 改进后：缩放 + 归一化到 [0, 1]
x1 = box[0] * scale_w / target_w
y1 = box[1] * scale_h / target_h
x2 = box[2] * scale_w / target_w
y2 = box[3] * scale_h / target_h
# 同时 clamp 到合法范围
x1, x2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
y1, y2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])
```

**`annotation_model.py`，`BoxRegressionHead.forward`：**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(self.mlp(x))   # 输出严格约束在 (0, 1)
```

**效果：**

- 模型输出与训练目标尺度完全一致，SmoothL1 loss 值从 `[0, 224]` 量级降至 `[0, 1]` 量级
- sigmoid 函数在 `(-6, 6)` 范围内梯度均匀，初始化后输出约为 0.5，接近 GT 分布中心
- 推理时框坐标 `[0, 1]` 乘以 224 即为像素坐标，无需额外后处理

---

### 改进 2：消除 EmbeddingRegressionHead 的信息瓶颈

**改进前（串联结构）：**

```
feature(512) → cls_head → logits(17) → map_head → embedding(300)
```

**改进后（并联结构）：**

```
feature(512) ──→ cls_head  → logits(17)
             └─→ emb_head  → embedding(300)
```

**关键代码（`EmbeddingRegressionHead.forward`）：**

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits    = self.cls_head(x)   # feature → logits，独立学习
    embedding = self.emb_head(x)   # feature → embedding，独立学习
    return embedding, logits
```

**效果：**

- 词向量回归直接监督视觉特征到语义空间的映射，梯度不再经过分类瓶颈
- 分类头和 embedding 头可以针对各自任务独立优化特征空间
- `map_head`（17→512→300 的反常结构）被移除，参数量更合理

---

### 改进 3：多目标损失函数重设计

**新损失函数组成：**

| 损失项 | 公式/函数 | 权重 | 原始 |
|--------|-----------|------|------|
| subject_box SmoothL1 | `SmoothL1(pred_sub, tgt_sub)` | 1.0 | 1.0 |
| object_box SmoothL1 | `SmoothL1(pred_obj, tgt_obj)` | 1.0 | 1.0 |
| **GIoU Loss（新增）** | `1 - GIoU(pred, tgt)` 均值 | 1.0 | — |
| action 分类交叉熵 | `CE(logits, idx)` + label smoothing | **2.0** | 0.5 |
| object 分类交叉熵 | `CE(logits, idx)` + label smoothing | **2.0** | 0.5 |
| action embedding | `CosineEmbeddingLoss` | **0.5** | 1.0（MSE） |
| object embedding | `CosineEmbeddingLoss` | **0.5** | 1.0（MSE） |

**GIoU Loss 的意义：**

Generalised IoU 定义为：

$$\mathcal{L}_{GIoU} = 1 - \left( IoU - \frac{|C \setminus (A \cup B)|}{|C|} \right)$$

其中 $C$ 是包含预测框和目标框的最小外包框。

当预测框与目标框不重叠时，标准 IoU 梯度为零，SmoothL1 的梯度仅指向"靠近目标坐标"。GIoU 通过外包框惩罚项，在非重叠情况下仍然提供有意义的梯度，驱使框向目标方向移动。

**CosineEmbeddingLoss vs MSE：**

- MSE：$\mathcal{L} = \|v_{pred} - v_{target}\|_2^2$，优化欧氏距离
- Cosine：$\mathcal{L} = 1 - \cos(v_{pred}, v_{target})$，优化方向一致性

GloVe 词向量的语义相似性由余弦相似度定义，使用 MSE 会惩罚模量差异（与语义无关），使用 cosine loss 更忠实于任务目标。

**Label Smoothing（标签平滑，`ε=0.1`）：**

将 one-hot 目标从 `[0, 0, 1, 0, ...]` 软化为 `[ε/K, ε/K, 1-ε+ε/K, ε/K, ...]`。

防止模型对训练集标签过拟合，提高对相近类别的泛化能力（例如 `grasp` 和 `wrapgrasp` 在语义上相近）。

---

### 改进 4：学习率调度器替换

**改进后：**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["epoch"], eta_min=1e-6
)
```

余弦退火调度的学习率变化：

$$lr_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{\pi t}{T}\right)$$

| 特性 | StepLR | CosineAnnealingLR |
|------|--------|-------------------|
| 变化方式 | 阶梯式下降 | 连续平滑余弦曲线 |
| 中后期学习率 | 非常低（1e-6），模型停止学习 | 缓慢下降，保持有效更新 |
| 超参数敏感性 | 需要精确调节 step_size | 只需设置 T_max |
| 防止局部最优 | 差（lr 单调递减） | 较好（余弦振荡给模型逃离机会） |

同时将 `weight_decay` 从 `1e-5` 提高到 `1e-4`，加强 L2 正则化以抑制过拟合。

---

### 改进 5：同步几何数据增强

在 `__getitem__` 中加入随机水平翻转，图像翻转与边界框坐标镜像同步执行：

```python
if self.augment and random.random() < 0.5:
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # 同步镜像 x 坐标：x_new = 1 - x_old（在归一化空间内）
    sx1, sy1, sx2, sy2 = subject_box
    subject_box = [1.0 - sx2, sy1, 1.0 - sx1, sy2]
    ox1, oy1, ox2, oy2 = object_box
    object_box  = [1.0 - ox2, oy1, 1.0 - ox1, oy2]
```

ColorJitter 的强度也适度提高（`brightness/contrast: 0.2 → 0.3`），增强颜色不变性。

---

## 四、改进效果预期

### 训练稳定性

| 指标 | 原始 | 改进后 |
|------|------|--------|
| 初始 box loss 量级 | ~100（像素尺度 SmoothL1） | ~0.2（归一化 SmoothL1） |
| 初始梯度方差 | 极大（尺度不匹配） | 正常范围 |
| 训练曲线 | 振荡剧烈，前期缓慢收敛 | 平滑下降 |

### 推理质量

| 指标 | 原始 | 改进后 |
|------|------|--------|
| 未充分训练时框输出 | 接近 0（线性输出初始值） | 约 0.5×224=112（sigmoid 中点） |
| 框坐标范围 | 无约束，可能超出图像 | 严格约束在图像内 |
| 类别预测 | 受弱监督（权重 0.5） | 受强监督（权重 2.0） |

### 指标可观测性

新增 **mean IoU** 指标，在训练日志中直接显示框预测质量：

```
Epoch [10/80] ... | train_iou=0.312 val_iou=0.287 | train_act_cls=0.641 val_act_cls=0.589 ...
```

---

## 五、各文件改动汇总

### `annotation/annotation_model.py`

| 类/函数 | 改动内容 |
|---------|---------|
| `BoxRegressionHead.forward` | 末尾加 `torch.sigmoid()`；dropout 降至 0.1 |
| `EmbeddingRegressionHead` | 重构为并联结构，移除串联的 `map_head`；两路均直接从 backbone 特征出发 |
| `AnnotationLoss` | 加入 GIoU loss、CosineEmbeddingLoss 替换 MSE、CrossEntropyLoss 加 label_smoothing；分类权重 0.5→2.0，embedding 权重 1.0→0.5 |
| `AnnotationMetrics` | 新增 `mean_iou` 指标 |
| `build_annotation_loss` | 读取新 config 键（giou、action_cls、object_cls、label_smoothing） |

### `annotation/annotation_dataset.py`

| 函数 | 改动内容 |
|------|---------|
| `resize_box` | 改为缩放后除以 target_size，输出 [0,1]；加 clamp 和排序保证合法性 |
| `__getitem__` | 加入同步水平翻转增强（p=0.5）；`transform` → `base_transform` |

### `annotation/config_annotation.yaml`

- `loss_weights` 更新为 7 个键（新增 giou、action_cls、object_cls、label_smoothing）

### `annotation/train_annotation.py`

- `StepLR` → `CosineAnnealingLR(T_max=epoch)`
- `weight_decay` 1e-5 → 1e-4
- 训练日志增加 giou_loss 和 mean_iou

### `backend.py`

- `_box_to_list`：移除推理时的 `torch.sigmoid()`（模型内部已有）
- `_run_inference`：来自机器人和记忆库的框均做归一化（若 max>1 则除以 224）
