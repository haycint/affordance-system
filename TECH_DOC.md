# Affordance System — 超详尽技术文档

> 版本: 2026-05-16
> 适用范围: `e:/毕业设计/final_files/Affordance-system` 全部源代码
> 阅读对象: 系统维护者 / 后续研究者 / 答辩评审

---

## 目录

1. [系统总览](#1-系统总览)
2. [软件架构](#2-软件架构)
3. [推理主模型 (IAG / IAG_TextEmb) 原理](#3-推理主模型-iag--iag_textemb-原理)
4. [记忆系统架构](#4-记忆系统架构)
5. [可学习记忆组件 (Indexer / Aligner) 原理与训练](#5-可学习记忆组件-indexer--aligner-原理与训练)
6. [图像记忆系统](#6-图像记忆系统)
7. [标注模型 (Annotation) 原理](#7-标注模型-annotation-原理)
8. [前端](#8-前端)
9. [WebSocket 接口字典](#9-websocket-接口字典)
10. [数据流端到端示例](#10-数据流端到端示例)
11. [文件清单与目录结构](#11-文件清单与目录结构)

---

## 1. 系统总览

本系统围绕 **3D Affordance Grounding (3D 可供性定位)** 任务构建,
集成了 4 大子系统:

| 子系统 | 关键模块 | 角色 |
|---|---|---|
| 主推理 | `model/MyNet.py` (IAG)、`model/iag_textemb_pipeline.py` (IAG_TextEmb) | 输入 (image + point cloud + sub_box + obj_box) → 输出 per-point affordance 概率 + 17 类分类 logits |
| 偏好记忆 | `memory_system/memory_*.py` | 在交互中累积 *per-point preference* 矩阵, 通过相似检索 + 跨实例对齐增强后续推理 |
| 图像记忆 | `memory_system/image_memory_*.py` | 以 `(object, affordance)` 为复合键缓存图像 ROI 特征, 在 JRA 之前完成特征平均增强 |
| 标注 | `annotation/annotation_model.py` | 给定一张 RGB 图像 + 文本 GloVe 词向量,自动生成 subject_box / object_box / affordance label, 替代纯人工标注 |

四者通过统一的 `backend.py` (FastAPI + WebSocket `/ws`) 暴露给前端,
前端通过 [robot.html](frontend/robot.html), [monitor/watch.html](frontend/monitor/watch.html),
[monitor/train.html](frontend/monitor/train.html) 提供
**机器人模拟器 / 观察员 / 管理员** 三类不同视角的交互。

---

## 2. 软件架构

### 2.1 总体拓扑

```
                        ┌────────────────────────────────────────┐
                        │              FastAPI 进程              │
                        │           (backend.py, :8000)          │
                        │                                        │
   ┌─────────┐  WS /ws  │  ┌──────────────────────────────────┐  │
   │ robot.  │ ─────────┼▶│ STATE                            │  │
   │ html    │ ◀───────┼──│   main_model        (IAG/Textemb)│  │
   └─────────┘          │  │   annotation_models {s1, s2}     │  │
                        │  │   pref_memory  (MemoryManager)   │  │
   ┌─────────┐          │  │   image_memory (ImageMemoryMgr)  │  │
   │ watch.  │ ─────────┼▶│   robot_state[robot_id]          │  │
   │ html    │ ◀───────┼──│   robot_ids / user_ids / admin_id│  │
   └─────────┘          │  │   model_lock (asyncio.Lock)      │  │
                        │  └──────────────────────────────────┘  │
   ┌─────────┐          │            │             │             │
   │ train.  │ ─────────┼────────────┘             ▼             │
   │ html    │ ◀───────┼─── API_TABLE (24 个 async handler) ─── │
   └─────────┘          └────────────────────────────────────────┘
                            │           │           │
                            ▼           ▼           ▼
                       ┌────────┐ ┌─────────┐ ┌──────────┐
                       │FAISS + │ │FAISS +  │ │images/   │
                       │SQLite  │ │SQLite   │ │*.npy     │
                       │(pref)  │ │(image)  │ │          │
                       └────────┘ └─────────┘ └──────────┘
```

### 2.2 后端核心 (`backend.py`)

* **单一 WebSocket 入口**: `/ws`, 所有 API 通过 `{action, ...payload}` 的统一信封分发到 `API_TABLE[action]`。
* **24 个 async handler** (见第 9 节), 每个签名:
  `async def api_xxx(payload: Dict, conn_id: str) -> Dict`
* **全局单例 `STATE`** 保存模型、记忆、连接、设备。
* **三类连接角色**:
  * `robot`: `register_robot` 注册,持续上送 `infer_from_robot` 推理请求
  * `user`: `register_user` + `watch_robot` 订阅某机器人的可视化流
  * `admin`: `register_admin` (口令校验), 解锁 `train_*` / `push_to_memory` / `*_memory_entry` / `delete_*_memory` 等管理 API
* **响应规范**: `_ok({...})` / `_err("msg")` 包装。
* **管理员校验**: `_require_admin(uuid_)` 比对 `STATE.admin_id`。
* **异步并发安全**:
  * `STATE.model_lock` 控制 GPU 模型独占推理
  * `STATE.connections` (dict[conn_id, WebSocket]) 通过 `async with conn.send_json` 串行化输出
  * `MemoryManager` 内部 daemon 线程 + `queue.Queue` 做异步入库

### 2.3 三类前端

| 页面 | 角色 | 关键功能 |
|---|---|---|
| `frontend/robot.html` | 机器人模拟器 | 上传场景图 + 点云、获取 affordance 预测、人工反馈 (点选 + 邻域 K-NN + 优秀/成功/失败) |
| `frontend/monitor/watch.html` | 观察员 | 实时查看某机器人的点云 + 预测热力图; 与 robot.html 同款"反馈"区, 允许 watcher 也送反馈 |
| `frontend/monitor/train.html` | 管理员 | poweron 加载模型 / 训练触发 / 数据集采样 / **入库记忆浏览 — 删除 — 训练 indexer/aligner** |

三个页面共用 [frontend/monitor/shared.js](frontend/monitor/shared.js) 中的:
* `connect(host)` / `call(action, payload)` / `on(event, cb)` — WebSocket 客户端
* `makeScene(canvas)` / `renderCloud(scene, points, pref, oldObj, size)` — three.js 渲染
* `prefToColor(value)` — `[-1, +1]` → 红绿热力色映射
* 导出 `THREE` 与 `OrbitControls`, 供二级页面叠加 raycaster

---

## 3. 推理主模型 (IAG / IAG_TextEmb) 原理

实现于 [model/MyNet.py](model/MyNet.py)。 复现自论文
*"Grounding 3D Object Affordance from 2D Interactions in Images"*。

### 3.1 输入

| 张量 | 形状 | 含义 |
|---|---|---|
| `img` | `[B, 3, H, W]` | 含有 *interactive subject* 与 *interactive object* 的 RGB 图 |
| `xyz` | `[B, 3, 2048]` | 待预测的 3D 点云 (N_raw = 2048) |
| `sub_box` | `[B, 4]` | 主体框 (像素坐标) |
| `obj_box` | `[B, 4]` | 客体框 (像素坐标) |

### 3.2 6 阶段数据流

```
img ─▶ Img_Encoder (ResNet-18 layer4)
                       │
                       ▼  + sub_box / obj_box
       get_mask_feature ─▶ F_i [B,C,4,4]   ROI of object
                          F_s [B,C,4,4]   ROI of subject
                          F_e [B,C,H,W]   scene-mask feature
                                          (1 − sub_mask ∪ obj_mask)
                       │
                       │ roi_align(F_e, full-image-box) → [B,C,4,4]
                       │
xyz ─▶ Point_Encoder (PointNetSetAbstractionMSG ×3)
                       │
                       ▼ F_p_wise[-1][1]   [B, C=512, N_p=64]
                       │
                       ▼ Joint Region Alignment
                          JRA(F_i [B,C,16],  F_p [B,C,64]) → F_j [B, N_p+N_i, C]
                       │ - 双向 cross-modulation
                       │ - inherent self-attention
                       ▼
                       Affordance Revealed Module
                          ARM(F_j, F_s, F_e) → affordance [B, N_p+N_i, C]
                       │ - F_j 作为 query, F_s/F_e 作为 KV (cross-attn)
                       ▼
                       Decoder (PointNet++ FP)
                          (_3daffordance, logits, to_KL)
                          _3daffordance: [B, N_raw, 1]  sigmoid后的逐点概率
                          logits:        [B, num_affordance=17]  分类
                          to_KL:         [F_ia.mT, I_align.mT]   蒸馏对齐用
```

### 3.3 关键模块逐项说明

#### Img_Encoder
* 复用 `torchvision.models.resnet18(weights=IMAGENET1K_V1)`,
  截取 `conv1 + bn1 + relu + maxpool + layer1..4`,
  最终输出 `[B, 512, 7, 7]` (224 输入)。

#### Point_Encoder (PointNet++ MSG)
* 三级 `PointNetSetAbstractionMsg`,
  逐级下采样 `2048 → 512 → 128 → N_p=64` 点;
  每级使用多尺度球查询 (radius=[0.1,0.2,0.4] 等)。
* 输出层级特征 `[l0_xyz, l0_pt]..[l3_xyz, l3_pt]` 供后续 FP 上采样。

#### Joint_Region_Alignment (JRA)
* 输入 `F_i:[B,C,H,W]` 与 `F_p:[B,C,N_p]`。
* `to_common` Conv1d block 把两者投影到公共空间。
* 双向 cross-modulation:
  ```
  φ      = P.T @ I  / √C
  P_enh  = I @ softmax(φ, dim=-1).T    # I-driven point enhance
  I_enh  = P @ softmax(φ, dim=1)       # P-driven image enhance
  ```
* 各自经过 `Inherent_relation` (多头自注意力 + LN 残差) 后拼接成 `F_j [B, N_p+N_i, C]`。

#### Affordance_Revealed_Module (ARM)
* `Cross_Attention(query=F_j, KV1=F_s, KV2=F_e)` 得到 `Θ_1, Θ_2`。
* 拼接 → `Conv1d(2C→C)` → ReLU → `affordance [B, N_p+N_i, C]`。
* **这是记忆 Indexer 的输入特征 (见 §5)**。

#### Decoder
* 用 PointNetFeaturePropagation 把 N_p=64 点的特征上采样回 N_raw=2048。
* `cls_head`: AvgPool 后 `[2C → C/2 → num_affordance]` 得分类 logits。
* `out_head`: 逐点 MLP + sigmoid 得 `[B, N_raw, 1]` 概率图。

### 3.4 IAG_TextEmb 变体

* 仅 Stage 6 替换为 `Decoder_TextEmb`, 输入额外的 `text_emb` (300-D GloVe → 512-D 投影),
  与 `F_j`、`affordance` 拼接 (`3*emb_dim`) 后进入分类与解码。
* 用途: 允许 same 视觉特征下根据"动作语义文本"切换 affordance 预测。

### 3.5 训练入口

* `train_textemb.py` 包含完整训练循环, 损失为:
  * Per-point BCE on `_3daffordance` ↔ GT mask
  * Classification CE on `logits` ↔ affordance label
  * KL distillation on `to_KL` (image-vs-point alignment)
* 数据集: `data_utils/dataset.py`, 读取 PIAD-like 数据。

---

## 4. 记忆系统架构

### 4.1 设计哲学

> 让模型"在不重新训练参数的前提下,从过去成功/失败的交互中受益"。

具体做法: 每完成一次 (image + point cloud) 推理 + 人工或自动反馈后,
将以下信息打包成 `MemoryEntry` 并入库:

| 字段 | 形状 | 作用 |
|---|---|---|
| `index_vector` | `[128]` L2 归一 | FAISS 检索键 (近邻搜索) |
| `point_cloud` | `[N, 3]` | 对齐时供几何参考 (当前实现未直接用) |
| `point_features` | `[N, 512]` | **对齐器的 K** (历史每点特征) |
| `preference_matrix` | `[N]` | **对齐器的 V** (历史每点偏好 ∈ [-1, +1]) |
| `reward` / `confidence` / `outcome` | scalar/str | 融合权重 |
| `affordance_label` / `object_category` | str | 过滤条件 |
| `timestamp` / `access_count` | scalar | 时间衰减 + LRU 淘汰 |

### 4.2 模块组成

```
                       MemoryManager
                       ─────────────
form_memory(arm_feat, …) ──┐                ┌── retrieve_and_fuse(arm_feat, current_pts,…)
                           │                │
                ┌──────────▼─────┐    ┌─────▼─────────┐
                │ MemoryIndexer  │    │ MemoryIndexer │ (query)
                │ (nn.Module)    │    │ same instance │
                │ ARM_feat→128-D │    │ ARM_feat→128-D│
                └────────┬───────┘    └─────┬─────────┘
                         │                  │
                         ▼                  ▼
                ┌───────────────┐    ┌──────────────────┐
                │  MemoryStore  │◀──▶│ MemoryRetriever  │
                │ FAISS+SQLite  │    │ filter(aff,outc, │
                │ index_vector  │    │   reward,sim)    │
                └───────────────┘    └─────┬────────────┘
                         ▲                 │ entries
                         │                 ▼
                         │           ┌──────────────────┐
              MemoryEntry│           │  MemoryAligner   │
                         │           │ cross-attn       │
                         │           │ F_curr × F_hist  │
                         │           │ → Pref_curr_k    │
                         │           └─────┬────────────┘
                         │                 │ aligned_prefs[K]
                         │                 ▼
                         │           ┌──────────────────┐
                         │           │  MemoryFusion    │
                         │           │ softmax(reward)  │
                         │           │ + time decay     │
                         │           └─────┬────────────┘
                         │                 │ Pref_fused
                         │                 ▼
                         │           apply_to_output:
                         │           final = σ(raw + α·Pref_fused)
                         │
            async daemon thread + queue.Queue
            (form_memory 非阻塞入库)
```

### 4.3 双路存储 (`memory_store.py`)

* **向量路径**: FAISS `IndexFlatIP(128)` (内积 ≈ 余弦因 L2-norm 输入); 不可用时退化为 NumPy 暴力 L2。
* **结构路径**: SQLite 表 `memories(id PK, data BLOB JSON, reward, outcome, timestamp, object_category, affordance_label, confidence, access_count)`, 双索引 `idx_affordance`, `idx_outcome`。
* **同步**: `_lock = threading.Lock()` 保证 FAISS + SQLite 同步增删。
* **淘汰**: `_evict_lru` 按 `(access_count, timestamp)` 升序淘汰。
* **分页**: `list_all(page, per_page)` 提供给管理员前端逐页浏览, 只返回轻量元数据避免大 BLOB 传输 ([memory_store.py:309-355](memory_system/memory_store.py#L309-L355))。

### 4.4 MemoryRetriever (`memory_retriever.py`)

* **过采样 → 后过滤** 模式: 取 `3 × top_k` 邻居, 按 `affordance_label` / `outcome` / `min_reward` / `similarity_threshold` 过滤, 截断到 `top_k`。
* 这样做是因为 FAISS 无法在索引阶段做这些谓词过滤,但实际仓库不大 (≤5k),代价可接受。

### 4.5 MemoryFusion (`memory_fusion.py`)

无状态融合器, NumPy / Torch 双实现:

```
score_k = reward_k                      # 基础分
        [* confidence_k]                # 置信度门 (可选)
        [+ λ * exp(-(now - t_k)/3600)]  # 时间衰减 (可选)
w = softmax(score · T)                  # 温度 T 控制锐度
Pref_fused = Σ_k w_k · Pref_curr_k
```

最终应用:
```
final = σ( raw_logits + α · Pref_fused )    α=0.3 (温和推动, 默认)
```

### 4.6 MemoryManager (`memory_manager.py`)

* 持有 indexer / store / retriever / aligner / fusion 五个子组件。
* 提供 4 个用户级方法:
  * `form_memory(...)` — 完成一次交互后入库 (异步, 默认走后台线程)
  * `retrieve_and_fuse(...)` — 推理前/中, 检索 + 对齐 + 融合, 返回 `Pref_fused [N_curr]`
  * `apply_memory_to_output(raw, pref)` — 把融合偏好作为残差加到 logits
  * `enhance_prediction(...)` — 上述 3 步的一站式封装
* 同时提供 3 个静态工具生成 preference matrix:
  * `generate_preference_from_ground_truth`
  * `generate_preference_from_prediction` — 用于自监督
  * `generate_preference_from_interaction` — 物理交互的球形邻域

### 4.7 持久化

* SQLite 始终持久化在 `<store_dir>/memories.db`。
* FAISS 索引按需 `save_index()` 写入 `<store_dir>/faiss.index` + `.ids` 文本映射。

---

## 5. 可学习记忆组件 (Indexer / Aligner) 原理与训练

### 5.1 MemoryIndexer

[memory_system/memory_indexer.py](memory_system/memory_indexer.py)

```
ARM 输出  F_a [B, N_p+N_i, C=512]
            │
            ▼  global average pool (或 channel-attention 加权)
        v_global [B, 512]
            │
            ▼  proj head:
            │   Linear(512 → 256)
            │   BatchNorm1d(256)
            │   ReLU
            │   Linear(256 → 128)
            ▼
        v_proj  [B, 128]
            │
            ▼  F.normalize(p=2, dim=-1)
        v_index [B, 128]     ← FAISS 检索键
```

* 默认 `pooling="avg"`; 可选 `"weighted"` 触发通道注意力 `Sigmoid(Linear(C → C/4 → C))`。
* 推理时调用 `compute_index_numpy(arm_feature)`, 内部 `torch.no_grad()` + CPU。
* **训练监督**: 静态方法 `contrastive_loss(v1, v2, labels, T=0.07)`:
  * 输入两次扰动视图 v1, v2,标签即 affordance 类别 id。
  * 计算 `sim = v1 · v2.T / T`,
    构造正样本 mask = `label_i == label_j` (对角排除)。
  * 损失: 标准 SupCon (Supervised Contrastive) 形式
    `L = −E_i [ Σ_{j∈P(i)} log(exp(sim_ij)/Σ_k exp(sim_ik)) / |P(i)| ]`
  * 物理意义: 把同一 affordance 的不同实例在 128-D 空间中拉近, 不同 affordance 推远 → 检索时更可能命中相关记忆。

### 5.2 MemoryAligner

[memory_system/memory_aligner.py](memory_system/memory_aligner.py)

#### 任务

> 给定 **当前**点云的 `F_curr [N_curr, 512]` 与 **历史**记忆的 `(F_hist [N_hist, 512], Pref_hist [N_hist])`,
> 把 *N_hist 个点上的历史偏好* "迁移"到 *N_curr 个当前点上*, 输出 `Pref_curr [N_curr]`。

#### 数学定义 (单头)

```
S = F_curr @ F_hist^T               # [N_curr, N_hist]  相似度矩阵
A = softmax(S / (√D · T))           # 注意力权重
Pref_curr = A @ Pref_hist           # [N_curr]
```

`T` (`temperature`) 控制软硬程度;更低 → 更尖锐 (只看少数邻居)。

#### 多头实现

* `feat_dim=512, num_heads=4, head_dim=128`。
* `Q,K = reshape(N, num_heads, head_dim).transpose(0,1) → [H, N, D/H]`。
* 每头独立做 `softmax(Q @ K.T) @ Pref_hist`,
  最后 **跨头求平均** 得到 `Pref_curr [N_curr]`。
* 多头可以并行捕捉不同语义相似度模式 (比如局部几何 + 语义方位)。

#### 可学习投影

* `use_learned_projection=True` 时, 两侧特征同时通过共享 `Linear → LayerNorm → ReLU`,
  得到任务特化的对齐空间; 比直接用 backbone 特征更稳健。

#### NumPy 兜底

* `align_numpy(F_curr, F_hist, Pref_hist)`:
  L2-norm 之后做余弦相似度 → softmax → 加权, **无梯度、无 learned proj**, 用于轻量推理。

### 5.3 端到端训练脚本

[memory_system/train_index_align.py](memory_system/train_index_align.py)

#### 5.3.1 数据源

直接读取已填充的 `MemoryStore` SQLite 数据库, 不依赖额外数据集:

```python
class PrefMemoryDataset(Dataset):
    def __init__(self, store, max_points=64):
        listing = store.list_all(page=1, per_page=10**9)
        self.ids = [...]
        # 预缓存所有 MemoryEntry 到 self._cache (实测条目少, 几 MB)
        # 构造 affordance label vocab self.label_to_id
    def __getitem__(self, idx):
        # 取 point_features [N, 512], preference [N], affordance label int
        # 截断 / 重复 padding 到 max_points=64
        return feats, prefs, label
```

返回 `(feats [B,N,D], prefs [B,N], labels [B])` 三元组。

#### 5.3.2 损失组合

1. **Indexer 监督对比损失**
   * 输入: `feats [B, N, D]`, 在前面拼接全局均值 token →
     `arm_seq = cat([feats.mean(1,keepdim=True), feats], dim=1)`
   * 两次高斯噪声扰动 (`σ=0.02`) 得到 `v1, v2`。
   * `loss_idx = MemoryIndexer.contrastive_loss(v1, v2, labels)`。
2. **Aligner 自监督 split-reconstruction**
   * 每个 entry 随机切分点为 `A, B` 两半。
   * 让 aligner 用 `(F_B, Pref_B)` 预测 `Pref_A`, MSE 损失。
   * 含义: 学到的注意力必须真正反映 feature → label 的对应关系。
3. **Aligner 跨实例 cross-entry 损失**
   * 同 batch 中找同 affordance 的 `(i, j)` 对, 让 aligner 用 j 预测 i。
   * 损失: `1 − cos(Pref_pred_centered, Pref_target_centered)` (零均值后余弦)。
   * 这是 *真正* 的 "跨物体迁移"信号 — 因为是不同实例的几何与点数, 必须依赖语义对齐才能成功。

总损失:
```
L = w_idx · L_idx + w_split · L_split + w_cross · L_cross
  default: 1.0,    1.0,         0.5
```

#### 5.3.3 训练循环

* Adam, lr=1e-3, gradient clip 5.0;
* 每 epoch 输出 idx / split / cross 三项损失;
* 50 epoch (默认) 后保存:
  ```
  {out_dir}/indexer.pt
  {out_dir}/aligner.pt
  ```
  仅保存 `state_dict`。

#### 5.3.4 CLI

```bash
python -m memory_system.train_index_align \
    --store_dir ./memory_store \
    --out_dir   ./memory_system/checkpoints \
    --epochs    50 \
    --batch_size 8 \
    --lr 1e-3
```

#### 5.3.5 在 backend.py 中触发

管理员前端 (train.html) 按下 `开始训练 Indexer/Aligner` →
`call('train_index_align', {epochs,batch,lr,store_dir})` →
`api_train_index_align` 用 `subprocess.Popen` 启动 `python -m memory_system.train_index_align`,
**实时把 stdout 逐行 push 给前端 (`train_log` 事件, stream='index_align')**,
完成时发 `train_done`, 训练结束自动落盘到 `pref_dir/{indexer,aligner}.pt`。

#### 5.3.6 在 poweron 时加载 checkpoint

`api_poweron` 中, `MemoryManager` 初始化后, 用如下逻辑装载权重 (向后兼容):

```python
def _resolve_ckpt(spec_key, default_name):
    p = spec.get(spec_key, "")
    if p and os.path.exists(p): return p
    cand = os.path.join(pref_dir, default_name)
    return cand if os.path.exists(cand) else None

idx_ckpt = _resolve_ckpt("indexer_ckpt", "indexer.pt")
if idx_ckpt:
    sd = torch.load(idx_ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    STATE.pref_memory.indexer.load_state_dict(
        {k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
# 同理 aligner.pt
```

无 checkpoint 时降级为随机初始化 (检索仍可用, 但精度低)。

---

## 6. 图像记忆系统

### 6.1 目标

补充偏好记忆: 在 **JRA 之前**, 借助同 `(object, affordance)` 历史图像的 ROI 特征,
平滑当前样本的图像特征, 提升 few-shot 推理稳定性。

### 6.2 ImageMemoryStore

[memory_system/image_memory_store.py](memory_system/image_memory_store.py)

* SQLite 表 `image_memories`, **复合主键索引** `(object_category, affordance_label)`:
  ```
  id PK, object_category, affordance_label, image_path,
  image_feature BLOB,  sub_box BLOB,  obj_box BLOB,
  F_i_blob,  F_s_blob,  F_e_blob, confidence, timestamp, access_count
  ```
* 原始图像作为 `.npy` 存到 `<store_dir>/images/<id>.npy`,
  避免把 H×W×3 字节灌进 SQLite。
* `image_feature` 是 ResNet18 layer4 输出 `[C,h,w]` 池化到 `[C=512]` 的向量,
  FAISS `IndexFlatIP` 用于 *soft similarity* 查询。
* `F_i / F_s / F_e` 直接缓存了 IAG_TextEmb 的 ROI/scene 特征,
  retrieval 阶段无需重跑 backbone。
* 淘汰策略: 每个 `(object, affordance)` key 内 FIFO, 超 `max_images_per_key=50` 淘汰最老。
* `list_all(page, per_page)` 与 `list_categories()` 服务管理员前端。

### 6.3 ImageMemoryManager

[memory_system/image_memory_manager.py](memory_system/image_memory_manager.py)

* 三种 averaging 策略:
  * `mean`: `F_avg = (F_cur + F_mem.mean) / 2`
  * `weighted`: `F_avg = α F_cur + (1−α) F_mem.mean`
  * `attention`: 学习门控 `g = σ(MLP([F_cur, F_mem]))`, `F_avg = g·F_cur + (1−g)·F_mem`
* `retrieve_and_average_feature(F_cur, object, affordance)`:
  1. 在 store 中查 `(object, affordance)` 命中, 取 `top_k_memory_images` 个;
  2. 把所有命中图像特征做空间平均池化, 与 `F_cur` 加权融合;
  3. 返回 `[B, C, h, w]` 形状不变, 可直接代入 `get_mask_feature(...)`。
* 写入: `add(image, image_feature, object, affordance, sub_box, obj_box, F_i, F_s, F_e)`。

### 6.4 与主推理的接入点

在 `forward_from_image_feature` (或后端的等价路径) 调用 `image_memory.retrieve_and_average_feature(F_I, ...)` 取代原 `F_I`, 让 JRA + ARM 看到的图像特征是"当前图 + 历史同类图"的折中。

---

## 7. 标注模型 (Annotation) 原理

替代纯人工划框 / 标 affordance。 配置文件 [annotation/config_annotation.yaml](annotation/config_annotation.yaml) 中:

* `affordance_labels`: 17 类 (grasp, sit, lift, ...)
* `object_labels`: 23 类
* `word_embed_dim`: 300 (GloVe)

提供两种实现, 由 `scheme` 配置选择。

### 7.1 Scheme 1 — 空间感知端到端模型 (`AnnotationModelScheme1`)

[annotation/annotation_model.py](annotation/annotation_model.py)

#### 7.1.1 输入

* `img [B, 3, 224, 224]` (归一化)
* `object_wv [B, 300]` GloVe 物体词向量

#### 7.1.2 主体架构

```
img ─▶ ResNet18 (去掉 avgpool/fc) ─▶ feat_map [B, 512, 7, 7]
object_wv ─▶ Linear(300→256)+LN+ReLU ─▶ text_vec [B, 256]

                    ┌── FiLM_sub(feat_map, text_vec) ─▶ feat_sub [B,512,7,7]
                    │
                    └── FiLM_obj(feat_map, text_vec) ─▶ feat_obj [B,512,7,7]
                                │
                                ▼
                       SpatialBoxHead_sub  ─▶ (cx, cy, w, h)_sub + heatmap_sub
                       SpatialBoxHead_obj  ─▶ (cx, cy, w, h)_obj + heatmap_obj

           GAP(feat_obj) ⊕ text_vec ─▶ MLP(768 → 512 → 17) ─▶ action_logits
```

#### 7.1.3 FiLMBlock — 文本条件化的关键

```
γ = 1 + Linear_γ(text_vec)        # 初始化为零 → γ≈1 (恒等)
β =     Linear_β(text_vec)
out = γ · feat + β               # 通道维 broadcast
```

意义: 让 *物体词向量* 选择"哪些视觉通道与该物体相关",
但不破坏 7×7 空间结构 — 这是 box 回归的关键!

#### 7.1.4 SpatialBoxHead — 空间归纳偏置

```
feat [B, C, 7, 7]
   │ Conv-BN-ReLU ×2 (hidden=128)
   ▼ h [B, 128, 7, 7]
   │ Conv2d(128→1) → heatmap_logits [B, 7, 7]
   │       │
   │       ▼  flatten + softmax
   │   prob_2d [B, 7, 7]
   │       │ soft-argmax: 把 grid 坐标乘 prob 求期望
   │       ▼
   │   (cx, cy) ∈ [0, 1]^2
   │
   │ pooled = (h × prob_2d.unsqueeze(1)).sum((2,3))    # heatmap 加权池化
   ▼
   wh_head MLP → log_wh → softplus → wh ∈ (0,1]
                       ↑ bias 初始化为 0 → softplus(0)≈0.69 ≈ 图像 7 成
                         (远离"塌缩到零"的退化解)
   ▼
   (cx, cy, w, h)
```

**为何用 soft-argmax**: 普通 MLP 直接出 `(cx,cy)` 在数据集均值处容易塌陷,
soft-argmax 是 grid 坐标的凸组合, 只在 heatmap 真正均匀时才会塌到中心 —
监督会立刻把它扭出来。

#### 7.1.5 损失 (`AnnotationLoss`) — CIoU + 反塌缩

历史问题: 旧 SmoothL1 + GIoU 组合下,模型偏好"小而居中"的均值框 (因为 SmoothL1 量纲极小, GIoU 信号被淹没)。 重新设计:

| 分量 | 公式 | 权重 | 作用 |
|---|---|---|---|
| L1(cxcywh) | `L1(pred_cxcywh, tgt_cxcywh)` | 1.0 | 与解码器同坐标系, 梯度尺度一致 |
| CIoU | `1 - IoU + ρ²(c,c_gt)/c_diag² + α·v` (v: 长宽比一致项) | **5.0** | 主定位信号 |
| Size-reg | `mean( max(0, 0.25·tgt_area − pred_area) )` | 0.5 | **反塌缩**: 预测面积 < 25% 目标时才有罚, 一旦达到立刻归零 |
| Action CE | label_smoothing=0.1 的交叉熵 | 1.0 | 17 类分类 |

CIoU 公式细节:
```
v = (4/π²) · ( atan(w_gt/h_gt) − atan(w/h) )²
α = v / ( (1 − IoU) + v )      # detached, 仅作权重
CIoU = IoU − ρ²/c² − α·v
L_CIoU = 1 − CIoU
```
ρ² = 中心点距离的平方, c² = 最小外接矩形对角线平方。

#### 7.1.6 评估指标

* `action_cls_acc`: argmax 准确率
* `mean_iou`: sub_box / obj_box 的平均 IoU (xyxy)

### 7.2 Scheme 2 — 多模型协作 (`AnnotationModelScheme2`)

三个独立 sub-model (不共享 backbone, 也可选 `share_backbone=True`):

```
BoxSubModel     img ─▶ ResNet18.avgpool → 512 ─▶ BoxRegressionHead ×2 ─▶ (sub_box, obj_box)
ActionSubModel  img ─▶ ResNet18         → 512 ─▶ EmbeddingRegressionHead → (act_embed[300], act_logits[17])
ObjectSubModel  img ─▶ ResNet18         → 512 ─▶ EmbeddingRegressionHead → (obj_embed[300], obj_logits[29])
```

总输出 4+4+300+300 = 608 维。 没有空间归纳偏置, 主要用作对照。

### 7.3 训练入口

[annotation/train_annotation.py](annotation/train_annotation.py) 提供 CLI 训练循环,
后端 API `train_annotation` 也可调用 (admin-only)。

---

## 8. 前端

### 8.1 robot.html — 机器人模拟器

* 输入: 上传图像 + 点云 (npy/obj/json), 选择 affordance + object 类别。
* 推理: `call('infer_from_robot', {...})` → 收到 `infer_done` 事件
  → `renderCloud(scene, points, preference, ...)` 绘制热力图。
* 反馈: 鼠标点选点 → K-NN 邻域加亮 (黄色) → 选"优秀/成功/失败" → `call('feedback', {robot_id, preference, outcome})`。

### 8.2 monitor/watch.html — 观察员

* `watch_robot(robot_id)` 订阅, 收到 `robot_state` 推送实时刷新画面。
* **本次新增**: 反馈区与 robot.html 同款,
  允许观察员也参与标注:
  * THREE.Raycaster (`threshold=0.02`) 取最近点;
  * `pickNeighbors(centerIdx, n)` 实现欧式 KNN 选择邻域;
  * `highlightSelection()` 重画 preference 然后把选中点强制涂黄 `(1.0, 1.0, 0.0)`;
  * 提交时 `pref[selected] = 1.0` 后调用 `feedback` API。

### 8.3 monitor/train.html — 管理员控制台

* **训练面板**: 主模型训练 / 标注模型训练。
* **数据集采样面板**: 随机抽一条样本以验证。
* **入库记忆面板** (本次新增, 第二个 `<script type="module">` 块):
  * "刷新 Pref 记忆" / "刷新 Image 记忆" 按钮, 分页列出条目。
  * 点击条目 → 调 `get_pref_memory_entry` / `get_image_memory_entry`
    → base64 还原点云 + 偏好, 在右侧 canvas 渲染:
    * Pref: `renderCloud(scene, ptsArr, pref, oldObj, size=4)`
    * Image: 原图 + 蓝框 `sub_box` (#1f6feb) + 红框 `obj_box` (#a02525)
      (自动检测归一化 vs 像素坐标)。
  * "删除" 按钮调 `delete_pref_memory` / `delete_image_memory`。
  * "训练 Indexer/Aligner" 按钮:
    填写 `epochs / batch / lr / store_dir` → `call('train_index_align', {...})`
    → 监听 `train_log` (stream='index_align') 实时滚动输出 → 收到 `train_done` 停止。
* `adminWatcher` 每 500ms 轮询 admin 鉴权状态, 解锁/锁定按钮。

### 8.4 shared.js (`frontend/monitor/shared.js`)

* WebSocket 单例 + 事件总线 `on/off/once/emit`。
* `prefToColor(v)`: `v ∈ [-1, +1]` → 红 (-1) → 灰 (0) → 绿 (+1) 插值。
* `renderCloud(scene, pts, pref, oldObj, size)`:
  THREE.BufferGeometry + PointsMaterial(`vertexColors: true`);
  返回 `{positions, preference}` 供二级 hook (反馈、选区) 复用。

---

## 9. WebSocket 接口字典

注册在 `API_TABLE` 中, 调用形式: `{ "action": "<name>", ...payload }`。

### 9.1 连接管理 (3)

| action | role | 功能 |
|---|---|---|
| register_robot | any | 注册为某 robot_id 的机器人 |
| register_user | any | 注册为 user, 拿到 user_id |
| watch_robot | user | 订阅 robot_id 推送 |
| register_admin | any | 管理员口令验证 |

### 9.2 模型生命周期 / 推理 / 反馈 (6)

| action | role | 功能 |
|---|---|---|
| poweron | admin | 加载主模型 + 标注模型 + 记忆系统 + indexer/aligner checkpoint |
| infer_from_robot | robot | (image + point cloud) → 预测 + 可选记忆增强 |
| infer_from_user_img | user | 用户图推理 (用于演示) |
| infer_user_pref | user | 用户偏好热力图 |
| feedback | robot/user | 写入偏好缓存; 主模型在则同时 `_capture_pref_entry` 准备 pref_memory 入库 |
| annotate | user | 自动标注 (Scheme1/2) |

### 9.3 数据集 + 训练 (3)

| action | role | 功能 |
|---|---|---|
| dataset_sample | admin | 抽数据集样本 |
| train_main | admin | 触发 train_textemb.py |
| train_annotation | admin | 触发 annotation 训练 |

### 9.4 记忆管理 (11) — 本次扩充

| action | role | 功能 |
|---|---|---|
| list_memory_cache | admin | 列出 memory_cache 目录下待入库 npz |
| push_to_memory | admin | 把 cache 推到 pref_memory + image_memory |
| toggle_pref_memory | admin | 开关 pref_memory 增强 |
| **list_pref_memory** | admin | 分页列出 pref memory entries |
| **get_pref_memory_entry** | admin | 取单条 (含 point_cloud_b64, preference_b64, scene_image) |
| **delete_pref_memory** | admin | 删条目 |
| **list_image_memory** | admin | 分页列出 image memory entries |
| **get_image_memory_entry** | admin | 取单条 (PNG base64 + sub_box + obj_box) |
| **delete_image_memory** | admin | 删条目 (含 .npy 文件清理) |
| **train_index_align** | admin | 子进程训练 Indexer/Aligner, 流式 stdout |

### 9.5 响应规范

* 成功: `{"ok": true, ...result}`
* 失败: `{"ok": false, "error": "..."}`
* 异步推送 (服务端主动): `train_log` / `train_done` / `robot_state` / `infer_done` 等。

---

## 10. 数据流端到端示例

### 10.1 一次"记忆增强推理"

```
robot.html
  │ ① 上传 image + xyz + sub_box + obj_box + (object, affordance)
  ▼
backend.api_infer_from_robot
  │ ② IAG/IAG_TextEmb forward → arm_feat, _3daffordance(raw), logits
  │
  │ ③ if pref_memory enabled:
  │      Pref_fused = manager.retrieve_and_fuse(
  │          arm_feature=arm_feat,
  │          current_point_cloud=xyz,
  │          current_point_features=F_p,
  │          affordance_label=...)
  │      final = σ(raw + 0.3 · Pref_fused)
  │
  │ ④ broadcast 'robot_state' (points, final, image, ...)
  ▼
watch.html       robot.html
   ├─ 渲染热力图 ◀──┘
   │
   ⑤ 用户在 watch.html / robot.html 选点 → 'feedback'
                  │
                  ▼
backend.api_feedback
   ⑥ 落 npz 到 memory_cache/, 同时 _capture_pref_entry 抓 ARM 特征
                  │
                  ▼
admin (train.html) 看到 cache → 'push_to_memory'
                  │
                  ▼
backend.api_push_to_memory
   ⑦ form_memory + ImageMemoryManager.add  → SQLite + FAISS
                  │
                  ▼ 异步入库 (后台线程)
   ⑧ 下次推理时 ③ 步即可命中刚入库的记忆
```

### 10.2 一次"在线训练 Indexer/Aligner"

```
train.html 按下"训练 Indexer/Aligner"
  │
  ▼ call('train_index_align', {epochs:50, batch:8, lr:1e-3, store_dir:...})
backend.api_train_index_align
  │ subprocess.Popen([python, '-m', 'memory_system.train_index_align', ...])
  │ for line in proc.stdout: send_json('train_log', {stream:'index_align', line})
  ▼
memory_system/train_index_align.py
  │ PrefMemoryDataset(store)
  │ loop epochs:
  │    indexer(arm_seq_v1), indexer(arm_seq_v2) → contrastive_loss
  │    aligner split & cross losses
  │    Adam step
  │ torch.save(indexer.state_dict(), pref_dir/indexer.pt)
  │ torch.save(aligner.state_dict(), pref_dir/aligner.pt)
  ▼
backend send_json('train_done', {ok:true})
  │
  ▼ 下次 poweron 自动加载 .pt
```

---

## 11. 文件清单与目录结构

```
Affordance-system/
├── backend.py                        ← FastAPI + WebSocket 路由, 24 API
├── model/
│   ├── MyNet.py                      ← IAG: Img_Encoder, Point_Encoder, JRA, ARM, Decoder
│   ├── iag_textemb_pipeline.py       ← IAG_TextEmb 6-stage 切片
│   ├── iag_pipeline.py               ← IAG 切片
│   ├── pipeline_api_server.py        ← 切片推理 RPC server
│   ├── pipeline_dataflow.py          ← 切片调度
│   ├── model_slicer.py / sliced_model.py
│   └── pointnet2_utils.py            ← PointNet++ MSG primitives
├── annotation/
│   ├── annotation_model.py           ← Scheme1 (FiLM + spatial heatmap + CIoU) / Scheme2
│   ├── annotation_dataset.py
│   ├── annotation_tool.py
│   ├── train_annotation.py
│   └── config_annotation.yaml
├── memory_system/
│   ├── memory_entry.py               ← MemoryEntry dataclass (base64 JSON ⇄ numpy)
│   ├── memory_indexer.py             ← MemoryIndexer (proj head + SupCon loss)
│   ├── memory_aligner.py             ← MemoryAligner (multi-head cross-attn)
│   ├── memory_fusion.py              ← reward-weighted softmax + α residual
│   ├── memory_retriever.py           ← over-fetch 3x + post filter
│   ├── memory_store.py               ← FAISS + SQLite (含本次新增 list_all 分页)
│   ├── memory_manager.py             ← 5 子组件 orchestrator + async daemon
│   ├── image_memory_store.py         ← (object, affordance) 复合键 SQLite + FAISS
│   ├── image_memory_manager.py       ← retrieve_and_average_feature
│   ├── train_index_align.py          ← (本次新增) Indexer + Aligner 联合训练
│   ├── prepopulate_image_memory.py
│   └── integration.py
├── frontend/
│   ├── robot.html                    ← 机器人模拟器
│   ├── monitor/
│   │   ├── shared.js                 ← WS 客户端 + THREE 渲染
│   │   ├── watch.html                ← (本次新增反馈区) 观察员
│   │   └── train.html                ← (本次新增入库浏览/删除/训练) 管理员
│   └── monitor_launcher.py
├── data_utils/
│   └── dataset.py
├── utils/
│   ├── eval.py, loss.py, utils.py, visualization.py
├── config/                           ← 模型/记忆 spec 文件
├── memory_libraries.json             ← 记忆库注册
├── train_textemb.py                  ← IAG_TextEmb 训练入口
└── requirements.txt
```

### 11.1 关键配置示例 (poweron payload)

```json
{
  "action": "poweron",
  "uuid": "<admin uuid>",
  "spec": {
    "model_ckpt": "./model_list/iag_textemb_best.pt",
    "annotation_ckpt_s1": "./annotation/scheme1.pt",
    "annotation_ckpt_s2": "./annotation/scheme2.pt",
    "pref_memory_dir": "./memory_store",
    "image_memory_dir": "./image_memory_store",
    "indexer_ckpt": "",        // 留空则用 pref_dir/indexer.pt
    "aligner_ckpt": "",        // 留空则用 pref_dir/aligner.pt
    "use_textemb": true
  }
}
```

---

## 附录 A — 关键设计决策与权衡

| 决策 | 备选 | 选择理由 |
|---|---|---|
| Indexer 用 SupCon, 而非 triplet | triplet, NT-Xent | SupCon 充分利用 batch 内所有同类正对, 收敛更稳; 不需要专门挖 hard negatives |
| Aligner 用 multi-head cross-attn 而非 KNN 软匹配 | KNN-soft, GNN | cross-attn 可端到端学习, 多头并行表达多种相似度模式; 训练时 split 自监督即可生效 |
| Fusion 用 softmax(rewards) 而非 max | max, average | softmax 在温度调参下覆盖了 max (T→0) 与 average (T→∞) 的两端, 灵活 |
| 标注 Scheme1 用 soft-argmax 而非 MLP-regress | 直接 MLP | MLP 容易塌缩到 数据集均值;heatmap+soft-argmax 是凸组合, 必须 heatmap 均匀才塌, 监督立刻修复 |
| 框损失用 CIoU 主导而非 SmoothL1 主导 | SmoothL1+GIoU | SmoothL1 在 [0,1] 坐标上的梯度~1e-3, 被 GIoU 淹没;CIoU 把 IoU + 中心距离 + 长宽比合一, 权 5.0 后成为主信号; 加 size-reg 兜底反塌缩 |
| 记忆存储用 FAISS + SQLite 双路 | 单纯 FAISS / 单纯 SQLite | 向量检索快, 结构化字段过滤靠 SQLite 索引; 双路通过 id 关联, 互补 |
| 图像记忆按 (object, affordance) 复合键 | 单一 affordance / FAISS only | 离散键检索零距离误差; FAISS 仅作 soft fallback |
| Async memory formation | 同步 | 入库不阻塞推理; daemon thread + queue 保证顺序与失败重试边界 |

---

## 附录 B — 维护索引 (本次新增/修改的文件)

| 文件 | 变更类型 | 摘要 |
|---|---|---|
| [memory_system/memory_store.py](memory_system/memory_store.py) | 修改 | 新增 `list_all` 分页 API |
| [memory_system/train_index_align.py](memory_system/train_index_align.py) | 新增 | Indexer + Aligner 联合训练脚本 |
| [backend.py](backend.py) | 修改 | poweron 加载 indexer/aligner.pt; 新增 7 个 admin API |
| [frontend/monitor/train.html](frontend/monitor/train.html) | 修改 | 新增"已入库记忆 — 浏览/删除/训练" 第二个 module 块 |
| [frontend/monitor/watch.html](frontend/monitor/watch.html) | 修改 | 新增反馈区 (raycaster + KNN + outcome 选择) |
| [annotation/annotation_model.py](annotation/annotation_model.py) | (此前会话修改) | Scheme1 重写为 spatial heatmap + FiLM + CIoU |

---

*文档结束*
