This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:
1. Primary Request and Intent:
   The conversation contains two stacked user requests:
   
   **Earlier request (completed in this session)**: "我希望你可以优化一下记忆管理：包括点云记忆和图像记忆。需要在前端的管理员部分可以查看内部的数据，包括点云+热力图和图片+标注的渲染。同时可以删除某些数据。请你注意：memory_system下的indexer和aligner是可学习的，给出训练代码和加载这两个模型的代码，放入后端的backend.py。需要你注意的还有：watch.html里面需要有一个类似机器人前端里的点云标注和反馈部分。"
   
   Decomposed into 5 sub-tasks (all now completed):
   (a) Optimize memory management (point-cloud + image)
   (b) Admin frontend can browse internal data, render point-cloud+heatmap and image+annotations
   (c) Admin can delete data
   (d) Provide training code for the learnable MemoryIndexer and MemoryAligner; wire checkpoint loading into backend.py
   (e) Add a point-cloud annotation + feedback section to watch.html mirroring robot.html
   
   **Most recent request (active)**: "总结所有工作，生成一份超详尽的技术文档。包括完整的软件架构，记忆系统的架构，模型原理，标注原理"
   ("Summarize all work, generate an extremely detailed technical document. Include the complete software architecture, memory system architecture, model principles, and annotation principles.")

2. Key Technical Concepts:
   - **MemoryIndexer** (nn.Module): pools ARM feature [B, N_p+N_i, C=512] → MLP(512→256→128) → L2-normalised index; static `contrastive_loss(v1, v2, labels, temperature=0.07)` for supervised affordance-label contrastive learning
   - **MemoryAligner** (nn.Module): multi-head cross-attention (`num_heads=4`) over per-point features; learnable shared linear projection (`use_learned_projection=True`); forward signature: `(F_curr [N_curr,D], F_hist [N_hist,D], Pref_hist [N_hist]) → Pref_curr [N_curr]`
   - **MemoryStore**: FAISS IndexFlatIP(128-dim) + SQLite `memories` table; LRU eviction; methods add/get/search/remove/count/clear
   - **ImageMemoryStore**: SQLite `image_memories` table with composite key (object_category, affordance_label); F_i/F_s/F_e BLOBs; raw images as .npy under store_dir/images/
   - **MemoryRetriever**: post-filters by affordance/outcome/min_reward; over-retrieves 3x then filters
   - **MemoryFusion**: stateless; softmax-weighted fusion `weights = softmax(rewards * temperature)`, supports time decay (exp(-age/3600)), confidence weighting, negative suppression; `apply_to_output(raw, pref, alpha=0.3)` adds residual before sigmoid
   - **MemoryManager**: orchestrator; async formation via thread + queue; `form_memory`, `retrieve_and_fuse`, `enhance_prediction`
   - **MemoryEntry** dataclass: index_vector [D], point_cloud [N,3], point_features [N,D_feat], preference_matrix [N], reward, outcome, action_parameters, timestamps; base64-encoded numpy arrays in JSON BLOB
   - **Backend WebSocket API pattern**: `async def api_*(payload, conn_id)` registered in `API_TABLE` dict; admin auth via `_require_admin(payload.get("uuid", ""))`; `_ok`/`_err` response wrappers
   - **THREE.js raycaster** for point selection (raycaster.params.Points.threshold = 0.02)
   - **shared.js**: monitor frontend module exposing connect/call/on/makeScene/renderCloud/prefToColor/THREE/OrbitControls
   - **Affordance system**: 17 affordance labels, 23 object categories from `annotation/config_annotation.yaml`
   - **AnnotationModelScheme1** (completed earlier): rewritten with spatial heatmap + FiLM + CIoU loss to fix box-collapse

3. Files and Code Sections:

   - **memory_system/memory_store.py** (modified)
     - Added pagination support to mirror ImageMemoryStore
     - New methods: `_db_list_all(page, per_page)` and public `list_all(page, per_page)` returning `{entries, total, page, per_page, total_pages}` with metadata-only rows (id, object_category, affordance_label, outcome, reward, confidence, timestamp, access_count)

   - **memory_system/train_index_align.py** (NEW)
     - Standalone trainer reading directly from a populated MemoryStore SQLite DB
     - `PrefMemoryDataset(Dataset)` class — preloads all entries, builds affordance label vocab, truncates/pads point clouds to `max_points=64`
     - Losses:
       - `MemoryIndexer.contrastive_loss(v1, v2, labels)` with two noise-augmented views (σ=0.02)
       - `aligner_self_split_loss` — split points into halves, predict half A's preference from half B
       - `aligner_cross_entry_loss` — for same-affordance pairs (i,j), predict prefs[i] from (feats[j], prefs[j]) with cosine-similarity loss
     - `train()` function with Adam + gradient clipping (max norm 5.0)
     - Saves `{out_dir}/indexer.pt` and `{out_dir}/aligner.pt`
     - CLI flags: `--store_dir`, `--out_dir`, `--epochs`, `--batch_size`, `--lr`, `--max_points`, `--feat_dim`, `--index_dim`

   - **backend.py** (modified — added 7 new API handlers + checkpoint loading)
     - **poweron checkpoint loading** (after `STATE.pref_memory = MemoryManager(...)`):
       ```python
       def _resolve_ckpt(spec_key, default_name):
           p = spec.get(spec_key, "") if p and os.path.exists(p): return p
           cand = os.path.join(pref_dir, default_name)
           return cand if os.path.exists(cand) else None
       
       idx_ckpt = _resolve_ckpt("indexer_ckpt", "indexer.pt")
       if idx_ckpt:
           sd = torch.load(idx_ckpt, map_location="cpu", weights_only=False)
           if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
           STATE.pref_memory.indexer.load_state_dict(
               {k.replace("module.", ""): v for k, v in sd.items()},
               strict=False)
       # ...same for aligner.pt
       ```
     - **New API handlers**: `api_list_pref_memory`, `api_get_pref_memory_entry` (returns point_cloud_b64, preference_b64, scene_image), `api_delete_pref_memory`, `api_list_image_memory`, `api_get_image_memory_entry` (loads .npy image, encodes to base64 PNG, decodes sub_box/obj_box), `api_delete_image_memory`, `api_train_index_align`
     - `api_train_index_align` launches `python -m memory_system.train_index_align` subprocess, streams stdout via `train_log` WebSocket events, sends `train_done` on completion
     - All registered in API_TABLE

   - **frontend/monitor/train.html** (modified — added second `<script type="module">` block)
     - New "已入库记忆 — 浏览 / 删除 / 训练" section with toolbar buttons mb-refresh-pref, mb-refresh-img, mb-train, and epochs/batch/lr inputs
     - Two flex columns: list + pagination on left, preview (canvas) on right, for both pref and image memory
     - `b64ToFloat32` helper decodes server-side base64 → Float32Array
     - `showPrefDetail(entryId)` calls `get_pref_memory_entry`, decodes point_cloud_b64 and preference_b64, renders via `renderCloud(scene, ptsArr, pref, oldObj, size=4)`
     - `showImgDetail(entryId)` calls `get_image_memory_entry`, draws image on canvas with sub_box (blue #1f6feb) and obj_box (red #a02525); auto-detects normalised vs pixel coords
     - `adminWatcher` polls every 500ms to unlock buttons when admin auth completes
     - Train log streaming via `on('train_log')` filtered by `stream === 'index_align'`

   - **frontend/monitor/watch.html** (modified)
     - Added feedback row (mirror of robot.html lines 119-129):
       ```html
       <div class="row" style="margin-top:6px">
         <label>反馈</label>
         <span class="status">邻域 n</span>
         <input id="neighbor-n" type="number" value="0" min="0" max="500"/>
         <select id="feedback-outcome">
           <option value="优秀">优秀</option><option value="成功">成功</option><option value="失败">失败</option>
         </select>
         <button id="btn-clear-sel">清空选区</button>
         <button id="feedback" class="primary" disabled>发送反馈</button>
         <span id="feedback-status" class="status">未追踪机器人</span>
       </div>
       ```
     - Added THREE.Raycaster with `params.Points.threshold = 0.02` for canvas-pred
     - `pickNeighbors(centerIdx, n)` — euclidean K-NN over predPositions
     - `highlightSelection()` — recolours pref then paints selected indices yellow (1.0, 1.0, 0.0)
     - Feedback handler:
       ```javascript
       const pref = Array.from(predPreference);
       for (const i of selectedPts) pref[i] = 1.0;
       const outcome = document.getElementById('feedback-outcome').value;
       await call('feedback', { robot_id: activeRobot, preference: pref, outcome });
       ```
     - Updated `onRobotUpdate` to capture `predPositions` and `predPreference` from `renderCloud` output, reset `selectedPts`, enable feedback button when `activeRobot` is set
     - Imported `THREE` from shared.js

   - **memory_system/memory_indexer.py** (read only)
     - `MemoryIndexer(nn.Module)` with proj head (Linear→BN→ReLU→Linear), L2-normalised output
     - Static `contrastive_loss(v1, v2, labels, temperature=0.07)` — same-affordance pairs are positives, masked diagonal excluded

   - **memory_system/memory_aligner.py** (read only)
     - Multi-head attention: `Q = F_curr.view(N, H, D/H).transpose(0,1)`, `attn = softmax(Q@K^T * scale / temp)`, `Pref_curr = mean over heads of A @ Pref_hist`
     - Optional shared projection: Linear→LayerNorm→ReLU

   - **memory_system/memory_fusion.py** (read only)
     - `fuse_numpy`: scores = rewards [* confidence] [+ λ * exp(-age/3600)]; weights = softmax(scores * temperature); weighted sum
     - `apply_to_output(raw, pref, alpha)`: `final = sigmoid(raw + alpha * pref)`

   - **memory_system/memory_retriever.py** (read only)
     - Over-retrieves 3x, filters by affordance/outcome/min_reward/similarity_threshold

4. Errors and fixes:
   - **Duplicate closing tags in train.html**: After injecting the new "已入库记忆" section, I accidentally left a duplicate `</body></html>` mid-file at line 475-476. Fixed by removing the duplicate set so only the final pair at end of file remained. Verified with grep showing two `<script type="module">` blocks and single closing tags at lines 750-751.
   - **No other errors encountered**; Python syntax check (`python -c "import ast; ast.parse(...)"`) passed for