"""
train_index_align.py
====================

Standalone trainer for the two learnable components of the memory system:

* :class:`MemoryIndexer` — projection head producing 128-D L2-normalised
  index vectors from ARM features (or from the per-point feature mean
  when ARM features are unavailable).
* :class:`MemoryAligner` — multi-head cross-attention that transfers a
  historical preference matrix onto a new point cloud.

Data source
-----------
Both components are trained directly off the populated preference
memory store (the SQLite database written by the backend / push_to_memory
pipeline).  Each :class:`MemoryEntry` carries everything we need:

* ``index_vector`` — current frozen 128-D vector (used only as a sanity
  check; the *new* indexer recomputes its own from ``point_features``).
* ``point_features`` — per-point features  [N, D_feat]
* ``preference_matrix`` — per-point label   [N]
* ``affordance_label``  — categorical key used for contrastive pairs.

Losses
------
* **Indexer loss**: supervised contrastive (``MemoryIndexer.contrastive_loss``)
  over the affordance labels.  Pairs from the same affordance class are
  treated as positive; everything else as negative.  Because real ARM
  features were not stored per-entry, we approximate the ARM feature by
  prepending an extra "global" token whose value is the mean of the
  point features — this gives the indexer a [B, 1+N, D] sequence to pool.

* **Aligner loss**: a *self-supervised* MSE objective: for each entry
  in a batch, split its N points into two disjoint halves A / B.  The
  aligner must reconstruct ``Pref[A]`` from ``F[A], F[B], Pref[B]``.
  This forces the attention weights to learn meaningful feature → label
  correspondences while only using data that already lives in the store.
  We additionally add an inter-entry term inside each affordance class
  so the aligner learns to transfer between *different* objects of the
  same affordance.

Usage
-----
    python -m memory_system.train_index_align \
        --store_dir ./memory_store \
        --out_dir   ./memory_system/checkpoints \
        --epochs    50

The checkpoints saved are simple ``state_dict`` files:

    {out_dir}/indexer.pt
    {out_dir}/aligner.pt

They are picked up automatically by the backend's ``poweron`` if placed
next to the SQLite database, or by passing explicit paths via the
spec keys ``indexer_ckpt`` / ``aligner_ckpt``.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .memory_entry import MemoryEntry
from .memory_indexer import MemoryIndexer
from .memory_aligner import MemoryAligner
from .memory_store import MemoryStore


# ----------------------------------------------------------------------
# Dataset built from a populated preference memory store
# ----------------------------------------------------------------------

class PrefMemoryDataset(Dataset):
    """Iterates over every entry in a :class:`MemoryStore`.

    Each sample yields:
        (point_features [N, D],
         preference_matrix [N],
         affordance_id (int))
    """

    def __init__(self, store: MemoryStore, max_points: int = 64):
        self.store = store
        self.max_points = max_points

        # Read every id + minimal metadata, build label vocab.
        listing = store.list_all(page=1, per_page=10 ** 9)
        self.ids: List[str] = [e["id"] for e in listing["entries"]]
        labels = sorted({e["affordance_label"] for e in listing["entries"]
                         if e.get("affordance_label")})
        self.label_to_id: Dict[str, int] = {lab: i for i, lab in enumerate(labels)}
        if not self.ids:
            raise RuntimeError(f"Memory store at {store.store_dir} is empty")

        # Pre-cache entries to avoid hammering SQLite (data is small).
        self._cache: Dict[str, MemoryEntry] = {}
        for eid in self.ids:
            ent = store.get(eid)
            if ent is None:
                continue
            self._cache[eid] = ent
        self.ids = [i for i in self.ids if i in self._cache]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        ent = self._cache[self.ids[idx]]

        pf = np.asarray(ent.point_features, dtype=np.float32)
        pref = np.asarray(ent.preference_matrix, dtype=np.float32).flatten()

        # Reshape flattened feature blobs back to [N, D]
        if pf.ndim == 1:
            d_total = pf.size
            n = pref.size if pref.size > 0 else self.max_points
            if n > 0 and d_total % n == 0:
                pf = pf.reshape(n, d_total // n)
            else:
                # fall back: assume feat_dim = 512
                pf = pf.reshape(-1, 512)
        if pref.size != pf.shape[0]:
            n = min(pref.size, pf.shape[0])
            pf = pf[:n]
            pref = pref[:n]

        # Truncate / random-sample to max_points so a fixed-size batch is feasible
        if pf.shape[0] > self.max_points:
            sel = np.random.choice(pf.shape[0], self.max_points, replace=False)
            pf = pf[sel]
            pref = pref[sel]
        elif pf.shape[0] < self.max_points and pf.shape[0] > 0:
            # pad by repeating
            reps = (self.max_points + pf.shape[0] - 1) // pf.shape[0]
            pf = np.tile(pf, (reps, 1))[: self.max_points]
            pref = np.tile(pref, reps)[: self.max_points]

        label = self.label_to_id.get(ent.affordance_label, 0)
        return (
            torch.from_numpy(pf),
            torch.from_numpy(pref),
            torch.tensor(label, dtype=torch.long),
        )


# ----------------------------------------------------------------------
# Loss helpers
# ----------------------------------------------------------------------

def aligner_self_split_loss(
    aligner: MemoryAligner,
    feats: torch.Tensor,    # [B, N, D]
    prefs: torch.Tensor,    # [B, N]
) -> torch.Tensor:
    """Self-supervised split-reconstruction loss for the aligner.

    For every entry we mask out half the points and ask the aligner to
    predict their preference from the other half.
    """
    B, N, D = feats.shape
    if N < 4:
        return feats.new_zeros(())

    half = N // 2
    losses = []
    for b in range(B):
        idx = torch.randperm(N, device=feats.device)
        a_idx = idx[:half]
        b_idx = idx[half: 2 * half]
        F_a = feats[b, a_idx]
        F_b = feats[b, b_idx]
        P_b = prefs[b, b_idx]
        P_a_target = prefs[b, a_idx]
        P_a_pred = aligner(F_a, F_b, P_b)
        losses.append(F.mse_loss(P_a_pred, P_a_target))
    return torch.stack(losses).mean()


def aligner_cross_entry_loss(
    aligner: MemoryAligner,
    feats: torch.Tensor,    # [B, N, D]
    prefs: torch.Tensor,    # [B, N]
    labels: torch.Tensor,   # [B]
) -> torch.Tensor:
    """For each pair (i, j) with same affordance label, ask the aligner to
    predict entry i's preference from entry j's (feats, prefs).

    We use the *cosine similarity* between predicted and target preferences
    as the optimisation target (encourages shape, not magnitude, alignment).
    """
    B = feats.shape[0]
    same = labels.unsqueeze(1).eq(labels.unsqueeze(0))
    same.fill_diagonal_(False)
    if not same.any():
        return feats.new_zeros(())

    losses = []
    pairs = same.nonzero(as_tuple=False)
    # Subsample to bound cost
    if pairs.shape[0] > 8:
        sel = torch.randperm(pairs.shape[0])[:8]
        pairs = pairs[sel]

    for i, j in pairs.tolist():
        P_i_pred = aligner(feats[i], feats[j], prefs[j])
        P_i_true = prefs[i]
        # Cosine-similarity loss (clipped 1 - cos_sim ≥ 0)
        a = P_i_pred - P_i_pred.mean()
        b = P_i_true - P_i_true.mean()
        denom = (a.norm() * b.norm() + 1e-8)
        cos = (a * b).sum() / denom
        losses.append(1.0 - cos)
    return torch.stack(losses).mean()


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def train(
    store_dir: str,
    out_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    max_points: int = 64,
    feat_dim: int = 512,
    index_dim: int = 128,
    indexer_weight: float = 1.0,
    aligner_split_weight: float = 1.0,
    aligner_cross_weight: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
):
    """Train both MemoryIndexer and MemoryAligner on the contents of an
    existing preference memory store, and dump checkpoints into ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    store = MemoryStore(store_dir=store_dir, index_dim=index_dim)
    dataset = PrefMemoryDataset(store, max_points=max_points)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 entries to train.")

    print(f"[train_index_align] entries={len(dataset)} "
          f"affordance_classes={len(dataset.label_to_id)} device={device}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    indexer = MemoryIndexer(emb_dim=feat_dim, index_dim=index_dim).to(device)
    aligner = MemoryAligner(feat_dim=feat_dim).to(device)
    indexer.train()
    aligner.train()

    opt = torch.optim.Adam(
        list(indexer.parameters()) + list(aligner.parameters()),
        lr=lr,
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running = {"idx": 0.0, "aln_split": 0.0, "aln_cross": 0.0, "n": 0}
        for feats, prefs, labels in loader:
            feats = feats.to(device)            # [B, N, D]
            prefs = prefs.to(device)            # [B, N]
            labels = labels.to(device)          # [B]
            B = feats.shape[0]
            if B < 2:
                continue

            # ── Indexer ────────────────────────────────────────────────
            # Treat point_features as the "ARM-like" sequence:
            # prepend a global mean token to mimic [B, 1+N, D] shape.
            global_tok = feats.mean(dim=1, keepdim=True)  # [B, 1, D]
            arm_seq = torch.cat([global_tok, feats], dim=1)  # [B, 1+N, D]
            v_index = indexer(arm_seq)                       # [B, index_dim]

            # Two augmented views: dropout-style noise on features
            noise1 = torch.randn_like(arm_seq) * 0.02
            noise2 = torch.randn_like(arm_seq) * 0.02
            v1 = indexer(arm_seq + noise1)
            v2 = indexer(arm_seq + noise2)
            loss_idx = MemoryIndexer.contrastive_loss(v1, v2, labels)

            # ── Aligner ───────────────────────────────────────────────
            loss_split = aligner_self_split_loss(aligner, feats, prefs)
            loss_cross = aligner_cross_entry_loss(aligner, feats, prefs, labels)

            loss = (indexer_weight * loss_idx
                    + aligner_split_weight * loss_split
                    + aligner_cross_weight * loss_cross)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(indexer.parameters()) + list(aligner.parameters()), 5.0
            )
            opt.step()

            running["idx"] += float(loss_idx) * B
            running["aln_split"] += float(loss_split) * B
            running["aln_cross"] += float(loss_cross) * B
            running["n"] += B

        n = max(running["n"], 1)
        dt = time.time() - t0
        print(
            f"[epoch {epoch:3d}/{epochs}] "
            f"idx={running['idx']/n:.4f} "
            f"aln_split={running['aln_split']/n:.4f} "
            f"aln_cross={running['aln_cross']/n:.4f} "
            f"({dt:.1f}s)"
        )

    # ── Save ────────────────────────────────────────────────────────────
    idx_path = os.path.join(out_dir, "indexer.pt")
    aln_path = os.path.join(out_dir, "aligner.pt")
    torch.save(indexer.state_dict(), idx_path)
    torch.save(aligner.state_dict(), aln_path)
    print(f"[train_index_align] saved -> {idx_path}")
    print(f"[train_index_align] saved -> {aln_path}")
    return idx_path, aln_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MemoryIndexer + MemoryAligner")
    p.add_argument("--store_dir", required=True,
                   help="Directory containing memories.db (the pref memory store).")
    p.add_argument("--out_dir", default="./memory_system/checkpoints",
                   help="Where to save indexer.pt and aligner.pt.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_points", type=int, default=64)
    p.add_argument("--feat_dim", type=int, default=512)
    p.add_argument("--index_dim", type=int, default=128)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    train(
        store_dir=args.store_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_points=args.max_points,
        feat_dim=args.feat_dim,
        index_dim=args.index_dim,
    )
