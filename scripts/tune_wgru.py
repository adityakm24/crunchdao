"""Pick a robust W_GRU for the averaged GRU neural member.

Prints the base+AVG blend TS-AUC across a fine W_GRU grid with honest 2-fold
halves, so we ship a weight that's robust (flat region), not the single VAL-
optimal spike. AVG member = mean of the per-seed VAL logits (= submission config,
which averages the GRU step-logits with equal weight).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from sb.metric import ts_auc_grouped

d = np.load("features/val_base_logits.npz")
yv, tv, sv = d["y"], d["t_online"], d["series_id"]
base = d["base_logit"]

seeds = [np.load(p)["val_logit"] for p in (
    "features/val_seq_logits.npz",
    "features/val_seq_logits_s1.npz",
    "features/val_seq_logits_s2.npz",
)]
mem = np.mean(seeds, axis=0)

uniq = np.unique(sv)
rng = np.random.default_rng(1)
rng.shuffle(uniq)
A = np.isin(sv, list(set(uniq[: len(uniq) // 2].tolist())))

print(f"BASE = {ts_auc_grouped(tv, yv, base):.4f}   "
      f"AVG member standalone = {ts_auc_grouped(tv, yv, mem):.4f}")
print(f"{'W_GRU':>6} {'full':>8} {'halfA':>8} {'halfB':>8} {'min':>8}")
for w in np.arange(0.15, 0.56, 0.05):
    bl = (1 - w) * base + w * mem
    full = ts_auc_grouped(tv, yv, bl)
    a = ts_auc_grouped(tv[A], yv[A], bl[A])
    b = ts_auc_grouped(tv[~A], yv[~A], bl[~A])
    print(f"{w:6.2f} {full:8.4f} {a:8.4f} {b:8.4f} {min(a, b):8.4f}")
