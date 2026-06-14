"""Pick a robust W_GRU for an averaged neural member of ARBITRARY members.

Generalises scripts/tune_wgru.py: pass the per-member VAL logit npz files as
args; they are averaged (= the submission config, which averages the recurrent
step-logits with equal weight) and the base+AVG blend TS-AUC is printed across a
fine W grid with honest 2-fold halves, so we ship a weight in a flat/robust
region, not a single-split spike.

Run: uv run python scripts/tune_wgru2.py features/val_seq_logits*.npz
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
from scipy.stats import rankdata, pearsonr

from sb.metric import ts_auc_grouped


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print("usage: tune_wgru2.py <seq_logit.npz> [<seq_logit.npz> ...]")
        return

    d = np.load("features/val_base_logits.npz")
    yv, tv, sv = d["y"], d["t_online"], d["series_id"]
    base = d["base_logit"]

    members = []
    for p in paths:
        v = np.load(p)["val_logit"]
        members.append(v)
        auc = ts_auc_grouped(tv, yv, v)
        r = pearsonr(rankdata(base), rankdata(v))[0]
        print(f"  {p}: standalone {auc:.4f}  rank-corr vs base {r:.3f}")
    mem = np.mean(members, axis=0)

    uniq = np.unique(sv)
    rng = np.random.default_rng(1)
    rng.shuffle(uniq)
    A = np.isin(sv, list(set(uniq[: len(uniq) // 2].tolist())))

    rmem = pearsonr(rankdata(base), rankdata(mem))[0]
    print(f"\nBASE = {ts_auc_grouped(tv, yv, base):.4f}   "
          f"AVG member standalone = {ts_auc_grouped(tv, yv, mem):.4f}  "
          f"rank-corr vs base = {rmem:.3f}  ({len(members)} members)")
    print(f"{'W_GRU':>6} {'full':>8} {'halfA':>8} {'halfB':>8} {'min':>8}")
    best = (0.0, -1.0)
    for w in np.arange(0.15, 0.61, 0.05):
        bl = (1 - w) * base + w * mem
        full = ts_auc_grouped(tv, yv, bl)
        a = ts_auc_grouped(tv[A], yv[A], bl[A])
        b = ts_auc_grouped(tv[~A], yv[~A], bl[~A])
        print(f"{w:6.2f} {full:8.4f} {a:8.4f} {b:8.4f} {min(a, b):8.4f}")
        if min(a, b) > best[1]:
            best = (float(w), min(a, b))
    print(f"\nmost robust (max min-half) W_GRU = {best[0]:.2f}  "
          f"(min-half {best[1]:.4f})")


if __name__ == "__main__":
    main()
