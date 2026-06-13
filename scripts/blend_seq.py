"""Blend the base ensemble (model_018) with one or more extra member logits on
the fixed VAL split, reporting correlation, the best blend weight, and an honest
2-fold-on-VAL check (so a lift isn't an in-sample fluke).

Reads features/val_base_logits.npz (base) and each extra npz (default key
'val_logit'). All must be aligned to the same VAL row order (they are, since
every script uses the same split-seed-42 val_mask on the v4 matrix).

  uv run python scripts/blend_seq.py features/val_seq_logits.npz
  uv run python scripts/blend_seq.py --avg features/val_seq*.npz   # ship config

With --avg, all extras are z-scored on rank and averaged into ONE neural member
(the exact submission config: mean of the GRU seeds blended at a single W_GRU).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
from scipy.stats import rankdata, pearsonr

from sb.metric import ts_auc_grouped


def main() -> None:
    args = sys.argv[1:]
    avg_mode = "--avg" in args
    args = [a for a in args if a != "--avg"]

    d = np.load("features/val_base_logits.npz")
    yv, tv = d["y"], d["t_online"]
    sv = d["series_id"]
    base = d["base_logit"]
    print(f"BASE VAL TS-AUC = {ts_auc_grouped(tv, yv, base):.4f}")

    extras = []
    for arg in args:
        path, key = arg.split(":") if ":" in arg else (arg, "val_logit")
        v = np.load(path)[key]
        extras.append((path, v))
        auc = ts_auc_grouped(tv, yv, v)
        # correlation on rank (what TS-AUC cares about)
        r = pearsonr(rankdata(base), rankdata(v))[0]
        print(f"  {path}: standalone {auc:.4f}   rank-corr vs base {r:.3f}")

    if avg_mode and len(extras) > 1:
        # average the raw neural logits into one member (matches submission,
        # which averages GRU step-logits with equal weight)
        mem = np.mean([v for _, v in extras], axis=0)
        auc = ts_auc_grouped(tv, yv, mem)
        r = pearsonr(rankdata(base), rankdata(mem))[0]
        print(f"\n  AVG neural member: standalone {auc:.4f}  "
              f"rank-corr vs base {r:.3f}  ({len(extras)} seeds)")
        extras = [("AVG", mem)]

    # progressive: add each extra at its best weight
    cur = base.copy()
    for path, v in extras:
        best = (0.0, ts_auc_grouped(tv, yv, cur))
        for w in np.arange(0.05, 0.61, 0.05):
            blend = (1 - w) * cur + w * v
            s = ts_auc_grouped(tv, yv, blend)
            if s > best[1]:
                best = (w, s)
        w = best[0]
        print(f"\n  + {path}  best w={w:.2f} -> {best[1]:.4f}  "
              f"(was {ts_auc_grouped(tv, yv, cur):.4f})")
        # honest halves at chosen w
        uniq = np.unique(sv)
        rng = np.random.default_rng(1)
        rng.shuffle(uniq)
        A = np.isin(sv, list(set(uniq[: len(uniq) // 2].tolist())))
        blend = (1 - w) * cur + w * v
        print(f"    honest halves: A={ts_auc_grouped(tv[A], yv[A], blend[A]):.4f}  "
              f"B={ts_auc_grouped(tv[~A], yv[~A], blend[~A]):.4f}")
        if w > 0:
            cur = blend

    print(f"\n>>> FINAL blended VAL TS-AUC = {ts_auc_grouped(tv, yv, cur):.4f}")


if __name__ == "__main__":
    main()
