"""Diagnose where a logit vector wins/loses on the VAL split, by elapsed break
age and by online step bucket. Reuses features/val_base_logits.npz. Pass an
extra npz with a 'val_logit' array (aligned to the same VAL rows) to compare.

Run: uv run python scripts/diag_cohorts.py [extra_logits.npz:key]
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
from sb.metric import ts_auc_grouped


def per_series_tau(y, sid, t):
    """tau (online step of break) per row; -1 where the series has no break."""
    order = np.lexsort((t, sid))
    tau_row = np.full(len(y), -1, dtype=np.int64)
    s_s, y_s, t_s = sid[order], y[order], t[order]
    i, n = 0, len(order)
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    tau_sorted = np.full(n, -1, dtype=np.int64)
    while i < n:
        j = i
        while j < n and s_s[j] == s_s[i]:
            j += 1
        seg_y, seg_t = y_s[i:j], t_s[i:j]
        if seg_y.any():
            tau_sorted[i:j] = seg_t[seg_y.argmax()]
        i = j
    return tau_sorted[inv]


def report(name, tv, yv, logit, tau, elapsed):
    print(f"\n=== {name}: overall TS-AUC = {ts_auc_grouped(tv, yv, logit):.4f} ===")
    print("  by elapsed break age (positives only contribute; neg=unbroken):")
    for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 10**9)]:
        # keep negatives (elapsed=-1) plus positives whose age in [lo,hi)
        keep = (elapsed < 0) | ((elapsed >= lo) & (elapsed < hi))
        if keep.sum() == 0:
            continue
        a = ts_auc_grouped(tv[keep], yv[keep], logit[keep])
        npos = int(((elapsed >= lo) & (elapsed < hi)).sum())
        print(f"    age [{lo:4d},{hi:<5}) : TS-AUC {a:.4f}   (n_pos={npos:,})")
    print("  by online step bucket:")
    for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10**9)]:
        keep = (tv >= lo) & (tv < hi)
        if keep.sum() == 0:
            continue
        a = ts_auc_grouped(tv[keep], yv[keep], logit[keep])
        print(f"    step [{lo:4d},{hi:<5}) : TS-AUC {a:.4f}   (rows={int(keep.sum()):,})")


def main() -> None:
    d = np.load("features/val_base_logits.npz")
    yv, tv, sv = d["y"], d["t_online"], d["series_id"]
    base = d["base_logit"]
    tau = per_series_tau(yv, sv, tv)
    elapsed = np.where(tau >= 0, tv - tau, -1)  # -1 for unbroken rows
    report("BASE (model_018)", tv, yv, base, tau, elapsed)

    for arg in sys.argv[1:]:
        path, key = arg.split(":") if ":" in arg else (arg, "val_logit")
        extra = np.load(path)[key]
        report(f"EXTRA {path}", tv, yv, extra, tau, elapsed)


if __name__ == "__main__":
    main()
