"""Phase-0 free squeeze on the cached VAL logits (no training).

TS-AUC is a weighted average of per-step cross-sectional AUCs, and the per-step
AUC at step t depends only on scores at step t -> the optimal blend weight is
SEPARABLE per online step. A per-step-bucket W_GRU is serve-computable (we know
the online step index i at each point) and deterministic, so it is a legal,
free lever. We fit per-bucket weights on one honest half and verify they also
help the other half (so it isn't a single-split fluke). Compared against the
shipped flat W_GRU=0.40.

Run: uv run python scripts/phase0_squeeze.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from sb.metric import ts_auc_grouped

FLAT_W = 0.40
BUCKETS = [(0, 75), (75, 200), (200, 400), (400, 10**9)]
GRID = np.round(np.arange(0.0, 0.611, 0.025), 4)


def main() -> None:
    d = np.load("features/val_base_logits.npz")
    yv, tv, sv = d["y"], d["t_online"], d["series_id"]
    base = d["base_logit"]
    seeds = [np.load(p)["val_logit"] for p in (
        "features/val_seq_logits.npz",
        "features/val_seq_logits_s1.npz",
        "features/val_seq_logits_s2.npz",
    )]
    mem = np.mean(seeds, axis=0)

    print(f"BASE        VAL TS-AUC = {ts_auc_grouped(tv, yv, base):.4f}")
    flat = (1 - FLAT_W) * base + FLAT_W * mem
    print(f"flat W={FLAT_W} VAL TS-AUC = {ts_auc_grouped(tv, yv, flat):.4f}")

    # honest halves by series
    uniq = np.unique(sv)
    rng = np.random.default_rng(1)
    rng.shuffle(uniq)
    A = np.isin(sv, list(set(uniq[: len(uniq) // 2].tolist())))

    def bucket_mask(lo, hi):
        return (tv >= lo) & (tv < hi)

    def best_w_per_bucket(fit_mask):
        """Per-bucket optimal w on the rows in fit_mask (separable per step)."""
        ws = {}
        for lo, hi in BUCKETS:
            m = fit_mask & bucket_mask(lo, hi)
            best = (FLAT_W, -1.0)
            for w in GRID:
                bl = (1 - w) * base[m] + w * mem[m]
                s = ts_auc_grouped(tv[m], yv[m], bl)
                if s > best[1]:
                    best = (float(w), s)
            ws[(lo, hi)] = best[0]
        return ws

    def apply_w(ws):
        w_vec = np.full(len(tv), FLAT_W)
        for (lo, hi), w in ws.items():
            w_vec[bucket_mask(lo, hi)] = w
        return (1 - w_vec) * base + w_vec * mem

    # fit on A, verify on B (and vice-versa)
    for fit_name, fit_m, ev_name, ev_m in (("A", A, "B", ~A), ("B", ~A, "A", A)):
        ws = best_w_per_bucket(fit_m)
        bl = apply_w(ws)
        flat_ev = ts_auc_grouped(tv[ev_m], yv[ev_m], flat[ev_m])
        bkt_ev = ts_auc_grouped(tv[ev_m], yv[ev_m], bl[ev_m])
        print(f"\nfit {fit_name}: per-bucket w = "
              + ", ".join(f"[{lo},{hi if hi < 10**9 else 'inf'})={w:.3f}"
                          for (lo, hi), w in ws.items()))
        print(f"  eval {ev_name}: flat={flat_ev:.4f}  per-bucket={bkt_ev:.4f}  "
              f"delta={bkt_ev - flat_ev:+.4f}")

    # full-VAL per-bucket (in-sample upper bound, for reference only)
    ws_full = best_w_per_bucket(np.ones(len(tv), bool))
    full_bl = apply_w(ws_full)
    print(f"\nin-sample per-bucket (full VAL) = "
          f"{ts_auc_grouped(tv, yv, full_bl):.4f}  (upper bound, not shippable)")
    print("  weights:", {f"[{lo},{hi if hi<10**9 else 'inf'})": w
                          for (lo, hi), w in ws_full.items()})


if __name__ == "__main__":
    main()
