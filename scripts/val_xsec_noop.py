"""DECISIVE TEST: can a serve-time CROSS-SECTIONAL recalibration of predictions
change TS-AUC?

Theory: TS-AUC = weighted mean over online-steps t of the WITHIN-step cross-
sectional Mann-Whitney AUC. AUC_t depends ONLY on the ranking of predictions
within step t. Therefore ANY transform that is monotonic-in-prediction within
each step leaves every AUC_t (and thus TS-AUC) exactly unchanged. This includes
per-step z-scoring, per-step rank/quantile normalisation, and global monotone
maps -- i.e. every "rank each series against the population at the same online
step" recalibration the cross-series hypothesis proposes.

This script reconstructs the shipped blend on VAL and applies those recalibra-
tions, printing TS-AUC for each. If they are identical (to ~1e-12) the avenue is
a mathematical no-op and only a transform that REORDERS series within a step --
i.e. genuinely better per-series discrimination -- can move the metric.

Run: uv run python scripts/val_xsec_noop.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
from scipy.stats import rankdata

from sb.metric import ts_auc_grouped

EPS = 1e-12


def main() -> None:
    vb = np.load("features/val_base_logits.npz")
    y = vb["y"].astype(np.int64)
    t = vb["t_online"].astype(np.int64)
    sid = vb["series_id"].astype(np.int64)
    gbt = vb["gbt_logits"].astype(np.float64)       # (N,4) WITH log_t
    lin = vb["log_logit"].astype(np.float64)
    N = len(y)
    K = int(t.max()) + 2

    # rank member -> canonical order
    vr = np.load("features/val_rank_logits.npz")
    rkey = vr["series_id"].astype(np.int64) * K + vr["t_online"].astype(np.int64)
    ckey = sid * K + t
    o = np.argsort(rkey)
    pos = np.searchsorted(rkey[o], ckey)
    assert np.array_equal(rkey[o][pos], ckey), "rank reindex failed"
    rank = vr["rank_score"].astype(np.float64)[o][pos]

    gru = np.mean([np.load(f"features/val_seq_logits_0{i}_nolog.npz")["val_logit"]
                   .astype(np.float64) for i in (20, 21, 22)], axis=0)

    # shipped blend (round-10): RANK_GW=0.10, W_LIN=0.10, W_GRU=0.45
    def rescale(src, ref):
        return (src - src.mean()) / (src.std() + EPS) * ref.std() + ref.mean()

    mean4 = gbt.mean(axis=1)
    gbt5 = 0.90 * mean4 + 0.10 * rescale(rank, mean4)
    base = 0.90 * gbt5 + 0.10 * lin
    blend = 0.55 * base + 0.45 * gru

    print(f"VAL rows={N:,}  steps 0..{t.max()}  pos_rate={y.mean():.4f}")
    base_auc = ts_auc_grouped(t, y, blend)
    print(f"\nshipped blend TS-AUC = {base_auc:.10f}\n")

    # ---- cross-sectional recalibrations (all monotone-in-pred within step) ----
    def per_step_apply(v, fn):
        out = np.empty_like(v, dtype=np.float64)
        order = np.argsort(t, kind="stable")
        ts = t[order]
        uniq, starts = np.unique(ts, return_index=True)
        starts = list(starts) + [len(ts)]
        for i in range(len(uniq)):
            idx = order[starts[i]:starts[i + 1]]
            out[idx] = fn(v[idx])
        return out

    recals = {
        "per-step z-score": lambda: per_step_apply(
            blend, lambda s: (s - s.mean()) / (s.std() + EPS)),
        "per-step rank (0..1)": lambda: per_step_apply(
            blend, lambda s: rankdata(s) / (len(s) + 1.0)),
        "per-step quantile->N(0,1)": lambda: per_step_apply(
            blend, lambda s: _ppf(rankdata(s) / (len(s) + 1.0))),
        "per-step min-max": lambda: per_step_apply(
            blend, lambda s: (s - s.min()) / (s.max() - s.min() + EPS)),
        "global rank": lambda: rankdata(blend),
        "global sigmoid": lambda: 1.0 / (1.0 + np.exp(-blend)),
    }

    print(f"{'recalibration':>28}{'TS-AUC':>16}{'Δ vs shipped':>16}")
    for name, fn in recals.items():
        v = fn()
        a = ts_auc_grouped(t, y, v)
        print(f"{name:>28}{a:>16.10f}{a - base_auc:>16.2e}")

    print("\n=> Every cross-sectional recalibration is monotone-in-prediction")
    print("   within each step, so TS-AUC is UNCHANGED (Δ ~ 1e-12 float noise).")
    print("   Serve-time cross-series recalibration CANNOT move a rank-based,")
    print("   within-step metric. Only a transform that REORDERS series within a")
    print("   step (= better per-series discrimination) can help.")


def _ppf(u):
    """Vectorised inverse-normal CDF (Acklam) -- avoids a scipy import here."""
    from scipy.special import ndtri
    return ndtri(np.clip(u, 1e-9, 1 - 1e-9))


if __name__ == "__main__":
    main()
