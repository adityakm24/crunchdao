"""Round 10 — DRASTIC-signal probe: do SPECTRAL (FFT) + ENTROPY/COMPLEXITY
features add cross-sectional ordering information the shipped blend does NOT
already have?

Motivation: cached-member re-blending is confirmed at ceiling (~0.6170 VAL).
A drastic jump needs a genuinely NEW per-series signal. The existing 162-feature
space is comprehensive in moments / distribution (KS, ECDF) / autocorrelation /
changepoint (CUSUM, Page-Hinkley, chi2) — but has ZERO spectral and ZERO
entropy/complexity content. Those are orthogonal to everything present, fully
serve-compatible (one series at a time) and pure-numpy (np.fft + argsort), so
they could in principle add. This probe decides it on VAL before any retrain.

Method (leak-free, mirrors production):
  1. Reconstruct the VAL split (seed 42, 20% = 2000 held-out train series).
  2. Reconstruct the exact shipped FINAL blend logit at every VAL row from the
     cached members (4 GBT + logistic + nolog rank + 3 nolog GRU) with the
     round-10 weights (RANK_GW=0.10, W_LIN=0.10, W_GRU=0.45).
  3. Stream the raw VAL series; at a grid of online steps compute candidate NEW
     features as a HISTORY-vs-ONLINE contrast (causal: only data up to step t).
  4. Standalone TS-AUC of each new feature (is there ANY signal?).
  5. Honest incremental test: z-score [shipped_logit, new_feats]; fit a logistic
     on series-disjoint half A, evaluate TS-AUC on half B (and reverse). Does the
     augmented model beat shipped-logit-alone on the held-out half?

A variance/power log-ratio is included as a POSITIVE CONTROL: it overlaps an
existing feature so it MUST show signal; if it does and the spectral/entropy
ones do not, the harness is sound and spectral/entropy are genuinely dead.

Run: uv run python scripts/val_spectral_probe.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "submission")

import numpy as np

from sb.metric import ts_auc_grouped
from sb.data import iter_series

SPLIT_SEED = 42
STEPS = np.array([30, 45, 60, 80, 100, 120, 140, 165, 190, 220, 250, 290, 340, 400, 470, 550],
                 dtype=np.int64)
WMAX = 128  # spectral/entropy window cap
EPS = 1e-9

RANK_GW = 0.10
W_LIN = 0.10
W_GRU = 0.45

FEAT_NAMES = [
    "spec_ent_onl",        # spectral entropy of the online window
    "spec_ent_contrast",   # spec_ent(online) - spec_ent(hist)
    "centroid_shift",      # spectral centroid(online) - centroid(hist)
    "lowband_logratio",    # log low-freq power online/hist (slow-drift / mean shift)
    "dompeak_logratio",    # log dominant-bin power online/hist (periodicity strength)
    "power_logratio[CTRL]",# log total power online/hist == variance ratio (CONTROL)
    "perm_ent_contrast",   # permutation entropy(online) - perm_ent(hist), order 3
]


def perm_ent(x, m=3):
    if len(x) < m + 1:
        return 0.5
    w = np.lib.stride_tricks.sliding_window_view(x, m)
    order = np.argsort(w, axis=1)
    codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(6.0))


def _psd(x):
    x = x - x.mean()
    X = np.fft.rfft(x)
    P = X.real ** 2 + X.imag ** 2
    return P[1:]  # drop DC


def feats(hist, onl):
    W = min(WMAX, len(onl), len(hist))
    if W < 8:
        return None
    h = hist[-W:].astype(np.float64)
    o = onl[-W:].astype(np.float64)
    Ph = _psd(h); Po = _psd(o)
    sh = Ph.sum() + EPS; so = Po.sum() + EPS
    Phn = Ph / sh; Pon = Po / so
    L = len(Phn)
    ent = lambda P: float(-(P * np.log(P + 1e-12)).sum() / np.log(len(P)))
    se_h = ent(Phn); se_o = ent(Pon)
    fr = np.arange(L) / L
    cen_h = float((fr * Phn).sum()); cen_o = float((fr * Pon).sum())
    lb = max(1, L // 4)
    low_h = Phn[:lb].sum(); low_o = Pon[:lb].sum()
    dom_h = Phn.max(); dom_o = Pon.max()
    pe_h = perm_ent(h); pe_o = perm_ent(o)
    return np.array([
        se_o,
        se_o - se_h,
        cen_o - cen_h,
        np.log((low_o + 1e-9) / (low_h + 1e-9)),
        np.log((dom_o + 1e-9) / (dom_h + 1e-9)),
        np.log(so / sh),
        pe_o - pe_h,
    ], dtype=np.float64)


def zscore(a):
    return (a - a.mean()) / (a.std() + EPS)


def reconstruct_shipped_logit():
    """Final shipped blend logit at every VAL row, plus sid/t/y (cache order)."""
    vb = np.load("features/val_base_logits.npz")
    y = vb["y"].astype(np.int64)
    t = vb["t_online"].astype(np.int64)
    sid = vb["series_id"].astype(np.int64)
    gbt = vb["gbt_logits"].astype(np.float64)  # 4-col, WITH log_t
    lin = vb["log_logit"].astype(np.float64)
    K = int(t.max()) + 2
    vr = np.load("features/val_rank_logits.npz")
    rkey = vr["series_id"].astype(np.int64) * K + vr["t_online"].astype(np.int64)
    ckey = sid * K + t
    o = np.argsort(rkey)
    pos = np.searchsorted(rkey[o], ckey)
    rank = vr["rank_score"].astype(np.float64)[o][pos]
    gru = np.mean([np.load(f"features/val_seq_logits_0{i}_nolog.npz")["val_logit"]
                   for i in (20, 21, 22)], axis=0)
    mean4 = gbt.mean(axis=1)
    r = (rank - rank.mean()) / (rank.std() + EPS) * mean4.std() + mean4.mean()
    gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * r
    base = (1 - W_LIN) * gbt5 + W_LIN * lin
    final = (1 - W_GRU) * base + W_GRU * gru
    return sid, t, y, final, K


def main():
    t0 = time.time()
    sid, t, y, shipped, K = reconstruct_shipped_logit()
    print(f"VAL rows={len(y):,}  shipped TS-AUC(full)={ts_auc_grouped(t, y, shipped):.4f}")

    # the cache (val_base_logits) ALREADY contains only the 2000 held-out VAL
    # series, so every unique sid here is a VAL series — use all of them.
    val_ids = set(np.unique(sid).tolist())
    print(f"VAL series={len(val_ids):,}  probe steps={STEPS.tolist()}")

    # lookup (sid*K + t) -> row index in the cache arrays
    key = sid * K + t
    order = np.argsort(key)
    key_s = key[order]

    rows = []  # (sid, t, y, shipped_logit, feat[7])
    nser = 0
    for s in iter_series("train"):
        if s.series_id not in val_ids:
            continue
        nser += 1
        hist = s.x_hist
        onl = s.x_online
        n_on = len(onl)
        for tt in STEPS:
            if tt >= n_on:
                continue
            f = feats(hist, onl[: int(tt) + 1])
            if f is None:
                continue
            ck = s.series_id * K + int(tt)
            j = np.searchsorted(key_s, ck)
            if j >= len(key_s) or key_s[j] != ck:
                continue
            ri = order[j]
            rows.append((s.series_id, int(tt), int(y[ri]), float(shipped[ri]), f))
        if nser % 500 == 0:
            print(f"  ...{nser} series  ({time.time()-t0:.0f}s)")

    sid_p = np.array([r[0] for r in rows], dtype=np.int64)
    t_p = np.array([r[1] for r in rows], dtype=np.int64)
    y_p = np.array([r[2] for r in rows], dtype=np.int64)
    base_p = np.array([r[3] for r in rows], dtype=np.float64)
    F = np.vstack([r[4] for r in rows])
    print(f"\nprobe points={len(rows):,} from {nser} series  ({time.time()-t0:.0f}s)")

    base_auc = ts_auc_grouped(t_p, y_p, base_p)
    print(f"\nshipped logit TS-AUC on probe points = {base_auc:.4f}\n")

    print("standalone TS-AUC of each NEW feature (|dev| from 0.5 = signal):")
    print(f"  {'feature':>22}  {'TS-AUC':>7}  {'|dev|':>6}")
    for j, nm in enumerate(FEAT_NAMES):
        a = ts_auc_grouped(t_p, y_p, F[:, j])
        print(f"  {nm:>22}  {a:>7.4f}  {abs(a-0.5):>6.4f}")

    # ---- honest incremental test: logistic on half A -> eval half B (+ reverse)
    from sklearn.linear_model import LogisticRegression
    up = np.unique(sid_p)
    np.random.default_rng(1).shuffle(up)
    A = set(up[: len(up) // 2].tolist())
    mA = np.isin(sid_p, list(A)); mB = ~mA

    bz = zscore(base_p)
    Fz = np.column_stack([zscore(F[:, j]) for j in range(F.shape[1])])
    Xall = np.column_stack([bz, Fz])

    def fit_eval(mtr, mev):
        clf = LogisticRegression(C=0.3, max_iter=2000)
        clf.fit(Xall[mtr], y_p[mtr])
        aug = clf.decision_function(Xall[mev])
        base_ev = ts_auc_grouped(t_p[mev], y_p[mev], base_p[mev])
        aug_ev = ts_auc_grouped(t_p[mev], y_p[mev], aug)
        return base_ev, aug_ev, clf.coef_[0]

    print("\nhonest incremental test (z[shipped]+z[new feats], series-disjoint):")
    bA, aA, cA = fit_eval(mA, mB)  # fit A, eval B
    bB, aB, cB = fit_eval(mB, mA)  # fit B, eval A
    print(f"  fitA->evalB:  base={bA:.4f}  augmented={aA:.4f}  delta={aA-bA:+.4f}")
    print(f"  fitB->evalA:  base={bB:.4f}  augmented={aB:.4f}  delta={aB-bB:+.4f}")
    print(f"  min-half delta = {min(aA-bA, aB-bB):+.4f}")
    coef = (np.abs(cA[1:]) + np.abs(cB[1:])) / 2
    print("\n  mean |logistic coef| on new feats (vs shipped coef "
          f"{(abs(cA[0])+abs(cB[0]))/2:.3f}):")
    for nm, c in sorted(zip(FEAT_NAMES, coef), key=lambda kv: -kv[1]):
        print(f"    {nm:>22}  {c:.4f}")

    d = min(aA - bA, aB - bB)
    print("\n=== VERDICT ===")
    if d > 0.0015:
        print(f"NEW spectral/entropy signal ADDS (min-half +{d:.4f}) -> worth a full retrain")
    else:
        print(f"no robust new signal (min-half {d:+.4f}) -> per-series space confirmed saturated")
    print(f"(done in {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
