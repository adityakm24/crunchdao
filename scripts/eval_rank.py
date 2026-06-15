"""Evaluate the LambdaRank member (model_032_rank) on VAL: standalone TS-AUC vs
the pointwise GBT base and each individual GBT, decorrelation (rank-corr), and
the incremental blend gate onto the shipped base+GRU stack (0.6160) with honest
halves. Blends in z-space (each model output affine-normalised) since the rank
score and the logit live on different scales — this is the deployable form.

Run: uv run python scripts/eval_rank.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from sb.metric import ts_auc_grouped

GRU_FILES = ("val_seq_logits.npz", "val_seq_logits_s1.npz", "val_seq_logits_s2.npz")


def z(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def half_mask_A(sid_val):
    uniq = np.unique(sid_val)
    rng = np.random.default_rng(1)
    rng.shuffle(uniq)
    a = set(uniq[: len(uniq) // 2].tolist())
    return np.isin(sid_val, list(a))


def main():
    base = np.load("features/val_base_logits.npz")
    sv, tv = base["series_id"], base["t_online"]
    yv = base["y"].astype(np.int64)
    gbt = base["gbt_logits"]          # (N, 4)
    log_logit = base["log_logit"]
    base_logit = base["base_logit"]
    gru = np.mean([np.load("features/" + f)["val_logit"] for f in GRU_FILES], axis=0)
    shipped = 0.55 * base_logit + 0.45 * gru

    # ---- align rank scores to base-cache row order via (sid, t) lookup ----
    r = np.load("features/val_rank_logits.npz")
    rk = {(int(s), int(t)): float(v)
          for s, t, v in zip(r["series_id"], r["t_online"], r["rank_score"])}
    rank = np.array([rk.get((int(sv[i]), int(tv[i])), np.nan) for i in range(len(sv))])
    miss = int(np.isnan(rank).sum())
    if miss:
        print(f"WARNING: {miss} rows missing rank score (filled with median)")
        rank[np.isnan(rank)] = np.nanmedian(rank)
    print(f"VAL rows={len(sv):,}  rank scores aligned (miss={miss})\n")

    # ---- standalone comparisons ----
    ship_full = ts_auc_grouped(tv, yv, shipped)
    base_sa = ts_auc_grouped(tv, yv, base_logit)
    rank_sa = ts_auc_grouped(tv, yv, rank)
    print("STANDALONE VAL TS-AUC:")
    for j in range(gbt.shape[1]):
        print(f"  gbt[{j}]           {ts_auc_grouped(tv, yv, gbt[:, j]):.4f}")
    print(f"  logistic          {ts_auc_grouped(tv, yv, log_logit):.4f}")
    print(f"  pointwise base    {base_sa:.4f}")
    print(f"  >>> RANK          {rank_sa:.4f}   (vs base {base_sa:+.4f})")
    print(f"  shipped (b+GRU)   {ship_full:.4f}")
    print(f"  rank-corr(rank, base)    = {np.corrcoef(rank, base_logit)[0,1]:.3f}")
    print(f"  rank-corr(rank, shipped) = {np.corrcoef(rank, shipped)[0,1]:.3f}\n")

    # ---- TEST A: rank as top-level member onto shipped (z-space) ----
    halfA = half_mask_A(sv)
    zsh, zrk = z(shipped), z(rank)
    sa = ts_auc_grouped(tv[halfA], yv[halfA], shipped[halfA])
    sb = ts_auc_grouped(tv[~halfA], yv[~halfA], shipped[~halfA])
    print("TEST A — blend (1-w)*shipped + w*rank  (z-space, honest halves):")
    print(f"{'w':>6}{'full':>9}{'halfA':>9}{'halfB':>9}{'min':>9}")
    print(f"{'ship':>6}{ship_full:>9.4f}{sa:>9.4f}{sb:>9.4f}{min(sa,sb):>9.4f}")
    for w in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        bl = (1 - w) * zsh + w * zrk
        full = ts_auc_grouped(tv, yv, bl)
        aA = ts_auc_grouped(tv[halfA], yv[halfA], bl[halfA])
        aB = ts_auc_grouped(tv[~halfA], yv[~halfA], bl[~halfA])
        tag = "  <--" if full > ship_full and min(aA, aB) >= min(sa, sb) else ""
        print(f"{w:>6.2f}{full:>9.4f}{aA:>9.4f}{aB:>9.4f}{min(aA,aB):>9.4f}{tag}")

    # ---- TEST B: rank ADDED to the GBT base mean, then full pipeline ----
    # base = 0.8*mean(GBT) + 0.2*log; add rank as a 5th GBT (scaled to GBT logit
    # scale) -> base5 -> final = 0.55*base5 + 0.45*GRU.
    gbt_mean_std = gbt.mean(axis=1).std()
    rank_as_gbt = z(rank) * gbt_mean_std + gbt.mean()
    print("\nTEST B — add rank as a 5th base GBT member:")
    print(f"{'gw':>6}{'full':>9}{'halfA':>9}{'halfB':>9}{'min':>9}")
    for gw in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3):
        # weighted mean of 4 GBT + rank, weight gw on rank (split rest over 4)
        gbt5 = (1 - gw) * gbt.mean(axis=1) + gw * rank_as_gbt
        base5 = 0.8 * gbt5 + 0.2 * log_logit
        final = 0.55 * base5 + 0.45 * gru
        full = ts_auc_grouped(tv, yv, final)
        aA = ts_auc_grouped(tv[halfA], yv[halfA], final[halfA])
        aB = ts_auc_grouped(tv[~halfA], yv[~halfA], final[~halfA])
        tag = "  <--" if full > ship_full and min(aA, aB) >= min(sa, sb) else ""
        lbl = "base" if gw == 0.0 else f"{gw:.2f}"
        print(f"{lbl:>6}{full:>9.4f}{aA:>9.4f}{aB:>9.4f}{min(aA,aB):>9.4f}{tag}")


if __name__ == "__main__":
    main()
