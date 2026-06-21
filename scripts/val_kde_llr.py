"""Round 11 probe — KDE likelihood-ratio feature (the ONE blueprint-named online
signal absent from our 162 features).

info.md Blueprint-1 names three online distributional signals: recursive CUSUM
(we have 14), online Wasserstein/eCDF distance (we have ecdf_l1 + windowed L1),
and a KDE-fitted density likelihood (we have ZERO). This probe builds the missing
class: fit a Gaussian KDE on the (z-scored) historical segment p_H, then score the
online stream by its excess negative log-likelihood under p_H. Unlike KS (sup-norm)
or eCDF-L1 (integral), the KDE-LLR is dominated by online points landing in the
LOW-DENSITY TAILS of the historical law -> sharper for location shifts that push
mass into the tail. Pure-numpy (logsumexp of Gaussians) => fully deployable.

Per-series normalization (baseline = self log-density of the historical sample)
uses ONLY that series' own history -> serve-compatible (no cross-series leak).

Gate (same as every member): standalone cross-sectional TS-AUC at dense steps,
best z-blend lift onto the round-10 shipped logit on VAL full + BOTH honest halves
(seed 1), and rank-correlation to shipped. PROMISING only if it lifts full AND both
halves with low rankcorr; else redundant -> the blueprint feature space is closed too.

Run: uv run python scripts/val_kde_llr.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from sb.data import iter_series        # noqa: E402
from sb.metric import ts_auc_grouped   # noqa: E402

SEED = 42
HALF_SEED = 1
STEP_GRID = list(range(40, 401, 6))
TRAIL = 128
MAX_ATOMS = 512
CLIP = 8.0
LOG2PI = float(np.log(2 * np.pi))
RANK_GW = 0.10
W_LIN = 0.10
W_GRU = 0.45


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def val_ids_split():
    import pandas as pd
    idx = pd.read_parquet("Dataset/y_train_index.parquet")
    uniq = np.array(sorted(idx.index.tolist()))
    np.random.default_rng(SEED).shuffle(uniq)
    val = uniq[: int(0.2 * len(uniq))]
    return np.array(sorted(int(v) for v in val))


def reconstruct_shipped():
    base = np.load("features/val_base_logits.npz")
    sid = base["series_id"].astype(np.int64)
    t = base["t_online"].astype(np.int64)
    y = base["y"].astype(np.int64)
    gbt = base["gbt_logits"].astype(np.float64)
    log_logit = base["log_logit"].astype(np.float64)
    mean4 = gbt.mean(axis=1)
    rk = np.load("features/val_rank_logits.npz")
    rk_key = {(int(s), int(tt)): float(v) for s, tt, v in
              zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank_score = np.array([rk_key[(int(s), int(tt))] for s, tt in zip(sid, t)])
    rank_as_gbt = (rank_score - rank_score.mean()) / (rank_score.std() + 1e-12)
    rank_as_gbt = rank_as_gbt * mean4.std() + mean4.mean()
    gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * rank_as_gbt
    base_logit = (1 - W_LIN) * gbt5 + W_LIN * log_logit
    gru = np.mean([np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"]
                   for m in ("020", "021", "022")], axis=0)
    shipped = (1 - W_GRU) * base_logit + W_GRU * gru
    lut = {(int(s), int(tt)): (float(sh), int(yy))
           for s, tt, sh, yy in zip(sid, t, shipped, y)}
    return lut


def kde_logp(x, atoms, bw):
    """log p(x) under Gaussian KDE with given atoms+bandwidth. Vectorized logsumexp."""
    # z: (len(x), len(atoms))
    z = -0.5 * ((x[:, None] - atoms[None, :]) / bw) ** 2
    m = z.max(axis=1)
    lse = m + np.log(np.exp(z - m[:, None]).sum(axis=1) + 1e-300)
    return lse - np.log(len(atoms)) - np.log(bw) - 0.5 * LOG2PI


def main():
    val_sorted = val_ids_split()
    print(f"VAL series: {len(val_sorted)}")
    lut = reconstruct_shipped()

    rng = np.random.default_rng(0)
    rows_sid, rows_t, rows_y, rows_ship = [], [], [], []
    f_cum, f_trail = [], []
    t0 = time.time()
    n = 0
    for s in iter_series("train", ids=list(val_sorted)):
        xh = s.x_hist.astype(np.float64)
        xo = s.x_online.astype(np.float64)
        if len(xo) <= STEP_GRID[0] + 5 or len(xh) < 50:
            continue
        mu, sd = xh.mean(), xh.std() + 1e-9
        zh = np.clip((xh - mu) / sd, -CLIP, CLIP)
        zo = np.clip((xo - mu) / sd, -CLIP, CLIP)

        atoms = zh if len(zh) <= MAX_ATOMS else zh[rng.choice(len(zh), MAX_ATOMS, replace=False)]
        bw = 0.9 * min(atoms.std() + 1e-9,
                       (np.subtract(*np.percentile(atoms, [75, 25])) / 1.34) + 1e-9) * len(atoms) ** (-0.2)
        bw = max(bw, 0.05)

        baseline = float(np.mean(kde_logp(atoms, atoms, bw)))   # self log-density (per-series anchor)
        d = kde_logp(zo, atoms, bw)                              # log p_H of each online point
        csum = np.cumsum(d)
        for t in STEP_GRID:
            if len(zo) <= t:
                continue
            look = lut.get((s.series_id, t))
            if look is None:
                continue
            cum_mean = csum[t - 1] / t
            lo = max(0, t - TRAIL)
            trail_mean = (csum[t - 1] - (csum[lo - 1] if lo > 0 else 0.0)) / (t - lo)
            rows_sid.append(s.series_id); rows_t.append(t)
            rows_y.append(look[1]); rows_ship.append(look[0])
            f_cum.append(baseline - cum_mean)       # excess neg-loglik (higher => more "broken")
            f_trail.append(baseline - trail_mean)
        n += 1
        if n % 400 == 0:
            print(f"   {n}/{len(val_sorted)}  {time.time()-t0:.0f}s")

    sid = np.asarray(rows_sid, np.int64)
    t = np.asarray(rows_t, np.int64)
    y = np.asarray(rows_y, np.int64)
    ship = np.asarray(rows_ship, np.float64)
    feats = {"kde_llr_cum": np.asarray(f_cum), "kde_llr_trail": np.asarray(f_trail)}
    print(f"dataset rows {len(y)}  pos_rate {y.mean():.3f}  series {len(set(sid.tolist()))}  "
          f"steps {len(set(t.tolist()))}  ({time.time()-t0:.0f}s)")

    # honest halves by series (rng HALF_SEED)
    u = np.array(sorted(set(int(x) for x in sid)))
    np.random.default_rng(HALF_SEED).shuffle(u)
    halfA = set(int(v) for v in u[: len(u) // 2])
    inA = np.array([int(x) in halfA for x in sid])

    shz = zscore(ship)
    base_full = ts_auc_grouped(t, y, sigmoid(shz))
    base_A = ts_auc_grouped(t[inA], y[inA], sigmoid(shz[inA]))
    base_B = ts_auc_grouped(t[~inA], y[~inA], sigmoid(shz[~inA]))
    print(f"\nshipped (this subset)  full={base_full:.4f}  halfA={base_A:.4f}  halfB={base_B:.4f}")

    best_overall = -1.0
    for name, fv in feats.items():
        # standalone cross-sectional TS-AUC (orient sign by full-set AUC)
        sa = ts_auc_grouped(t, y, fv)
        if sa < 0.5:
            fv = -fv
            sa = 1.0 - sa
        rc = float(np.corrcoef(ship, fv)[0, 1])
        hz = zscore(fv)
        print(f"\n=== {name}   standalone TS-AUC={sa:.4f}   rankcorr={rc:+.3f} ===")
        best = None
        for w in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
            p = sigmoid(shz + w * hz)
            f = ts_auc_grouped(t, y, p)
            a = ts_auc_grouped(t[inA], y[inA], p[inA])
            b = ts_auc_grouped(t[~inA], y[~inA], p[~inA])
            flag = "  <-- both halves up" if (a > base_A + 1e-5 and b > base_B + 1e-5) else ""
            print(f"  w={w:4.2f}  full={f:.4f} ({f-base_full:+.4f})  "
                  f"halfA={a:.4f} ({a-base_A:+.4f})  halfB={b:.4f} ({b-base_B:+.4f}){flag}")
            robust = min(f - base_full, a - base_A, b - base_B)
            if best is None or robust > best[1]:
                best = (w, robust)
        print(f"  BEST robust min-lift across (full,A,B): {best[1]:+.4f} @ w={best[0]}")
        best_overall = max(best_overall, best[1])

    print("\n================ VERDICT ================")
    if best_overall > 0.0005:
        print(f"PROMISING: robust min-lift {best_overall:+.4f} -> productionize KDE-LLR feature(s).")
    else:
        print(f"REDUNDANT: best robust min-lift {best_overall:+.4f} (<= +0.0005). The one missing "
              f"blueprint feature (KDE-LLR) adds nothing -> per-series signal space fully closed.")


if __name__ == "__main__":
    main()
