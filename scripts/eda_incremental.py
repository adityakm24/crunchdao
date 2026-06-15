"""Round 6 — decisive ORTHOGONALITY test for the EDA-winning detectors.

The standalone AUC of a detector is irrelevant if it is redundant with the base
model. What matters is the INCREMENTAL cross-sectional AUC when the detector is
blended on top of the actual base prediction. This script:

  1. loads the cached VAL base logits (features/val_base_logits.npz), keyed by
     (series_id, t_online), with the per-series label;
  2. streams each VAL series and, at several metric-heavy snapshot steps, computes
     the top candidate detectors (Anderson-Darling, energy/CvM, Hurst, Wasserstein,
     plus KS as the in-set reference) on (hist vs online[:t]);
  3. at each step, aligns detector values with the base logit for the same rows and
     measures AUC(base) vs AUC(base + lambda * z(detector)) over a small lambda grid,
     plus rank-correlation(base, detector);
  4. aggregates TS-AUC style (weight = n_pos * n_neg per step).

A detector is worth building into v6 only if it shows a CONSISTENT positive blend
lift AND modest rank-correlation with the base. Run:
    uv run python scripts/eda_incremental.py
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from sb.data import iter_series  # noqa: E402

SEED = 42
SNAP_STEPS = [100, 150, 200, 250, 300, 400]
LAMBDAS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
RNG = np.random.default_rng(SEED)


# ----------------------------- detectors (reused) -----------------------------
def _acf1(v):
    v = v - v.mean()
    d = float(np.dot(v, v))
    return float(np.dot(v[:-1], v[1:]) / d) if d > 1e-12 else 0.0


def _energy_cvm(xh, xw):
    a = np.sort(xh); b = np.sort(xw)
    if len(a) > 400:
        a = a[np.linspace(0, len(a) - 1, 400).astype(int)]
    if len(b) > 400:
        b = b[np.linspace(0, len(b) - 1, 400).astype(int)]
    grid = np.concatenate([a, b]); grid.sort()
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    dx = np.diff(grid)
    cv = (fa - fb) ** 2
    return float(np.sum(0.5 * (cv[1:] + cv[:-1]) * dx))


def _anderson_darling(xh, xw):
    a, b = xh, xw
    n, m = len(a), len(b)
    pooled = np.concatenate([a, b])
    order = np.argsort(pooled, kind="mergesort")
    is_a = np.concatenate([np.ones(n), np.zeros(m)])[order]
    N = n + m
    Fa = np.cumsum(is_a) / n
    Fb = np.cumsum(1 - is_a) / m
    H = np.arange(1, N + 1) / N
    w = H * (1 - H); w[w < 1e-6] = 1e-6
    d = (Fa - Fb) ** 2 / w
    return float(np.mean(d[:-1]))


def _ks(xh, xw):
    a = np.sort(xh); b = np.sort(xw)
    grid = np.concatenate([a, b]); grid.sort()
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def _wasserstein1(xh, xw):
    a = np.sort(xh); b = np.sort(xw)
    q = np.linspace(0, 1, 64)
    qa = np.quantile(a, q); qb = np.quantile(b, q)
    sd = xh.std() + 1e-9
    return float(np.mean(np.abs(qa - qb)) / sd)


def _hurst(v):
    v = v - v.mean(); n = len(v)
    if n < 32:
        return 0.5
    tau = []; ll = []
    for lag in (2, 4, 8, 16, 32):
        if lag >= n:
            break
        diff = v[lag:] - v[:-lag]
        s = np.sqrt(np.mean(diff ** 2)) + 1e-12
        tau.append(np.log(s)); ll.append(np.log(lag))
    if len(tau) < 3:
        return 0.5
    return float(np.polyfit(ll, tau, 1)[0])


def _hurst_change(xh, xw):
    return abs(_hurst(xw) - _hurst(xh))


CANDS = {
    "anderson_darling": _anderson_darling,
    "energy_cvm": _energy_cvm,
    "ks_ref": _ks,
    "wasserstein1": _wasserstein1,
    "hurst_change": _hurst_change,
}


def cross_auc(score, label):
    """Mann-Whitney AUC; returns (auc, n_pos, n_neg) or (nan,0,0)."""
    label = np.asarray(label)
    n_pos = int(label.sum()); n_neg = int(len(label) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan, n_pos, n_neg
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sl = score[order]
    ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    i = 0
    while i < len(sl):
        j = i
        while j + 1 < len(sl) and sl[j + 1] == sl[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    auc = (ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else v * 0.0


def main():
    print("loading base logits cache ...")
    d = np.load("features/val_base_logits.npz", allow_pickle=True)
    sid_arr = d["series_id"]; t_arr = d["t_online"]; base = d["base_logit"]; y = d["y"]
    base_lookup = {}
    label_of = {}
    for i in range(len(sid_arr)):
        base_lookup[(int(sid_arr[i]), int(t_arr[i]))] = float(base[i])
        # label is per (sid,t): broken-by-t. We use the row label directly.
    val_ids = sorted(set(int(s) for s in sid_arr))
    print(f"  {len(val_ids)} VAL series, {len(sid_arr)} rows")

    # per (cand, step) accumulators
    auc_base = defaultdict(list)        # (step) -> list of (auc, w)
    auc_blend = {c: defaultdict(list) for c in CANDS}   # cand -> step -> [(best_auc, w, best_lam)]
    rankcorr = {c: [] for c in CANDS}

    # stream val series; compute detectors at snapshot steps
    processed = 0
    for s in iter_series("train", ids=val_ids):
        xh = s.x_hist.astype(np.float64)
        xo = s.x_online.astype(np.float64)
        sid = s.series_id
        for t in SNAP_STEPS:
            if t > len(xo):
                continue
            key = (sid, t)
            if key not in base_lookup:
                continue
            # store per-step row: base logit, label, detector values
            xw = xo[:t]
            if len(xw) < 10:
                continue
            row = {"base": base_lookup[key]}
            # label: broken by t  ->  has_break and tau_index < t
            lab = 1 if (s.has_break and s.tau_index is not None and s.tau_index < t) else 0
            row["y"] = lab
            for c, fn in CANDS.items():
                try:
                    row[c] = fn(xh, xw)
                except Exception:
                    row[c] = 0.0
            _stash.setdefault(t, []).append(row)
        processed += 1
        if processed % 200 == 0:
            print(f"  ...{processed} series")

    # evaluate per step
    print("\nper-step incremental evaluation")
    for t in SNAP_STEPS:
        rows = _stash.get(t, [])
        if len(rows) < 30:
            continue
        b = np.array([r["base"] for r in rows], dtype=np.float64)
        lab = np.array([r["y"] for r in rows], dtype=np.int64)
        a0, npos, nneg = cross_auc(b, lab)
        if not np.isfinite(a0):
            continue
        w = npos * nneg
        auc_base[t].append((a0, w))
        for c in CANDS:
            dv = np.array([r[c] for r in rows], dtype=np.float64)
            rankcorr[c].append(np.corrcoef(zscore(b), zscore(dv))[0, 1] if dv.std() > 1e-12 else 0.0)
            best = (a0, 0.0)
            for lam in LAMBDAS:
                blended = b + lam * zscore(dv) * b.std()
                ab, _, _ = cross_auc(blended, lab)
                if ab > best[0]:
                    best = (ab, lam)
            auc_blend[c][t].append((best[0], w, best[1]))

    # aggregate TS-AUC style
    def wmean(pairs):
        num = sum(a * w for a, w in pairs); den = sum(w for _, w in pairs)
        return num / den if den else np.nan

    base_ts = wmean([p for t in SNAP_STEPS for p in auc_base[t]])
    print(f"\nbase TS-AUC (snapshot approx): {base_ts:.4f}\n")
    print(f"{'candidate':<20}{'base':>8}{'blend':>8}{'lift':>8}{'rankcorr':>10}{'best_lam':>9}")
    for c in CANDS:
        pairs = [(a, w) for t in SNAP_STEPS for (a, w, _) in auc_blend[c][t]]
        blend_ts = wmean(pairs)
        # most common best lambda
        lams = [lm for t in SNAP_STEPS for (_, _, lm) in auc_blend[c][t]]
        lam_mode = max(set(lams), key=lams.count) if lams else 0.0
        rc = float(np.nanmean(rankcorr[c]))
        print(f"{c:<20}{base_ts:>8.4f}{blend_ts:>8.4f}{blend_ts - base_ts:>+8.4f}{rc:>10.3f}{lam_mode:>9.2f}")


_stash: dict = {}

if __name__ == "__main__":
    main()
