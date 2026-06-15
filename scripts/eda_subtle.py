"""Deep EDA round 6 — hunt for signal on SUBTLE / distribution-only breaks.

The base model is weakest exactly where most breaks live: ~68% of breaks are
"subtle" (no clear mean/var/acf shift) and score only ~0.56 cross-sectional AUC.
This script tests a battery of *candidate* online detectors that are NOT in the
current feature set, to see which ones separate broken-vs-not at a fixed online
step, with emphasis on the subtle subset.

For each candidate statistic we compute, at several snapshot online steps t, the
value for every VAL series using only information available at t (historical
segment + online points up to t), then measure the cross-sectional AUC at that
step (broken-by-t = positive, else negative), aggregated TS-AUC style. We report
overall AUC and the AUC restricted to subtle-break positives — the lever.

Run: uv run python scripts/eda_subtle.py
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from sb.data import iter_series  # noqa: E402

SEED = 42
SNAP_STEPS = [50, 100, 150, 200, 300, 400]   # metric-heavy online steps
TRAIL = 120                                   # trailing-window length for windowed stats
RNG = np.random.default_rng(SEED)


# ----------------------------- candidate detectors -----------------------------
# Each takes (x_hist, x_win) where x_win is the online points (cumulative or
# trailing) and returns a scalar that should be LARGE when a break is present.

def _acf1(v):
    v = v - v.mean()
    d = float(np.dot(v, v))
    return float(np.dot(v[:-1], v[1:]) / d) if d > 1e-12 else 0.0


def _abs_acf_change(xh, xw):
    return abs(_acf1(np.abs(xw - xw.mean())) - _acf1(np.abs(xh - xh.mean())))


def _sq_acf_change(xh, xw):
    a = (xw - xw.mean()) ** 2
    b = (xh - xh.mean()) ** 2
    return abs(_acf1(a) - _acf1(b))


def _spectral_entropy(v):
    v = v - v.mean()
    n = len(v)
    if n < 16:
        return 0.0
    f = np.fft.rfft(v * np.hanning(n))
    p = (f.real ** 2 + f.imag ** 2)
    s = p.sum()
    if s <= 1e-12:
        return 0.0
    p = p / s
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)) / np.log(len(p) + 1e-12))


def _spec_entropy_change(xh, xw):
    return abs(_spectral_entropy(xw) - _spectral_entropy(xh))


def _perm_entropy(v, m=3):
    n = len(v)
    if n < m + 2:
        return 0.0
    # ordinal patterns of length m
    from itertools import permutations
    perms = {p: i for i, p in enumerate(permutations(range(m)))}
    counts = np.zeros(len(perms))
    for i in range(n - m + 1):
        pat = tuple(np.argsort(v[i:i + m]))
        counts[perms[pat]] += 1
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(perms)))


def _perm_entropy_change(xh, xw):
    return abs(_perm_entropy(xw) - _perm_entropy(xh))


def _energy_distance(xh, xw):
    # 1-D energy distance via sorted CDFs (cheap, exact for 1-D up to scaling)
    a = np.sort(xh)
    b = np.sort(xw)
    # subsample hist for speed
    if len(a) > 400:
        a = a[np.linspace(0, len(a) - 1, 400).astype(int)]
    if len(b) > 400:
        b = b[np.linspace(0, len(b) - 1, 400).astype(int)]
    grid = np.concatenate([a, b])
    grid.sort()
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    dx = np.diff(grid)
    cv = (fa - fb) ** 2
    return float(np.sum(0.5 * (cv[1:] + cv[:-1]) * dx))  # Cramer-von Mises-ish


def _anderson_darling(xh, xw):
    # 2-sample AD statistic (tail-weighted KS); use pooled ranks
    a, b = xh, xw
    n, m = len(a), len(b)
    pooled = np.concatenate([a, b])
    order = np.argsort(pooled, kind="mergesort")
    is_a = np.concatenate([np.ones(n), np.zeros(m)])[order]
    N = n + m
    Fa = np.cumsum(is_a) / n
    Fb = np.cumsum(1 - is_a) / m
    H = np.arange(1, N + 1) / N
    w = H * (1 - H)
    w[w < 1e-6] = 1e-6
    d = (Fa - Fb) ** 2 / w
    return float(np.mean(d[:-1]))


def _wasserstein1(xh, xw):
    a = np.sort(xh); b = np.sort(xw)
    q = np.linspace(0, 1, 64)
    qa = np.quantile(a, q); qb = np.quantile(b, q)
    sd = xh.std() + 1e-9
    return float(np.mean(np.abs(qa - qb)) / sd)


def _iqr_ratio(xh, xw):
    iqh = np.subtract(*np.percentile(xh, [75, 25])) + 1e-9
    iqw = np.subtract(*np.percentile(xw, [75, 25])) + 1e-9
    return abs(np.log(iqw / iqh))


def _turning_point_rate_change(xh, xw):
    def tpr(v):
        if len(v) < 3:
            return 0.0
        d = np.diff(v)
        return float(np.mean((d[:-1] * d[1:]) < 0))
    return abs(tpr(xw) - tpr(xh))


def _hurst(v):
    v = v - v.mean()
    n = len(v)
    if n < 32:
        return 0.5
    lags = [2, 4, 8, 16, 32]
    tau = []
    ll = []
    for lag in lags:
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
    "abs_acf_change": _abs_acf_change,
    "sq_acf_change": _sq_acf_change,
    "spec_entropy_change": _spec_entropy_change,
    "perm_entropy_change": _perm_entropy_change,
    "energy_cvm": _energy_distance,
    "anderson_darling": _anderson_darling,
    "wasserstein1": _wasserstein1,
    "iqr_log_ratio": _iqr_ratio,
    "turning_point_change": _turning_point_rate_change,
    "hurst_change": _hurst_change,
}


def classify_break(xh, post):
    if len(post) < 20:
        return "short_post"
    mu, sd = xh.mean(), xh.std() + 1e-12
    z = (post - mu) / sd
    dev = dict(
        mean=abs(float(z.mean())) / 0.25,
        var=abs(float(np.log(post.var() / (sd * sd) + 1e-12))) / 0.30,
        acf=abs(_acf1(post) - _acf1(xh)) / 0.12,
    )
    best = max(dev, key=dev.get)
    return best if dev[best] >= 1.0 else "subtle"


def main():
    import pandas as pd
    idx = pd.read_parquet("Dataset/y_train_index.parquet")
    tau_map = idx["tau_index"].to_dict()
    uniq = np.array(sorted(tau_map.keys()))
    RNG.shuffle(uniq)
    n_val = int(0.2 * len(uniq))
    val_ids = set(uniq[:n_val].tolist())

    # gather per-series cached arrays
    series = {}
    btype = {}
    print(f"loading {len(val_ids)} VAL series ...")
    for s in iter_series("train", ids=[int(v) for v in val_ids]):
        series[s.series_id] = (s.x_hist.astype(np.float64), s.x_online.astype(np.float64),
                               s.tau_index if s.tau_index is not None else -1)
        if s.has_break:
            btype[s.series_id] = classify_break(s.x_hist, s.x_online[s.tau_index:])
    sids = list(series.keys())
    n_break = sum(1 for v in btype.values() if v != "short_post")
    from collections import Counter
    print("break-type counts:", dict(Counter(btype.values())))

    # For each candidate, accumulate cross-sectional AUC over snapshot steps.
    def cross_auc(vals, labels):
        # Mann-Whitney AUC
        pos = vals[labels == 1]; neg = vals[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan, 0
        order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
        ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
        rsum = ranks[:len(pos)].sum()
        auc = (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        return float(auc), len(pos)

    results = {c: [] for c in CANDS}
    results_subtle = {c: [] for c in CANDS}

    for mode in ["cumulative", "trailing"]:
        print(f"\n========== {mode} window ==========")
        for c in CANDS:
            results[c] = []; results_subtle[c] = []
        for t in SNAP_STEPS:
            # build value vectors + labels at this step
            vv = {c: [] for c in CANDS}
            lab = []
            subtle_flag = []
            for sid in sids:
                xh, xo, tau = series[sid]
                if len(xo) <= 5:
                    continue
                tt = min(t, len(xo))
                if mode == "cumulative":
                    xw = xo[:tt]
                else:
                    xw = xo[max(0, tt - TRAIL):tt]
                if len(xw) < 16:
                    continue
                is_pos = 1 if (tau >= 0 and tau < tt) else 0
                lab.append(is_pos)
                subtle_flag.append(1 if (is_pos and btype.get(sid) == "subtle") else 0)
                for c, fn in CANDS.items():
                    try:
                        vv[c].append(fn(xh, xw))
                    except Exception:
                        vv[c].append(0.0)
            lab = np.array(lab); subtle_flag = np.array(subtle_flag)
            for c in CANDS:
                arr = np.nan_to_num(np.array(vv[c]))
                a, npos = cross_auc(arr, lab)
                results[c].append(a)
                # subtle-only: positives = subtle breaks, negatives = all non-pos
                sub_lab = lab.copy()
                keep = (subtle_flag == 1) | (lab == 0)
                asub, _ = cross_auc(arr[keep], lab[keep])
                results_subtle[c].append(asub)
        # report
        print(f"{'candidate':24s} {'AUC_all(mean)':>14s} {'AUC_subtle(mean)':>17s}")
        ranked = sorted(CANDS, key=lambda c: -np.nanmean(results_subtle[c]))
        for c in ranked:
            print(f"{c:24s} {np.nanmean(results[c]):>14.4f} {np.nanmean(results_subtle[c]):>17.4f}")


if __name__ == "__main__":
    main()
