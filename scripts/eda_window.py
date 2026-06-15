"""Round 6 — does a RAW TRAILING-WINDOW shape representation add signal that the
shipped stack (GBT summary-stats + aggregate-fed GRU) lacks?

Four avenues exhausted prove single-statistic and bagging diversity are mined out
on the 162-feat calibrated input. The one untested lever is a DIFFERENT INPUT: the
raw per-series-standardized online stream seen through a WINDOWED receptive field
(local shape), not summary stats and not a single point (the failed raw-GRU).

Before building a full 1D-CNN member, cheaply probe whether ANY fixed bank of
matched filters over the trailing window separates breaks ON TOP OF the shipped
logit. We compute, at snapshot steps, a battery of windowed shape features on the
standardized stream:
  - step/edge matched filter (difference of leading vs trailing half means) at
    multiple window lengths (the canonical level-shift detector with a receptive
    field);
  - local variance-envelope contrast (recent var vs mid var);
  - local lag-1 autocorr contrast;
  - max abs short-time mean over sub-blocks (sharp transient);
  - ramp/trend matched filter (linear-weighted sum).
For each we measure incremental AUC vs the cached SHIPPED logit (base+GRU) over a
small lambda grid, plus rank-correlation. If even the best windowed shape feature
gives ~0 incremental lift, the raw-window CNN is also dead and we consolidate.
If something shows a real +lift with modest corr, it justifies the CNN.

Run: uv run python scripts/eda_window.py
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from sb.data import iter_series  # noqa: E402

SNAP_STEPS = [100, 150, 200, 250, 300, 400]
LAMBDAS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
WINS = (20, 40, 80, 160)


def _edge(zw, L):
    """Step matched filter over the last L points: |mean(2nd half) - mean(1st half)|."""
    w = zw[-L:] if len(zw) >= L else zw
    if len(w) < 8:
        return 0.0
    h = len(w) // 2
    return abs(w[h:].mean() - w[:h].mean())


def _var_env(zw, L):
    w = zw[-L:] if len(zw) >= L else zw
    if len(w) < 8:
        return 0.0
    h = len(w) // 2
    return abs(np.log((w[h:].var() + 1e-6) / (w[:h].var() + 1e-6)))


def _acf_env(zw, L):
    def a1(v):
        v = v - v.mean(); d = float(np.dot(v, v))
        return float(np.dot(v[:-1], v[1:]) / d) if d > 1e-12 else 0.0
    w = zw[-L:] if len(zw) >= L else zw
    if len(w) < 12:
        return 0.0
    h = len(w) // 2
    return abs(a1(w[h:]) - a1(w[:h]))


def _transient(zw, L, blk=8):
    w = zw[-L:] if len(zw) >= L else zw
    if len(w) < blk * 2:
        return 0.0
    nb = len(w) // blk
    means = np.array([w[i * blk:(i + 1) * blk].mean() for i in range(nb)])
    return float(np.max(np.abs(means)))


def _ramp(zw, L):
    w = zw[-L:] if len(zw) >= L else zw
    if len(w) < 8:
        return 0.0
    n = len(w)
    t = np.linspace(-1, 1, n)
    return abs(float(np.dot(w, t)) / n)


def cross_auc(score, label):
    label = np.asarray(label)
    n_pos = int(label.sum()); n_neg = int(len(label) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan, n_pos, n_neg
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sl = score[order]
    ranks[order] = np.arange(1, len(score) + 1)
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
    print("loading caches ...")
    base = np.load("features/val_base_logits.npz")
    sid_arr, t_arr = base["series_id"], base["t_online"]
    gru = np.mean([np.load("features/" + f)["val_logit"] for f in
                   ("val_seq_logits.npz", "val_seq_logits_s1.npz", "val_seq_logits_s2.npz")], axis=0)
    shipped = 0.55 * base["base_logit"] + 0.45 * gru
    ship_lookup = {(int(sid_arr[i]), int(t_arr[i])): float(shipped[i]) for i in range(len(sid_arr))}
    val_ids = sorted(set(int(s) for s in sid_arr))
    print(f"  {len(val_ids)} VAL series")

    cands = {}
    for L in WINS:
        cands[f"edge{L}"] = (lambda zw, L=L: _edge(zw, L))
        cands[f"varenv{L}"] = (lambda zw, L=L: _var_env(zw, L))
        cands[f"acfenv{L}"] = (lambda zw, L=L: _acf_env(zw, L))
        cands[f"transient{L}"] = (lambda zw, L=L: _transient(zw, L))
        cands[f"ramp{L}"] = (lambda zw, L=L: _ramp(zw, L))

    stash = {c: {t: [] for t in SNAP_STEPS} for c in cands}
    processed = 0
    for s in iter_series("train", ids=val_ids):
        xh = s.x_hist.astype(np.float64)
        xo = s.x_online.astype(np.float64)
        mu, sd = xh.mean(), xh.std() + 1e-12
        z = (xo - mu) / sd
        sid = s.series_id
        for t in SNAP_STEPS:
            if t > len(xo):
                continue
            key = (sid, t)
            if key not in ship_lookup:
                continue
            zw = z[:t]
            lab = 1 if (s.has_break and s.tau_index is not None and s.tau_index < t) else 0
            for c, fn in cands.items():
                stash[c][t].append((fn(zw), lab, ship_lookup[key]))
        processed += 1
        if processed % 250 == 0:
            print(f"  ...{processed}")

    def wmean(pairs):
        num = sum(a * w for a, w in pairs); den = sum(w for _, w in pairs)
        return num / den if den else np.nan

    # shipped TS-AUC baseline
    any_c = next(iter(cands))
    base_pairs = []
    for t in SNAP_STEPS:
        rows = stash[any_c][t]
        if len(rows) < 30:
            continue
        b = np.array([r[2] for r in rows]); lab = np.array([r[1] for r in rows])
        a0, npos, nneg = cross_auc(b, lab)
        if np.isfinite(a0):
            base_pairs.append((a0, npos * nneg))
    ship_ts = wmean(base_pairs)
    print(f"\nshipped TS-AUC (snapshot approx): {ship_ts:.4f}\n")
    print(f"{'shape feature':<16}{'standalone':>11}{'blend':>8}{'lift':>8}{'rankcorr':>10}{'lam':>7}")

    results = []
    for c in cands:
        sa_pairs = []; bl_pairs = []; rcs = []; lams = []
        for t in SNAP_STEPS:
            rows = stash[c][t]
            if len(rows) < 30:
                continue
            sv = np.array([r[0] for r in rows], dtype=np.float64)
            lab = np.array([r[1] for r in rows], dtype=np.int64)
            b = np.array([r[2] for r in rows], dtype=np.float64)
            asa, npos, nneg = cross_auc(sv, lab)
            if not np.isfinite(asa):
                continue
            w = npos * nneg
            sa_pairs.append((asa, w))
            rcs.append(np.corrcoef(zscore(b), zscore(sv))[0, 1] if sv.std() > 1e-12 else 0.0)
            a0, _, _ = cross_auc(b, lab)
            best = (a0, 0.0)
            for lam in LAMBDAS:
                blended = b + lam * zscore(sv) * b.std()
                ab, _, _ = cross_auc(blended, lab)
                if ab > best[0]:
                    best = (ab, lam)
            bl_pairs.append((best[0], w)); lams.append(best[1])
        sa = wmean(sa_pairs); bl = wmean(bl_pairs)
        rc = float(np.nanmean(rcs)); lam_mode = max(set(lams), key=lams.count) if lams else 0.0
        results.append((c, sa, bl, bl - ship_ts, rc, lam_mode))

    for c, sa, bl, lift, rc, lam in sorted(results, key=lambda r: -r[3]):
        print(f"{c:<16}{sa:>11.4f}{bl:>8.4f}{lift:>+8.4f}{rc:>10.3f}{lam:>7.2f}")


if __name__ == "__main__":
    main()
