"""Deeper EDA: break-type structure, detectability bounds, label sanity.

Questions answered here (feeding the round-3 modelling decisions):
 1. For broken series, what does the post-break segment look like vs historical
    on each statistic (mean z, log-var, acf1, KS, diff-var)? -> which detectors
    have headroom.
 2. How much of the positive mass at each step is "fresh" (elapsed < 25)?
    -> structural ceiling on per-step AUC.
 3. Label sanity: no-break series whose online segment is grossly different
    from historical (the W23 forum fix mentioned 29 such series).
 4. Per-series statistic null-spread: how heterogeneous are sliding-window
    stats across series under H0? -> motivates per-series empirical-null
    normalisation of features.

Run: uv run python scripts/eda2.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from sb.data import iter_series

rng = np.random.default_rng(0)

rows = []
fresh_stats = []
label_suspects = []
null_spread = []

n_done = 0
for s in iter_series("train"):
    x_h, x_o = s.x_hist, s.x_online
    n, T = len(x_h), len(x_o)
    mu, sd = x_h.mean(), x_h.std() + 1e-12

    if s.has_break:
        tau = s.tau_index
        post = x_o[tau:]
        pre = np.concatenate([x_h[-500:], x_o[:tau]])
        if len(post) >= 20:
            zpost = (post - mu) / sd
            dpost = np.diff(post)
            dh = np.diff(x_h)
            # acf1 of post vs hist
            def acf1(v):
                v = v - v.mean()
                d = np.dot(v, v)
                return np.dot(v[:-1], v[1:]) / d if d > 0 else 0.0
            # KS between post and historical sample
            hs = np.sort(x_h)
            ks = np.abs(np.searchsorted(hs, np.sort(post)) / len(hs)
                        - np.arange(1, len(post) + 1) / len(post)).max()
            rows.append(dict(
                sid=s.series_id, tau=tau, T=T, n=n, post_len=len(post),
                mean_z=abs(zpost.mean()) * np.sqrt(len(post)),
                mean_shift=abs(zpost.mean()),
                logvar=abs(np.log(post.var() / (sd * sd) + 1e-12)),
                acf_diff=abs(acf1(post) - acf1(x_h)),
                dvar_ratio=abs(np.log((dpost.var() + 1e-12) / (dh.var() + 1e-12))),
                ks=ks,
            ))
    else:
        # label sanity: full online vs historical KS + var-ratio
        if T >= 50:
            hs = np.sort(x_h)
            ks = np.abs(np.searchsorted(hs, np.sort(x_o)) / len(hs)
                        - np.arange(1, T + 1) / T).max()
            lv = abs(np.log(x_o.var() / (sd * sd) + 1e-12))
            mz = abs((x_o.mean() - mu) / sd) * np.sqrt(T)
            if ks > 0.35 or lv > 1.2 or mz > 6:
                label_suspects.append((s.series_id, T, round(ks, 3), round(lv, 2), round(mz, 1)))

    # null spread of window-50 stats over the HISTORICAL segment (every 25th pos)
    if n_done % 10 == 0 and n >= 1200:
        w = 50
        zs = []
        lvs = []
        for st in range(n // 2, n - w, 25):
            win = x_h[st:st + w]
            zs.append((win.mean() - mu) / (sd / np.sqrt(w)))
            lvs.append(np.log(win.var() / (sd * sd) + 1e-12))
        null_spread.append((np.std(zs), np.std(lvs)))
    n_done += 1

import pandas as pd

df = pd.DataFrame(rows)
print(f"\nBroken series with post_len>=20: {len(df)}")
print("\n=== Post-break deviation magnitudes (which detector has signal?) ===")
for c in ["mean_shift", "logvar", "acf_diff", "dvar_ratio", "ks"]:
    q = df[c].quantile([0.25, 0.5, 0.75, 0.9]).round(3).tolist()
    print(f"  {c:12s} q25/50/75/90 = {q}")

# how many breaks are detectable by ANY single statistic at a loose threshold
TH = dict(mean_shift=0.25, logvar=0.30, acf_diff=0.12, dvar_ratio=0.30, ks=0.15)
det = pd.DataFrame({c: df[c] > th for c, th in TH.items()})
print(f"\nShare detectable by each stat (loose thresholds {TH}):")
print(det.mean().round(3).to_string())
print(f"Share detectable by >=1 stat: {det.any(axis=1).mean():.3f}")
print(f"Share detectable by NONE:    {(~det.any(axis=1)).mean():.3f}")

ns = np.array(null_spread)
print(f"\n=== Per-series null spread of window-50 stats (should be ~1.0 / const if iid) ===")
print(f"  std of mean-z across series: median={np.median(ns[:,0]):.3f}, "
      f"q10={np.quantile(ns[:,0],0.1):.3f}, q90={np.quantile(ns[:,0],0.9):.3f}")
print(f"  std of logvar across series: median={np.median(ns[:,1]):.3f}, "
      f"q10={np.quantile(ns[:,1],0.1):.3f}, q90={np.quantile(ns[:,1],0.9):.3f}")
print("  (wide q10-q90 range => per-series empirical-null normalisation needed)")

print(f"\n=== Label suspects (no-break but grossly different online) : {len(label_suspects)} ===")
for t in label_suspects[:25]:
    print("  sid,T,ks,logvar,mean_z:", t)
