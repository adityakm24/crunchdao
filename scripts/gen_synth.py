"""Round 11 — synthetic break-series generator (DGP-matched).

Generates synthetic series whose statistical fingerprint matches the real DGP
measured by eda_dgp.py, then runs the SAME feature pipeline (features4.
StreamingDetector) so the output plugs straight into the training matrix.

Realism choices (matched to features/dgp_params.npz):
  - segment lengths bootstrapped from real (n_hist, n_online)
  - process = AR(1) with Student-t innovations; rho bootstrapped from real;
    df derived from real per-series excess kurtosis (heavy tails preserved)
  - unit historical variance (real data has std_h == 1.000 exactly): standardize
    each series by its OWN historical mean/std (the detector does this anyway)
  - break injected with prob = real frac_broken; tau ~ real tau-fraction pool;
    (delta_mean, log_var_ratio, delta_AR1) bootstrapped JOINTLY from real broken
    series (preserves marginals AND cross-correlations AND the type mix)
  - optional --tilt: oversample moderately-detectable breaks (bootstrap weight
    rises for KS in [0.10,0.30]) to add resolution at the metric-heavy frontier

Synthetic series ids start at 1_000_000 (no collision with real 0..9999).
Output matches train_features_v4.npz schema: X,y,series_id,t_online,feature_names.

Run: uv run python scripts/gen_synth.py --n 5000 --seed 0 --tilt 0 \
        --out features/synth_features_v4.npz
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from sb.features4 import StreamingDetector, FEATURE_NAMES, N_FEATURES   # noqa: E402

SID0 = 1_000_000


def ar1_series(n, rho, df, sigma, mean, x0, rng):
    """AR(1): x_t = mean + rho*(x_{t-1}-mean) + innov; stationary var ~ sigma^2."""
    if df >= 200 or df <= 4.2:
        e = rng.standard_normal(n) if df >= 200 else \
            rng.standard_t(max(df, 4.3), size=n) / np.sqrt(max(df, 4.3) / (max(df, 4.3) - 2))
    else:
        e = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2))
    innov = e * sigma * np.sqrt(max(1.0 - rho * rho, 1e-6))
    x = np.empty(n, dtype=np.float64)
    prev = x0
    for i in range(n):
        prev = mean + rho * (prev - mean) + innov[i]
        x[i] = prev
    return x


def kurt_to_df(k):
    if k <= 0.05:
        return 250.0
    df = 4.0 + 6.0 / k
    return float(np.clip(df, 4.5, 250.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tilt", type=float, default=0.0,
                    help="0=faithful; >0 oversamples moderate-KS breaks")
    ap.add_argument("--out", default="features/synth_features_v4.npz")
    args = ap.parse_args()

    p = np.load("features/dgp_params.npz")
    n_hist_pool = p["n_hist"]; n_online_pool = p["n_online"]
    rho_pool = p["rho_h"]; kurt_pool = p["kurt_h"]
    bdm = p["brk_dmean"]; blv = p["brk_lvr"]; bda = p["brk_dacf"]
    bks = p["brk_ks"]; btf = p["brk_taufrac"]
    frac_broken = float(p["frac_broken"])
    nbrk = len(bdm)

    # break bootstrap weights (optionally tilt toward moderate KS frontier)
    if args.tilt > 0:
        w = 1.0 + args.tilt * np.exp(-((bks - 0.20) ** 2) / (2 * 0.08 ** 2))
        w = w / w.sum()
    else:
        w = None

    rng = np.random.default_rng(args.seed)
    det = StreamingDetector()
    X_chunks, y_chunks, id_chunks, t_chunks = [], [], [], []
    t0 = time.time()
    n_points = 0
    n_broken = 0

    for k in range(args.n):
        nh = int(rng.choice(n_hist_pool))
        no = int(rng.choice(n_online_pool))
        no = max(no, 30)
        rho = float(rng.choice(rho_pool))
        df = kurt_to_df(float(rng.choice(kurt_pool)))

        # historical + (pre-break) online as one stationary AR(1), unit var
        burn = 100
        full = ar1_series(burn + nh + no, rho, df, 1.0, 0.0, 0.0, rng)[burn:]
        x_hist = full[:nh].copy()
        x_online = full[nh:nh + no].copy()

        broken = rng.random() < frac_broken
        tau = -1
        if broken:
            j = int(rng.choice(nbrk, p=w)) if w is not None else int(rng.integers(nbrk))
            tau_frac = float(btf[j])
            tau = int(np.clip(round(tau_frac * no), 3, no - 3))
            dmean = float(bdm[j]); lvr = float(blv[j]); dacf = float(bda[j])
            rho_post = float(np.clip(rho + dacf, -0.95, 0.95))
            sigma_post = float(np.exp(0.5 * lvr))
            x0 = x_online[tau - 1]
            post = ar1_series(no - tau, rho_post, df, sigma_post, dmean, x0, rng)
            x_online[tau:] = post

        # standardize by historical stats (real data has std_h == 1.000)
        mu = x_hist.mean(); sd = x_hist.std() + 1e-9
        x_hist = (x_hist - mu) / sd
        x_online = (x_online - mu) / sd

        det.calibrate(x_hist)
        T = len(x_online)
        feats = np.empty((T, N_FEATURES), dtype=np.float32)
        for i, xv in enumerate(x_online):
            feats[i] = det.update(float(xv))
        y = np.zeros(T, dtype=np.int8)
        if tau >= 0:
            y[tau:] = 1
            n_broken += 1
        X_chunks.append(feats)
        y_chunks.append(y)
        id_chunks.append(np.full(T, SID0 + k, dtype=np.int32))
        t_chunks.append(np.arange(T, dtype=np.int32))
        n_points += T
        if (k + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {k+1:5d}/{args.n} series  {n_points:>9,} pts  {el:6.1f}s "
                  f"({1e6*el/max(n_points,1):.1f} us/pt)  broken={n_broken}", flush=True)

    X = np.vstack(X_chunks); y = np.concatenate(y_chunks)
    sid = np.concatenate(id_chunks); t_online = np.concatenate(t_chunks)
    print(f"\nSynthetic: X={X.shape}  pos_rate={y.mean():.4f}  "
          f"broken_series={n_broken}/{args.n} ({n_broken/args.n:.3f})")
    np.savez(args.out, X=X, y=y, series_id=sid, t_online=t_online,
             feature_names=np.array(FEATURE_NAMES))
    print(f"saved {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
