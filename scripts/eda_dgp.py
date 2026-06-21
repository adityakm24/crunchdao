"""Round 11 — quantitative DGP characterization for synthetic augmentation.

To generate REALISTIC synthetic break series we must match the real generating
process. This measures, over real train series, the distributions the generator
must reproduce:
  - segment lengths (n_hist, n_online)
  - historical process: AR(1) coef, innovation std, excess kurtosis
  - pre-break online vs historical (sanity: same DGP before tau?)
  - break magnitudes for broken series: delta_mean (in hist-std units),
    log variance ratio, delta AR(1), KS(pre,post)
  - tau fraction (tau_index / n_online) distribution (uniform?)
  - fraction broken; break-type mix (mean-only / var-only / both)

Run: uv run python scripts/eda_dgp.py
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "src")
from sb.data import iter_series   # noqa: E402


def ar1(x):
    x = x - x.mean()
    if len(x) < 5 or x.std() < 1e-9:
        return 0.0
    return float(np.clip(np.dot(x[1:], x[:-1]) / (np.dot(x, x) + 1e-12), -0.99, 0.99))


def excess_kurt(x):
    x = (x - x.mean()) / (x.std() + 1e-12)
    return float((x ** 4).mean() - 3.0)


def ks(a, b):
    grid = np.concatenate([a, b])
    grid.sort()
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def pct(name, arr, ps=(5, 25, 50, 75, 95)):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    qs = np.percentile(arr, ps)
    print(f"  {name:22s} n={len(arr):5d}  " + "  ".join(f"p{p}={q:+.3f}" for p, q in zip(ps, qs)))


def main():
    n_hist, n_online, taufrac = [], [], []
    rho_h, std_h, kurt_h = [], [], []
    rho_innov = []
    # pre-break online vs hist (no-break online for unbroken series too)
    d_mean, lvr, d_acf, ks_pp = [], [], [], []
    pre_vs_hist_dmean = []
    n_broken = 0
    n_total = 0
    innov_pool = []   # standardized innovations to inspect marginal

    for s in iter_series("train"):
        xh = s.x_hist.astype(np.float64)
        xo = s.x_online.astype(np.float64)
        if len(xh) < 50 or len(xo) < 30:
            continue
        n_total += 1
        n_hist.append(len(xh)); n_online.append(len(xo))
        r = ar1(xh)
        rho_h.append(r); std_h.append(xh.std()); kurt_h.append(excess_kurt(xh))
        # AR(1) innovations of historical
        innov = xh[1:] - xh.mean() - r * (xh[:-1] - xh.mean())
        rho_innov.append(ar1(innov))
        if len(innov_pool) < 200000:
            innov_pool.extend(((innov - innov.mean()) / (innov.std() + 1e-12)).tolist())

        sdh = xh.std() + 1e-9
        if s.tau_index is not None and 5 < s.tau_index < len(xo) - 5:
            n_broken += 1
            taufrac.append(s.tau_index / len(xo))
            pre = xo[:s.tau_index]; post = xo[s.tau_index:]
            d_mean.append((post.mean() - pre.mean()) / sdh)
            lvr.append(np.log((post.var() + 1e-9) / (pre.var() + 1e-9)))
            d_acf.append(ar1(post) - ar1(pre))
            ks_pp.append(ks(pre, post))
            pre_vs_hist_dmean.append((pre.mean() - xh.mean()) / sdh)
        elif s.tau_index is None:
            pre_vs_hist_dmean.append((xo.mean() - xh.mean()) / sdh)

    print(f"\nTOTAL series {n_total}  broken {n_broken} ({n_broken/n_total:.3f})")
    print("\n--- segment lengths ---")
    pct("n_hist", n_hist); pct("n_online", n_online)
    print("\n--- historical process ---")
    pct("AR(1) rho_h", rho_h); pct("std_h", std_h); pct("excess_kurt_h", kurt_h)
    pct("AR(1) of innovations", rho_innov)
    ip = np.asarray(innov_pool)
    print(f"  standardized innovation marginal: excess_kurt={(ip**4).mean()-3:.3f}  "
          f"p1={np.percentile(ip,1):.2f} p99={np.percentile(ip,99):.2f}")
    print("\n--- sanity: pre-break(or unbroken) online mean vs hist (should ~0) ---")
    pct("pre_vs_hist_dmean", pre_vs_hist_dmean)
    print("\n--- tau fraction (uniform => flat) ---")
    pct("tau_index/n_online", taufrac, ps=(5, 25, 50, 75, 95))
    print("\n--- BREAK magnitudes (broken series) ---")
    pct("delta_mean (hist-std)", d_mean)
    pct("|delta_mean|", np.abs(d_mean))
    pct("log_var_ratio", lvr)
    pct("delta_AR1", d_acf)
    pct("KS(pre,post)", ks_pp)
    # break-type mix
    dm = np.abs(np.asarray(d_mean)); lv = np.abs(np.asarray(lvr)); da = np.abs(np.asarray(d_acf))
    mean_only = np.mean((dm > 0.25) & (lv < 0.3))
    var_only = np.mean((dm < 0.1) & (lv > 0.4))
    both = np.mean((dm > 0.25) & (lv > 0.4))
    acf_sig = np.mean(da > 0.15)
    subtle = np.mean(np.asarray(ks_pp) < 0.15)
    print(f"\n  type mix: mean-only~{mean_only:.2f}  var-only~{var_only:.2f}  "
          f"both~{both:.2f}  acf-shift~{acf_sig:.2f}  subtle(KS<.15)~{subtle:.2f}")

    # ---- save params for the synthetic generator (bootstrap source) ----
    np.savez(
        "features/dgp_params.npz",
        n_hist=np.asarray(n_hist, np.int32),
        n_online=np.asarray(n_online, np.int32),
        rho_h=np.asarray(rho_h, np.float64),
        kurt_h=np.asarray(kurt_h, np.float64),
        brk_dmean=np.asarray(d_mean, np.float64),
        brk_lvr=np.asarray(lvr, np.float64),
        brk_dacf=np.asarray(d_acf, np.float64),
        brk_ks=np.asarray(ks_pp, np.float64),
        brk_taufrac=np.asarray(taufrac, np.float64),
        frac_broken=np.float64(n_broken / max(n_total, 1)),
    )
    print("\nsaved features/dgp_params.npz")


if __name__ == "__main__":
    main()
