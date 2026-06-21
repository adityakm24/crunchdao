"""(B) probe 2 — Chronos FORECAST-RESIDUAL break signal (the steelman of B1).

Embedding-distance was dead (instance-norm blind to level shifts). Forecasting is
the other TSFM use and DOES see level shifts: forecast the online segment from the
break-free historical context; a break spikes |actual - forecast| / forecast-spread.

Fixed historical context (no post-break adaptation) -> horizon H online steps.
Per online step residual vs label(t>=tau), scored with TS-AUC, vs mean_shift on the
same rows. Only the first H online steps are testable, so this measures whether the
signal EXISTS at all. If <=0.55, the TSFM avenue is conclusively closed.

Run: HF_HUB_OFFLINE=1 uv run python scripts/probe_chronos_fcst.py --n 4000
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import torch

from sb.data import iter_series
from sb.metric import ts_auc_grouped

H = 48
CTX = 512
MODEL = "models/chronos-bolt-small"
QL = [0.1, 0.5, 0.9]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cpu")

    ctxs, metas = [], []   # meta: (sid, mu_h, sd_h, x_online[:H], tau)
    t0 = time.time()
    n = 0
    for s in iter_series("train"):
        if n >= args.n:
            break
        if s.n_hist < 64 or s.n_online < H:
            continue
        xh = np.asarray(s.x_hist, dtype=np.float64)
        xo = np.asarray(s.x_online[:H], dtype=np.float64)
        ctxs.append(torch.tensor(xh[-CTX:], dtype=torch.float32))
        metas.append((s.series_id, xh.mean(), xh.std() + 1e-9, xo,
                      s.tau_index if s.tau_index is not None else -1))
        n += 1
    print(f"[collect] {n} series, H={H}  {time.time()-t0:.0f}s")

    t0 = time.time()
    q_all = []
    bs = 256
    for i in range(0, len(ctxs), bs):
        q, _ = pipe.predict_quantiles(ctxs[i:i + bs], prediction_length=H,
                                      quantile_levels=QL)
        q_all.append(q.detach().numpy())   # (B, H, 3)
    Q = np.concatenate(q_all, axis=0)
    print(f"[forecast] Q {Q.shape}  {time.time()-t0:.0f}s")

    t_r, y_r, resid, mshift = [], [], [], []
    for k, (sid, mu_h, sd_h, xo, tau) in enumerate(metas):
        lo, med, hi = Q[k, :, 0], Q[k, :, 1], Q[k, :, 2]
        spread = np.maximum(hi - lo, 1e-6)
        for t in range(H):
            r = abs(xo[t] - med[t]) / spread[t]            # normalized forecast residual
            t_r.append(t); y_r.append(1 if (tau >= 0 and t >= tau) else 0)
            resid.append(r)
            mshift.append(abs(xo[t] - mu_h) / sd_h)         # naive level-shift baseline
    t = np.asarray(t_r); y = np.asarray(y_r)
    resid = np.asarray(resid); mshift = np.asarray(mshift)
    print(f"[rows] {len(t)} rows, pos_rate={y.mean():.3f}")

    # cumulative running-max of residual (a break stays broken -> integrate evidence)
    # per series, within the H-block
    cmax = np.empty_like(resid)
    i = 0
    for k in range(len(metas)):
        cmax[i:i + H] = np.maximum.accumulate(resid[i:i + H])
        i += H

    print("\n[TS-AUC]")
    print(f"  fcst_resid          = {ts_auc_grouped(t, y, resid):.4f}")
    print(f"  fcst_resid_cummax   = {ts_auc_grouped(t, y, cmax):.4f}")
    print(f"  mean_shift (base)   = {ts_auc_grouped(t, y, mshift):.4f}")
    print("\n>>> TSFM forecast-residual is real signal only if >0.55 and beats mean_shift")


if __name__ == "__main__":
    main()
