"""Build the per-step PIT (probability-integral-transform) stream — a genuinely
NEW, distribution-free input channel for the sequence members.

MOTIVATION (the documented bottleneck, context.md S6): 68% of breaks are subtle
distribution-only shifts (median KS ~ 0.12, near the two-sample detection limit).
Every existing input is either a calibrated SCALAR moment-summary (KS, ECDF-L1,
windowed skew/kurt) or the Gaussian-parametric raw z = (x - mu_h)/sd_h. The
scalar summaries DISCARD the temporal trajectory; the z-stream assumes Gaussianity
(only mean/var-optimal). Neither feeds the model the distribution-free per-step
position of x_t within the FULL historical law.

The PIT fixes both. For each online point x_t, with F_hat = empirical CDF of the
historical window,
    pit_t = F_hat(x_t) in [0, 1].
Under H0 (no break) pit_t ~ Uniform(0,1) regardless of the underlying law; under
ANY distributional break (mean, variance, skew, kurtosis, multimodality) the pit
sequence departs from uniform in a shape the calibrated KS scalar only summarizes
as a single max-deviation. Feeding the pit TRAJECTORY to causal attention keeps
the temporal shape (clustering of extremes, gradual drift) that the scalars throw
away -- a different, complementary signal by construction, and distribution-free
so it targets the subtle non-Gaussian majority directly.

Channels (all distribution-free, from pit alone):
    pit       F_hat(x_t) in [0,1]              -> any distributional shift
    pit_c     2*pit - 1 in [-1,1]              -> signed mean-direction
    surprise  -log(2*min(pit,1-pit) + 1/n_h)   -> tail emphasis (unbounded at tails)

Aligned to the SAME global (sid, t) order as features/train_features_v4 so the
split-seed-42 mask, labels, and cached logits line up. Built like build_raw_stream.

Run:  uv run python scripts/build_pit_stream.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

import numpy as np

from sb.data import iter_series

OUT = "features/train_pit_stream.npz"
PIT_NAMES = ["pit", "pit_c", "surprise"]


def pit_feats(x_online, x_hist):
    xo = np.asarray(x_online, dtype=np.float64)
    xh = np.asarray(x_hist, dtype=np.float64)
    n = xh.size
    if n == 0:
        pit = np.full(xo.shape, 0.5, dtype=np.float64)
    else:
        xs = np.sort(xh)
        # mid-rank PIT: average of the <= and < ranks, mapped into the open (0,1)
        # interval via the (r + 0.5)/n plotting position so the tails stay finite.
        r_le = np.searchsorted(xs, xo, side="right")
        r_lt = np.searchsorted(xs, xo, side="left")
        pit = (0.5 * (r_le + r_lt)) / n
        pit = np.clip(pit, 0.5 / n, 1.0 - 0.5 / n)
    pit_c = 2.0 * pit - 1.0
    eps = 1.0 / max(n, 1)
    surprise = -np.log(2.0 * np.minimum(pit, 1.0 - pit) + eps)
    return np.stack([pit, pit_c, surprise], axis=1).astype(np.float32)


def main() -> None:
    X_chunks, y_chunks, id_chunks, t_chunks = [], [], [], []
    t0 = time.time()
    n_series = 0
    for s in iter_series("train"):
        T = s.n_online
        if T == 0:
            continue
        X_chunks.append(pit_feats(s.x_online, s.x_hist))
        y = np.zeros(T, dtype=np.int8)
        if s.tau_index is not None and s.tau_index >= 0:
            y[s.tau_index:] = 1
        y_chunks.append(y)
        id_chunks.append(np.full(T, s.series_id, dtype=np.int64))
        t_chunks.append(np.arange(T, dtype=np.int64))
        n_series += 1
        if n_series % 1000 == 0:
            print(f"  {n_series} series  ({time.time()-t0:.0f}s)")

    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    sid = np.concatenate(id_chunks)
    t = np.concatenate(t_chunks)
    order = np.lexsort((t, sid))
    assert np.array_equal(order, np.arange(len(order))), "pit stream order drift"
    np.savez(OUT, X=X, y=y, series_id=sid, t_online=t,
             feature_names=np.array(PIT_NAMES))
    print(f"wrote {OUT}  X={X.shape}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
