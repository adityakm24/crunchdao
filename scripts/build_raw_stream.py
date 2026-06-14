"""Build the RAW standardized online stream for a different-input neural member.

The round-4/5 recurrent members all consume the SAME 152 calibrated features as
the GBTs, so they are input-bottlenecked (the LSTM's rank-corr vs base 0.883 is
even higher than the GRU's 0.847 despite a different architecture). The genuine
decorrelation lever is a different INPUT: feed a GRU only the historically-
standardized raw value z = (x - mu_h) / sd_h and a couple of pointwise
transforms, with NO calibrated features. The recurrence integrates the raw
signal over time; mean breaks show as z drifting, variance breaks as z**2 rising,
serial-dependence breaks via the recurrence. This is complementary to the
calibrated base by construction.

Output matrix is aligned to the SAME global order as features/train_features_v4
(series in id order, online steps in time order -> lexsort(t, sid)), so the
split-seed-42 val_mask and labels line up with the cached base/seq logits.

Raw pointwise features (all from z, clipped to +-CLIP):
    z      level     -> mean shift
    z**2   energy    -> variance change
    |z|    magnitude -> heavy tails / scale

Run:  uv run python scripts/build_raw_stream.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np

from sb.data import iter_series

OUT = "features/train_raw_stream.npz"
CLIP = 8.0
RAW_NAMES = ["z", "z2", "absz"]


def raw_feats(x_online, mu, sd):
    z = (np.asarray(x_online, dtype=np.float64) - mu) / sd
    np.clip(z, -CLIP, CLIP, out=z)
    return np.stack([z, z * z, np.abs(z)], axis=1).astype(np.float32)


def main() -> None:
    X_chunks, y_chunks, id_chunks, t_chunks = [], [], [], []
    t0 = time.time()
    n_series = 0
    for s in iter_series("train"):
        T = s.n_online
        if T == 0:
            continue
        x = np.asarray(s.x_hist, dtype=np.float64)
        if x.size:
            mu = float(x.mean())
            sd = max(float(x.std(ddof=1)) if x.size > 1 else 1.0, 1e-9)
        else:
            mu, sd = 0.0, 1.0
        X_chunks.append(raw_feats(s.x_online, mu, sd))
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
    # already in (sid, t) order since iter_series is id-sorted and per-series
    # online steps are time-ordered; assert to be safe.
    order = np.lexsort((t, sid))
    assert np.array_equal(order, np.arange(len(order))), "raw stream order drift"
    np.savez(OUT, X=X, y=y, series_id=sid, t_online=t,
             feature_names=np.array(RAW_NAMES))
    print(f"wrote {OUT}  X={X.shape}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
