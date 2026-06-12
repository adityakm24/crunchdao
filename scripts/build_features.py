"""Build the labeled (series, online-step) feature matrix for training.

Streams every training series through the O(1) StreamingDetector, emitting one
feature row per online observation with label = 1 if the break has already
occurred at/with this step (t >= tau_index), else 0.

Saves to features/train_features.npz:
  X         float32 [N, F]
  y         int8    [N]
  series_id int32   [N]
  t_online  int32   [N]
  feature_names

Run:  uv run python scripts/build_features.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np

from sb.data import iter_series
from sb.features import StreamingDetector, FEATURE_NAMES, N_FEATURES

OUT = "features"
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    det = StreamingDetector()
    X_chunks, y_chunks, id_chunks, t_chunks = [], [], [], []

    t0 = time.time()
    n_series = 0
    n_points = 0
    for s in iter_series("train"):
        det.calibrate(s.x_hist)
        T = s.n_online
        if T == 0:
            continue
        feats = np.empty((T, N_FEATURES), dtype=np.float32)
        for i, x in enumerate(s.x_online):
            feats[i] = det.update(float(x))
        y = np.zeros(T, dtype=np.int8)
        if s.tau_index is not None:
            y[s.tau_index:] = 1
        X_chunks.append(feats)
        y_chunks.append(y)
        id_chunks.append(np.full(T, s.series_id, dtype=np.int32))
        t_chunks.append(np.arange(T, dtype=np.int32))
        n_series += 1
        n_points += T
        if n_series % 1000 == 0:
            el = time.time() - t0
            print(f"  {n_series:5d} series, {n_points:>9,} points, {el:6.1f}s "
                  f"({1e6*el/max(n_points,1):.1f} us/pt)", flush=True)

    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    sid = np.concatenate(id_chunks)
    t_online = np.concatenate(t_chunks)
    print(f"\nMatrix: X={X.shape} y_pos_rate={y.mean():.4f} "
          f"mem={X.nbytes/1e6:.0f}MB")

    np.savez(
        os.path.join(OUT, "train_features.npz"),
        X=X, y=y, series_id=sid, t_online=t_online,
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"Saved {OUT}/train_features.npz in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
