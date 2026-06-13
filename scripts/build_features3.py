"""Build the v2 labeled feature matrix (features3.StreamingDetector).

Identical protocol to build_features.py but with the v2 extractor and output
features/train_features_v3.npz (NOT committed; .npz are LFS-tracked and the
remote proxy rejects LFS pushes -- regenerate with this script instead).

Run:  uv run python scripts/build_features3.py                  # single process
      uv run python scripts/build_features3.py --shard 0 --n-shards 4   # parallel
      uv run python scripts/build_features3.py --merge --n-shards 4     # combine
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np

from sb.data import iter_series
from sb.features3 import StreamingDetector, FEATURE_NAMES, N_FEATURES

OUT = "features"
os.makedirs(OUT, exist_ok=True)


def merge(n_shards: int) -> None:
    Xs, ys, sids, ts = [], [], [], []
    for k in range(n_shards):
        d = np.load(os.path.join(OUT, f"train_features_v3.shard{k}.npz"),
                    allow_pickle=True)
        Xs.append(d["X"]); ys.append(d["y"])
        sids.append(d["series_id"]); ts.append(d["t_online"])
    X = np.vstack(Xs); y = np.concatenate(ys)
    sid = np.concatenate(sids); t_online = np.concatenate(ts)
    order = np.lexsort((t_online, sid))   # deterministic global order
    np.savez(os.path.join(OUT, "train_features_v3.npz"),
             X=X[order], y=y[order], series_id=sid[order],
             t_online=t_online[order], feature_names=np.array(FEATURE_NAMES))
    print(f"merged {n_shards} shards -> X={X.shape}")
    for k in range(n_shards):
        os.remove(os.path.join(OUT, f"train_features_v3.shard{k}.npz"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge(args.n_shards)
        return

    det = StreamingDetector()
    X_chunks, y_chunks, id_chunks, t_chunks = [], [], [], []

    t0 = time.time()
    n_series = 0
    n_points = 0
    for s in iter_series("train"):
        if args.shard >= 0 and (s.series_id % args.n_shards) != args.shard:
            continue
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

    suffix = f".shard{args.shard}" if args.shard >= 0 else ""
    np.savez(
        os.path.join(OUT, f"train_features_v3{suffix}.npz"),
        X=X, y=y, series_id=sid, t_online=t_online,
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"Saved {OUT}/train_features_v3{suffix}.npz in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
