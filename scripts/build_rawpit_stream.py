"""Merge the raw z-stream and the PIT stream into one 6-channel learned-detector
input (train_rawpit_stream.npz), for the maximally-decorrelated "only" member.

Both inputs are already aligned to the v4 global (sid, t) order, so this is a
pure column concatenate — no series re-iteration. The combined channel set gives
causal attention BOTH the Gaussian-parametric trajectory (z, z**2, |z|, optimal
for mean/variance breaks) AND the distribution-free trajectory (pit, pit_c,
surprise, optimal for the subtle non-Gaussian majority) with NO calibrated
features, i.e. the richest different-INPUT detector we can feed the model.

Run:  uv run python scripts/build_rawpit_stream.py
"""
from __future__ import annotations

import numpy as np

RAW = "features/train_raw_stream.npz"
PIT = "features/train_pit_stream.npz"
OUT = "features/train_rawpit_stream.npz"


def main() -> None:
    r = np.load(RAW, allow_pickle=True)
    p = np.load(PIT, allow_pickle=True)
    assert np.array_equal(r["series_id"], p["series_id"]), "sid mismatch"
    assert np.array_equal(r["t_online"], p["t_online"]), "t mismatch"
    X = np.concatenate([r["X"].astype(np.float32), p["X"].astype(np.float32)], axis=1)
    names = np.array([str(n) for n in r["feature_names"]]
                     + [str(n) for n in p["feature_names"]])
    np.savez(OUT, X=X, y=r["y"], series_id=r["series_id"], t_online=r["t_online"],
             feature_names=names)
    print(f"wrote {OUT}  X={X.shape}  chans={list(names)}")


if __name__ == "__main__":
    main()
