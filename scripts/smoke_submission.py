"""Fast integration smoke for submission/main.py: train on a small subset of
series, infer over the reduced local test, check determinism. Validates the
round-4 GRU integration without the full 10k-series train.

Run: uv run python scripts/smoke_submission.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "submission")

import numpy as np

from sb.data import iter_series
import main as sub

MODEL_DIR = "submission/_model_smoke"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_subset(n=250):
    for i, s in enumerate(iter_series("train")):
        if i >= n:
            break
        tau = s.tau_index if s.tau_index is not None else -1
        yield (s.series_id, s.x_hist, s.x_online, tau)


def test_pairs():
    pairs, lengths = [], []
    for s in iter_series("test"):
        pairs.append((s.x_hist, s.x_online))
        lengths.append(s.n_online)
    return pairs, lengths


def run_infer(pairs):
    gen = sub.infer(iter(pairs), MODEL_DIR)
    next(gen)
    return np.array([float(v) for v in gen], dtype=np.float64)


def main() -> None:
    print("train() on 250 series ...")
    t0 = time.time()
    sub.train(train_subset(250), MODEL_DIR)
    print(f"  done in {time.time()-t0:.1f}s")

    pairs, lengths = test_pairs()
    n = sum(lengths)
    t0 = time.time()
    s1 = run_infer(pairs)
    dt = time.time() - t0
    print(f"infer() {len(pairs)} series / {n} pts in {dt:.1f}s "
          f"({1e6*dt/n:.1f} us/pt -> ~{dt*5e6/n/60:.0f} min full test)")
    print(f"score range [{s1.min():.4f}, {s1.max():.4f}] mean {s1.mean():.4f}")

    s2 = run_infer(pairs)
    md = float(np.max(np.abs(s1 - s2)))
    print(f"determinism max|diff| = {md:.2e}  ({'PASS' if md < 1e-8 else 'FAIL'})")


if __name__ == "__main__":
    main()
