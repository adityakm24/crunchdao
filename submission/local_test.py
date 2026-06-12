"""Mimic the Crunch runner for submission/main.py.

  1. train() on all 10k training series (deterministic LightGBM).
  2. infer() over the reduced local test set, collecting yielded scores.
  3. Compute TS-AUC against y_test.reduced.
  4. Determinism check: re-run infer() and assert max|diff| < 1e-8.

Run:  uv run python submission/local_test.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "submission")

import numpy as np
import pandas as pd

from sb.data import iter_series, load_test_targets
from sb.metric import ts_auc_grouped
import main as sub

MODEL_DIR = "submission/_model"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_datasets():
    for s in iter_series("train"):
        tau = s.tau_index if s.tau_index is not None else -1
        yield (s.series_id, s.x_hist, s.x_online, tau)


def test_datasets():
    """(x_historical, x_online) pairs in reduced-test order, plus bookkeeping."""
    ids, lengths = [], []
    pairs = []
    for s in iter_series("test"):
        ids.append(s.series_id)
        lengths.append(s.n_online)
        pairs.append((s.x_hist, s.x_online))
    return pairs, ids, lengths


def run_infer(pairs):
    gen = sub.infer(iter(pairs), MODEL_DIR)
    next(gen)  # consume readiness yield
    return np.array([float(v) for v in gen], dtype=np.float64)


def main() -> None:
    print("Training submission model on 10k series ...")
    t0 = time.time()
    sub.train(train_datasets(), MODEL_DIR)
    print(f"  train() done in {time.time()-t0:.1f}s")

    pairs, ids, lengths = test_datasets()
    n_points = sum(lengths)
    print(f"Inferring over reduced test: {len(ids)} series, {n_points} points ...")
    t0 = time.time()
    scores = run_infer(pairs)
    dt = time.time() - t0
    print(f"  infer() done in {dt:.1f}s ({1e6*dt/n_points:.1f} us/point) "
          f"-> est. full 10k-series test ~ {dt*5_000_000/n_points/60:.1f} min")
    assert len(scores) == n_points, (len(scores), n_points)

    # assemble (id, time) index in the same order infer produced
    rows = []
    for sid, L in zip(ids, lengths):
        rows.extend((sid, t) for t in range(L))
    idx = pd.MultiIndex.from_tuples(rows, names=["id", "time_online"])
    pred = pd.Series(scores, index=idx, name="prediction")

    # targets
    yt = load_test_targets().reset_index().sort_values(["id", "time"])
    yt["time_online"] = yt.groupby("id").cumcount()
    key = yt.set_index(["id", "time_online"])["target"]
    y = key.loc[rows].to_numpy().astype(np.int64)
    t_online = np.array([t for _, t in rows], dtype=np.int64)

    tsauc = ts_auc_grouped(t_online, y, scores)
    print(f"\n>>> Submission reduced-test TS-AUC : {tsauc:.4f}  (baseline EWMA 0.4806)")

    print("Determinism re-check ...")
    scores2 = run_infer(pairs)
    md = float(np.max(np.abs(scores - scores2)))
    print(f"  max|diff| over re-run = {md:.2e}  ({'PASS' if md < 1e-8 else 'FAIL'} @1e-8)")


if __name__ == "__main__":
    main()
