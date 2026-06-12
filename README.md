# ADIA Lab Structural Break — Real-Time Edition

Streaming structural-break detection for the [CrunchDAO competition](https://hub.crunchdao.com/competitions/structural-break-real-time).
At each online observation we emit a calibrated probability in `[0, 1]` that a
permanent break has already occurred. Submissions are scored by **Time-Stratified
AUC (TS-AUC)** — a per-step, cross-sectional AUC averaged over time.

## Approach (Blueprint 1: O(1) streaming meta-features + LightGBM)

1. **Calibrate** a clean baseline from the break-free historical segment
   (mean, std, skew, kurt, lag-1 acf, decile eCDF).
2. **Stream** the online segment, maintaining ~O(1) per-step state: cumulative
   moments, EWMA z-scores, mean & variance CUSUM, Page-Hinkley drift, tail mass,
   cumulative/windowed eCDF distance (KS/L1), and trailing-window (50 & 150)
   two-sample stats (mean z, log-var ratio, skew, kurt, KS, lag-1 acf diff).
3. **Score** each step with a deterministic LightGBM binary classifier trained on
   `(features, step-label)` pairs. **Pure-time features are dropped** — TS-AUC is
   cross-sectional at fixed t, so time carries no rank information and only wastes
   model capacity (this single fix moved us from below-random to a clear win).

## Results (local)

| Model | Held-out VAL TS-AUC (2000 series) | Reduced test (100) |
| --- | --- | --- |
| Provided baseline (EWMA z-score) | — | 0.4806 |
| Round 1 (LightGBM, content-only) | 0.5702 | ~0.52 |
| **Round 2 (multi-scale + time-normalised + TS-AUC select)** | **0.5812** | ~0.53 |

Leaderboard context: top-10 cutoff ≈ 0.6135, #1 ≈ 0.6322.

Two changes drove round 2: (1) **select iterations by TS-AUC**, not pointwise AUC;
(2) **keep `log_t`/`t_over_nhist`** so the model normalises accumulating statistics
by elapsed time via interactions (constant within a step → no cross-sectional leak).

Submission is deterministic (max re-run diff `0.0`) and runs the full 10k-series
test set in ~10 min (15h budget).

## Layout

```
Dataset/                 competition parquet files (X/y train + reduced test)
src/sb/                  reusable library: data loader, metric, feature extractor
scripts/                 eda, build_features, train, train_rank, sweep
submission/main.py       self-contained train()/infer() submission
submission/local_test.py mimics the Crunch runner end-to-end + determinism check
experiments/             EXPERIMENT_LOG.md (full reproducibility trail)
reports/                 EDA tables + figures
```

## Reproduce

```bash
uv run python scripts/eda.py             # EDA tables + figures
uv run python scripts/build_features.py  # 5M-row feature matrix -> features/
uv run python scripts/train.py           # train + held-out/reduced-test TS-AUC
uv run python submission/local_test.py   # end-to-end submission + determinism
```

See `experiments/EXPERIMENT_LOG.md` for the full experiment history, the EDA-driven
break-type taxonomy, the ceiling analysis, and ideas to push past ~0.57.
