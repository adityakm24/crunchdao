# ADIA Lab Structural Break — Real-Time Edition

Streaming structural-break detection for the [CrunchDAO competition](https://hub.crunchdao.com/competitions/structural-break-real-time).
At each online observation we emit a calibrated probability in `[0, 1]` that a
permanent break has already occurred. Submissions are scored by **Time-Stratified
AUC (TS-AUC)** — a per-step, cross-sectional AUC averaged over time.

## Approach — O(1) streaming meta-features + per-series null calibration + LightGBM ensemble

1. **Calibrate** from the break-free historical segment: not just the mean/std/
   skew/kurt/acf/eCDF baseline, but the **per-series null distribution** (loc &
   scale) of every sliding-window statistic, measured at dyadic scales 8…512.
2. **Stream** the online segment with ~O(1) per-step state, reporting each window
   statistic in **calibrated** (per-series z/p) units: cumulative moments, EWMA,
   mean & variance CUSUM, Page-Hinkley, calibrated window mean/var/acf/KS/L1,
   multiscale **scan** maxima + running-max, an **AR(1)-prewhitened** CUSUM/EWMA
   bank, calibrated **derived streams** (|z|, Δz, median-crossing), and calibrated
   **higher moments** (skew/kurt). Why calibration: TS-AUC ranks series
   *cross-sectionally at a fixed step*, so a raw |z|=3 from a wandering series must
   not outrank |z|=3 from a quiet one — per-series null normalisation removes that
   scale heterogeneity. This was the round-3 breakthrough.
3. **Score** with a deterministic **ensemble** (equal mean-logit): 3 LightGBM
   members on the v2 feature subset + 1 on the v4 superset + a logistic-regression
   member (different model class → real decorrelation). Pure-time features stay
   dropped; iterations are selected by a custom TS-AUC `feval`.

## Results (held-out VAL, 2000-series internal split)

| Model | VAL TS-AUC |
| --- | --- |
| Provided baseline (EWMA z-score) | ~0.4806 |
| Round 1 (LightGBM, content-only) | 0.5702 |
| Round 2 (multi-scale + time-normalised + TS-AUC select) | 0.5812 |
| Round 3 — v2 per-series null calibration (single model) | 0.5988 |
| **Round 3 — final ensemble (4 GBT + logistic, mean-logit)** | **0.6041** |

Leaderboard context: top-10 cutoff ≈ 0.6135, #1 ≈ 0.6322 — the final model is
~0.9 pt under top-10. Round 3 added **+2.3 pts** over round 2 by per-series
empirical-null calibration of the streaming statistics; ensembling correlated
GBTs (corr 0.97–0.99) then caps out, so the logistic member's model-class
diversity provides the last lift. Submission is deterministic (smoke-test re-run
diff `0.0`) and runs the full 10k-series test in ~48 min (15 h budget).

## Layout

```
Dataset/                 competition parquet files (X/y train + reduced test)
src/sb/                  reusable library: data loader, metric, feature extractor
scripts/                 eda, build_features, train, train_rank, sweep
artifacts/models/        versioned model checkpoints (model_001, model_002, ...)
submission/main.py       self-contained train()/infer() submission
submission/local_test.py mimics the Crunch runner end-to-end + determinism check
experiments/             EXPERIMENT_LOG.md (full reproducibility trail)
reports/                 EDA tables + figures
```

## Reproduce (round-3 final model)

```bash
# build the v4 feature matrix (2 shards at a time — 4 concurrent OOMs 15 GB):
for k in 0 1; do uv run python scripts/build_features4.py --shard $k --n-shards 4 & done; wait
for k in 2 3; do uv run python scripts/build_features4.py --shard $k --n-shards 4 & done; wait
uv run python scripts/build_features4.py --merge --n-shards 4
# train the ensemble members, pick the blend, build & verify the submission:
uv run python scripts/train2.py --model-id model_003 --features features/train_features_v2.npz --no-retrain
uv run python scripts/ensemble_search.py     # diverse-blend selection
uv run python scripts/final_blend.py         # logistic member + blend weight
uv run python scripts/make_submission3.py    # regenerate submission/main.py
uv run python submission/local_test.py       # end-to-end + determinism (1e-8)
```

> The big parquet/npz are Git LFS; on a fresh clone fetch them from the GitHub
> media CDN (see `context.md` §3). Feature matrices are git-ignored — regenerate.

See `context.md` (orientation, findings, future directions), `experiments.md`
(per-iteration ledger), and `experiments/EXPERIMENT_LOG.md` for the full history,
break-type taxonomy, ceiling analysis, and the ranked next steps to push past 0.61.
