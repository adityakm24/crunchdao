# Context & Handoff — ADIA Lab Structural Break (Real-Time Edition)

> Read this first if you are picking up the research. It tells you what the
> competition is, what we built, what we *know*, where the ceiling is, and the
> highest-leverage things to try next. The blow-by-blow experiment ledger lives
> in `experiments.md` (and `experiments/EXPERIMENT_LOG.md`).

---

## 1. The competition in one paragraph

CrunchDAO **structural-break-real-time** (`hub.crunchdao.com/competitions/structural-break-real-time`).
Each task is a single univariate series split into a **historical** segment
(`period==1`, guaranteed break-free) and an **online** segment (`period==2`).
At *every* online observation we must emit a probability in `[0, 1]` that a
**permanent structural break has already occurred at or before this step**. It is
a streaming/online problem: you see points one at a time, cannot look ahead, and
cannot re-read the series. The score is **TS-AUC (Time-Stratified AUC)**.

### TS-AUC — the only thing that matters
At each online step `t`, take every series' score at step `t`, and compute a
**cross-sectional AUC** that ranks broken-vs-not-broken series *at that step*.
Average those per-step AUCs over `t`, weighted by `n_pos(t) * n_neg(t)`.

Consequences that drove every design decision:
- **It is a ranking metric across series at a fixed step**, not a per-series
  classification. Absolute calibration per series is irrelevant; only the
  cross-sectional *ordering* at each `t` matters.
- **Any feature that is constant across series at a fixed `t` carries zero rank
  information** (e.g. the raw time index `t`). Feeding such features as a primary
  signal *hurts* — see the "time-feature trap" below.
- Per-step weight peaks around online steps **~150–250** and tapers after ~600,
  so early-to-mid detection dominates the score.

---

## 2. Current submission (what is live)

- **Submitted:** submission **#1** to project `chinchilla` (userId 13086).
  Dashboard: `https://hub.crunchdao.com/competitions/structural-break-real-time/projects/13086/chinchilla/submissions/1`
- **Model:** `submission/main.py` — self-contained `train()` + `infer()`.
  - O(1)-per-step streaming feature extractor (`StreamingDetector`, 62 raw
    features) feeding a deterministic **LightGBM** binary classifier
    (`num_leaves=31, lr=0.03, min_data_in_leaf=1000, ff=0.8, bagging=0.8`,
    `288` rounds, `deterministic=True`).
  - Features actually used = 62 minus dropped `{t, log_n_hist, chi2*, acf2*}`.
  - Positives down-weighted near the break (`weight = clip(elapsed/50, 0.2, 1)`).
- **Validated locally with `crunch test` before push:** train ~2.5 min, infer on
  reduced test, **determinism check PASSED** (tolerance 1e-8). Memory ~1.6 GB.
- **Held-out VAL TS-AUC ≈ 0.5812** (2000-series internal split).
  Baseline EWMA ≈ 0.4806. Round-1 model 0.5702.

> NOTE: `crunch push` uploads `main.py` *and* whatever is in `resources/`. After
> `crunch test`, `resources/lgbm.txt` contained the locally trained model, so it
> was uploaded too. The cloud re-runs our deterministic `train()` regardless, so
> this is harmless — but be aware the resources model is shipped.

---

## 3. How to work in this repo

Environment: macOS (Apple Silicon), **`uv`** venv, **Python 3.13**. Always use
`uv run` / `uv add`. Core deps: numpy 2.4, pandas 3.0, scipy 1.17, lightgbm 4.6,
pyarrow, matplotlib.

```
Dataset/                 competition parquet (X/y train + reduced test) — symlink/local copy
src/sb/                  library: data.py (loader), metric.py (fast TS-AUC), features.py (extractor)
scripts/                 eda, build_features, train, train_rank, sweep{,2,3}, ablation, make_submission
artifacts/models/        versioned iteration artifacts: model_001, model_002, ...
submission/main.py       self-contained train()/infer() — THE submission artifact
submission/local_test.py mimics the Crunch runner + determinism check
experiments/EXPERIMENT_LOG.md  full reproducibility trail
structural-break-real-time-chinchilla/  the crunch project (created by `crunch setup`); main.py copied here
```

### `src/sb/features.py` is the single source of truth
`submission/main.py` is **auto-generated** from it by
`scripts/make_submission.py` (it inlines the extractor so the cloud needs no
project imports). **Edit `src/sb/features.py`, then regenerate** — never hand-edit
`main.py`, or train/serve will drift.

### Typical loop
```bash
uv run python scripts/eda.py             # EDA tables + figures (reports/)
uv run python scripts/build_features.py  # ~5M-row feature matrix -> features/
uv run python scripts/train.py --model-id model_00X  # save iteration model under artifacts/models/
uv run python scripts/make_submission.py # regenerate submission/main.py from src/sb/features.py
uv run python submission/local_test.py   # end-to-end + determinism
```

### Iteration artifact rule (must follow)
- Every meaningful training run gets a new model id folder:
  `artifacts/models/model_001`, `model_002`, `model_003`, ...
- `scripts/train.py --model-id <id>` writes `artifacts/models/<id>/lgbm.txt`.
- It also mirrors the same file to `model/lgbm.txt` so existing scripts stay compatible.
- Current best checkpoint is stored at `artifacts/models/model_001/lgbm.txt`.

### Submitting again
```bash
cd structural-break-real-time-chinchilla
cp ../submission/main.py main.py
uv run crunch test          # ALWAYS run this first — catches interface/determinism errors
uv run crunch push -m "..." # token already stored in .crunchdao/token
```

---

## 4. Data facts (from `scripts/eda.py`)

- 10,000 training series; **49.7% contain a break** (balanced, p≈0.5).
- Historical length ~Uniform[1000, 5000] (median ~3000).
- Online length ~Uniform[10, 999] (median ~500).
- Break position `tau_frac` ~Uniform[0, 1] — a break is equally likely anywhere
  in the online segment.
- **Break-type taxonomy** (of series with measurable pre/post windows):
  - mean-shift (|Δmean|>0.5):       **5.7%**
  - variance change (std ratio>1.5): **14.2%**
  - autocorrelation (|Δacf1|>0.2):   **20.6%**
  - **subtle / distribution-only:    68.2%** (KS(pre,post) median ≈ 0.12)

**Modelling implication:** mean-only detectors (the EWMA baseline) are mismatched
— <6% of breaks are mean shifts. You must capture variance, distribution shape,
and serial-dependence changes. That is exactly what `StreamingDetector` does.

---

## 5. The two findings that mattered most

1. **The time-feature trap.** Including the raw time index `t` (or `log_n_hist`)
   as a feature made the model rank *below random* on TS-AUC (TEST 0.4465) even
   while pointwise AUC looked great (0.699). TS-AUC is cross-sectional at fixed
   `t`, so `t` is constant across series and pure noise for ranking; the GBT
   wasted capacity on it. **Drop raw `t` / `log_n_hist`.**

2. **But keep `log_t` and `t_over_nhist` — when selecting by TS-AUC.** These let
   the model *normalise* the accumulating statistics (CUSUM, Page-Hinkley grow
   with `t`) via interactions. They are still constant within a step (no leak),
   but they recalibrate growing features. Re-adding them moved 0.5645 → 0.5776.
   This only works in combination with the next point.

3. **Select boosting iterations by TS-AUC, not pointwise AUC.** Pointwise AUC
   peaks early then fits cross-time ordering the metric ignores. We added a custom
   LightGBM `feval` using the fast Mann-Whitney TS-AUC in `src/sb/metric.py`. This
   is a *correctness* fix and was half of the round-2 gain.

Together: 0.5702 (round 1) → **0.5812** (round 2).

---

## 6. The honest ceiling (read before optimizing)

We are on a **plateau around 0.57–0.58** for this architecture (single GBT +
handcrafted streaming features). A hyperparameter sweep and ~6 feature-set
variants all cluster there → we are **signal/feature-limited, not
capacity-limited**. Why the ceiling is real:

- `tau ~ Uniform` over the online segment, so at any step a large fraction of
  "broken" series broke *just before* `t` and are near-undetectable → this caps
  the achievable per-step AUC structurally.
- 68% of breaks are statistically subtle (small KS).
- Pointwise AUC (~0.69) ≫ TS-AUC (~0.58): the residual is cross-time ordering the
  metric deliberately ignores — not exploitable.

**Leaderboard context:** top-10 cutoff ≈ **0.6135**, #1 ≈ 0.6322. We need ~+3 pts
of *genuine* cross-sectional signal, which almost certainly requires a different
representation, not more tuning of the current one.

---

## 7. Highest-leverage directions to try next (ranked)

1. **Stronger subtle-shift distance features.** Replace/augment decile-KS with
   **energy distance** or **MMD** between the trailing window and the historical
   eCDF. These dominate KS for the subtle distributional breaks that are 68% of
   the data. Maintainable approximately in streaming form with fixed reference
   quantiles. *Most promising given the taxonomy.*
2. **Dependence-break features.** Higher-lag autocorrelation, partial-acf, and
   **spectral-slope change** (Welch/periodogram on a trailing window). 20.6% of
   breaks are acf changes and our acf features rank well — there is headroom.
3. **Per-break-type expert models + gating.** Train specialists (mean / variance
   / acf / distribution) and a gate; the 2025 winners leaned on epistemic
   diversity. Ensemble diversity is the most reliable way off a single-model
   plateau.
4. **Step-importance sample weighting / focal loss.** Weight rows by
   `n_pos(t)*n_neg(t)` (the metric's own weight) and/or focal loss on hard rows
   near `tau`. Cheap to try; aligns training loss with the score.
5. **TSFM drift embeddings (heavy).** Chronos / Moirai / TimesFM trailing-window
   embeddings as drift features. Must bundle weights (cloud has no internet on the
   competition accelerator); big lift, big effort — do last.

### Cheap experiments to run first
- Energy-distance trailing-window feature (idea 1) — add to `WindowDist.stats`.
- Step-importance weighting (idea 4) — one-line change in `train()` weight vector.
- A second GBT on a *disjoint* feature subset, averaged with the current one
  (cheapest form of idea 3).

---

## 8. Gotchas / constraints

- **Determinism is enforced** (`crunch test` re-runs infer on 10% at 1e-8
  tolerance). No RNG in the feature path; LightGBM `deterministic=True,
  force_row_wise=True`. Keep it that way.
- **No look-ahead, no series re-read** in `infer()`; everything must be O(1)
  per step. Full 10k-series test runs in ~10 min — comfortably inside the budget.
- **Forbidden-library check** on push. Stick to numpy/scipy/lightgbm/joblib.
- The competition accelerator (if you go the GPU/TSFM route) has **no internet** —
  bundle any model weights as a Kaggle/crunch resource, don't pip-install at runtime.
- `submission/main.py` is generated — edit `src/sb/features.py` + regenerate.
