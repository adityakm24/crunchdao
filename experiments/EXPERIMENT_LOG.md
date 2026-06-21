# Experiment Log — ADIA Lab Structural Break Real-Time Challenge

Goal: beat the public leaderboard on **TS-AUC** (Time-Stratified AUC) for the
2026 real-time structural-break detection competition (CrunchDAO).

> **Status (2026-06-21):** rounds 4–11 are ledgered in `experiments.md` /
> `context.md`. Latest shipped = **submission #9** (round 11, +2-seed causal-
> attention member), real public **0.6049 (60.49%)** — new PB, **+0.0053** over the
> prior best. VAL→real offset ≈ **−0.014**; gap to #1 (0.6322) ≈ 0.027.

Environment: macOS (Apple Silicon), `uv` venv, Python 3.13.13 (cloud runtime is
3.13.x). Core deps: numpy 2.4, pandas 3.0, scipy 1.17, scikit-learn 1.9,
lightgbm 4.6, pyarrow 24, matplotlib 3.11.

Data (`Dataset/`):
- `X_train.parquet`  35,036,464 rows, MultiIndex (id,time), cols [value, period].
  `period==1` historical (break-free), `period==2` online.
- `y_train.parquet`  per-(id,time) `target` over the ONLINE portion = ideal
  step label (0 before break, 1 from break on).
- `y_train_index.parquet`  per-id `tau_index` (0-based pos in online, -1=no break)
  and `tau` (absolute time, -1=no break). tau = n_hist + tau_index.
- Reduced local test: `X_test.reduced.parquet` (100 series, ids>=10000),
  `y_test.reduced.parquet` (per-step target), `y_test_index.reduced.parquet`.

---

## EDA findings (2026-06-12) — `scripts/eda.py`

- 10,000 training series; **49.7% have a break**, 50.3% none (balanced, matches p=0.5).
- Historical length ~Uniform[1000, 5000] (median 2998).
- Online length ~Uniform[10, 999] (median 502).
- Break position `tau_frac` ~Uniform[0,1] (mean 0.488) — break is equally likely anywhere.
- **Break-type taxonomy** (breaks with measurable pre/post windows):
  - mean-shift (|Δmean|>0.5):        **5.7%**
  - variance (std ratio >1.5x):      **14.2%**
  - autocorrelation (|Δacf1|>0.2):   **20.6%**
  - subtle / distribution-only:      **68.2%**
  - KS(pre,post) median 0.118, p90 0.296 → most breaks are statistically *subtle*.
- **TS-AUC weight** `n_pos(t)*n_neg(t)` peaks around online steps ~150–250 and
  tapers after ~600. Early-to-mid online detection dominates the score.

### Implications for modelling
1. Mean-only detectors (baseline EWMA) are mismatched: <6% of breaks are mean shifts.
   Must capture variance, distribution shape, and serial-correlation changes.
2. TS-AUC is a **cross-sectional ranking** metric → outputs must be globally
   calibrated across series. A supervised classifier on (features, step-label)
   naturally produces calibrated cross-sectional scores (Blueprint 1).
3. Must be **O(1) per step** (10k series x up to 1k steps = up to 10M scores,
   15h/week budget). All features maintained with constant-time recursion.

---

## Experiments

### EXP-000 Baseline (provided): streaming EWMA z-score, tanh squash
- ALPHA=0.05, KAPPA=3.0. Mean-only.
- Reported local TS-AUC in quickstarter notebook: **0.4806** (≈ random).
- Status: reference point to beat.

### EXP-001a: O(1) tracker + LightGBM, WITH time features (t, log_t, ...)
- 27 features incl. absolute time. LightGBM binary, 5.0M rows.
- Pointwise val AUC 0.699 but **VAL TS-AUC 0.5687, TEST TS-AUC 0.4465** (below random!).
- Diagnosis: top features were `t`, `log_t`, `t_over_nhist` — pure time. TS-AUC is
  cross-sectional at fixed t, so time is constant across series and useless; it
  diverted model capacity. Classic trap noted in the docs.

### EXP-001b: drop pure-time features (t, log_t, log_n_hist, t_over_nhist)
- 23 content-only features. best_iter=224.
- **VAL TS-AUC 0.5605, reduced TEST TS-AUC 0.5138** (baseline 0.4806). First win.
- Top gain: cusum_absmax, online_acf1, tail3_excess, cum_kurt_diff, ks_stat,
  cum_skew, acf1_diff, tail2_excess, cum_kurt, cum_logvar_ratio.
- Weakness: cumulative features dilute post-break signal (mix pre/post online points).

### EXP-002: add variance CUSUM + Page-Hinkley + trailing-window(50) dist stats
- 34 features. Pointwise AUC 0.686. **VAL TS-AUC 0.5613**, reduced TEST 0.4940.
- cusum_var_absmax became the dominant feature. Val TS-AUC flat vs EXP-001b.

### EXP-003: replace raw growing accumulators with sqrt(n)-normalized versions
- Dropped raw cusum_*/ph_mean, kept only /sqrt(n) versions (t-invariant theory).
- **VAL TS-AUC 0.5558** (worse). Conclusion: raw accumulators carry genuine
  within-step magnitude signal; keep BOTH raw and normalized. LightGBM can't use
  t directly anyway, so raw growth does not actively hurt within-step ranking.

### EXP-004: WindowDist refactor — two windows (50,150) w/ localized acf, 19-knot eCDF
- 38 features (per-window: mean_z, logvar, skew, kurt, ks, acf1_diff). Keep raw+norm accumulators.
- Pointwise AUC 0.694. **VAL TS-AUC 0.5702 (best)**, reduced TEST 0.5132.
- Note: window features rank low in gain (corr. with cusum_var); cusum_var_absmax still top.
- Gap pointwise(0.694) >> TS-AUC(0.57): pointwise classifier still exploits cross-time order.

### EXP-005: lambdarank grouped by online step (REJECTED)
- group = online-step index; binary relevance; objective=lambdarank.
- **VAL TS-AUC 0.5223** (worse); diverged after iter 1 (auc 0.558 -> 0.489 by iter 50).
- Reason: LambdaRank optimizes NDCG (top-of-list emphasis), not full-ranking AUC.
  Misaligned with TS-AUC. Pointwise binary remains the right objective.

### EXP-006: hyperparameter sweep (pointwise) — `scripts/sweep.py`
- leaves{31,63,127,255} x min_data{200,500,1000} x lr{.03,.05} x ff{.6,.8}.
- All configs cluster at **TS-AUC 0.566-0.570**. Best 0.5703 (leaves63,md500,lr.03,ff.6),
  tied with EXP-004 0.5702. Conclusion: FEATURE/SIGNAL-LIMITED, not capacity-limited.

---

## Current best & submission

- **Model: EXP-004 pointwise LightGBM, 34 content features.**
  - Held-out (2000 series) **VAL TS-AUC = 0.5702**.
  - Reduced local test (100 series) TS-AUC = 0.51-0.52 (high variance, tiny set).
  - vs provided baseline EWMA = 0.4806.
- **Self-contained submission: `submission/main.py`** (train + infer, no project imports).
  - End-to-end local runner `submission/local_test.py`:
    train() 84s; infer() 116 us/point -> est. ~9.6 min for the full 10k-series test
    (<< 15h budget); reduced-test TS-AUC 0.5182; **determinism max|diff| = 0.0 (PASS @1e-8)**.

### Why TS-AUC is hard here (ceiling analysis)
- tau ~ Uniform over the online segment, so at any step t a large share of "broken"
  series broke only just before t and are near-undetectable -> caps per-step AUC.
- 68% of breaks are statistically subtle (KS(pre,post) median 0.12).
- Pointwise AUC (~0.694) >> TS-AUC (~0.57): residual gap is cross-time ordering the
  metric ignores; not exploitable for score.

---

## Round 2 (targeting the leaderboard: top-10 = 61.35-63.22% TS-AUC)

Leaderboard context: #1 intermediate-pavel 63.22; #5 farukcan-saglam 62.51 and
#7 brandao 61.99 are the 2025 winners. Top-10 cutoff ~61.35.

### EXP-007: multi-scale windows (25/50/100/200) + var-of-diffs + multi-k CUSUM
- 57 features. With pointwise-AUC early stop: val 0.564 (WORSE than EXP-004).

### *** FIX: select iterations by TS-AUC, not pointwise AUC ***
- Added a custom LightGBM `feval` computing TS-AUC on the val set; fast
  Mann-Whitney rank implementation in `sb.metric` keeps the callback cheap.
- This is a correctness fix: pointwise AUC peaks early then fits cross-time noise
  the metric ignores; TS-AUC selection picks a genuinely better iteration.

### *** BREAKTHROUGH: re-add time features (log_t, t_over_nhist) ***  `scripts/ablation.py`
- Earlier "drop ALL time features" was confounded by pointwise selection. With
  TS-AUC selection, time features let the model NORMALISE the accumulating
  statistics via interactions (constant within a step, so no cross-sectional leak).
- Ablation (TS-AUC select):
  - content-only (drop time): 0.5645
  - **content + log_t + t_over_nhist: 0.5776**  <- big jump
  - content + ALL time (incl t, log_n_hist): 0.5747 (raw t / log_n_hist add noise)
- Decision: drop only {t, log_n_hist}; keep log_t & t_over_nhist.

### EXP-006b capacity sweep (TS-AUC select) `scripts/sweep2.py`
- Shallower + more regularised is best: **leaves31, lr.03, md1000 -> 0.5796**
  (deep trees / leaves255 overfit; ff.5 hurts).

### EXP-006c weighting `scripts/sweep3.py`
- Down-weight near-undetectable fresh-break positives (weight = clip(elapsed/50, 0.2, 1)):
  **0.5812** (best). scale_pos_weight and aggressive feature_fraction hurt.

### EXP-008: chi-square dist-distance + lag-2 acf (REJECTED)
- Added per-window chi2 + cumulative chi2 + online_acf2/acf2_diff: val 0.5777 (WORSE
  than 0.5812). The model consistently prefers a leaner feature set; dropped them.

---

## CURRENT BEST (round 2) & submission

- Feature set: EXP-007 (multi-scale windows 25/50/100/200, multi-k CUSUM, var CUSUM,
  Page-Hinkley, var-of-diffs, EWMA, cumulative moments/eCDF) + log_t + t_over_nhist.
  Drop {t, log_n_hist, chi2*, acf2*}.
- Model: LightGBM binary, num_leaves=31, lr=0.03, min_data=1000, ff=0.8, bagging=0.8,
  elapsed-weighting (ramp=50), iterations selected by TS-AUC.
- **Held-out VAL TS-AUC = 0.5812** (was 0.4806 baseline, 0.5702 round-1).
- Reduced local test (100 series, noisy): 0.524-0.532.
- Submission `submission/main.py` (auto-generated by `scripts/make_submission.py`):
  train 148s; infer 124 us/point -> ~10 min full test; **determinism diff = 0.0 (PASS)**.

### Honest gap to leaderboard
- Held-out VAL 0.581 vs top-10 cutoff 0.6135. Need ~+3.2 pts. We are at a plateau
  for this single-GBT + handcrafted-feature architecture (capacity & many feature
  variants all cluster 0.57-0.58).

### Ideas not yet tried (to break past ~0.58)
1. Multi-scale CUSUM/GLR mixtures (more allowance values k); adaptive windows.
2. Higher-lag autocorrelation & spectral-slope change features (for dependence breaks).
3. Per-break-type expert models + gating (epistemological diversity, 2025 Alphabot).
4. TSFM embeddings (Chronos/Moirai/TimesFM) as drift features (bundle weights; heavy).
5. Sample weighting by step importance n_pos(t)*n_neg(t); focal loss on hard rows.
6. Energy-distance / MMD trailing-window-vs-historical (stronger than decile KS for subtle shifts).

---

## ROUND 3 (per-series empirical-null calibration) — 0.5812 -> 0.6041

Full per-iteration ledger is in `experiments.md`; this is the summary.

EDA refresh (`scripts/eda2.py`): the H0 spread of sliding-window stats varies
2-3x across series (std of window-50 mean-z ranges 0.53..1.83). Because TS-AUC is
a CROSS-SECTIONAL ranking at a fixed step, that scale heterogeneity is exactly
what limits the metric -> calibrate every statistic to each series' own null.

- **v2** (`src/sb/features2.py`, model_003): per-series null-calibrated window
  stats (loc/scale measured on the break-free historical segment at dyadic scales
  8..512) + multiscale scan + running-max + AR(1)-prewhitened CUSUM/EWMA bank +
  per-series null-descriptor constants. **VAL 0.5988** (+1.76 over round 2). The
  null-descriptor constants and prewhitened/scan features dominate feature gain.
- **v3** (`features3.py`, model_010-012): + calibrated derived streams (|z|, dz,
  median-crossing). Single-model 0.587-0.595 (below v2) but decorrelated.
- **v4** (`features4.py`, model_015-016): + null-calibrated higher moments
  (skew/kurt windows 100/200/400/800). Single-model 0.592-0.596; calibrated
  skew/kurt rank top-12 by gain.
- **Ensembling**: all GBTs correlate 0.97-0.99 (`ensemble_search.py`) -> averaging
  caps ~0.602. Learned stacking REJECTED (overfits ranking, 0.55-0.59 OOS,
  `stack_eval.py`). Best diverse GBT blend {003,008,009(v2)+015(v4)} = 0.6031.
- **Logistic member** (`final_blend.py`, model_017): different model class,
  corr 0.93 -> blend 0.8*mean(GBT logits)+0.2*logistic = **VAL 0.6041** (shipped).
- **FINAL** (model_018): `submission/main.py` via `scripts/make_submission3.py`
  inlines v2+v3+v4 (162 feats, exact parity) + the 4-GBT+logistic ensemble.
  Smoke test determinism max|diff|=0.0; ~0.4 ms/point (~33 min full test).

Honest gap: VAL 0.6041 vs top-10 cutoff 0.6135 -> ~+0.9 pt short. GBT ensembling
is exhausted; next pts need a new representation. Ranked next steps (see
context.md/experiments.md): (1) calibrate the CUMULATIVE detectors by per-series
H0 growth (same proven family), (2) a neural sequence member (decorrelation),
(3) TSFM drift embeddings. Settled-don't-retry: stacking, big trees, step
weighting, postprocessing, raw uncalibrated features.
