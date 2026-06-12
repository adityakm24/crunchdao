# Experiments — ADIA Lab Structural Break (Real-Time Edition)

Chronological ledger of everything tried, with metrics, decisions, and reasons.
Metric is **TS-AUC** (held-out internal VAL split of 2000 series unless noted).
Baseline EWMA z-score ≈ **0.4806**. Companion docs: `context.md` (orientation +
future directions), `experiments/EXPERIMENT_LOG.md` (original log).

## Data

- `X_train.parquet` — 35,036,464 rows, MultiIndex `(id, time)`, cols `[value, period]`.
  `period==1` historical (break-free), `period==2` online.
- `y_train.parquet` — per-`(id,time)` `target` over the ONLINE portion = ideal
  step label (0 before break, 1 from break onward).
- `y_train_index.parquet` — per-id `tau_index` (0-based pos in online, −1 = no
  break) and `tau` (absolute time; `tau = n_hist + tau_index`).
- Reduced local test: `X_test.reduced.parquet` (100 series, ids ≥ 10000) +
  `y_test.reduced.parquet` + `y_test_index.reduced.parquet`.

## Metric harness

`src/sb/metric.py` implements a fast **Mann-Whitney rank** TS-AUC so it can run as
a LightGBM `feval` callback every iteration without dominating training time.

---

## Results summary

| Exp | Change | VAL TS-AUC | Decision |
| --- | --- | --- | --- |
| 000 | Provided EWMA z-score baseline | ~0.4806 | reference |
| 001a | O(1) tracker + LGBM, **with** raw time feats | 0.5687 (TEST **0.4465**) | ✗ time-trap |
| 001b | drop pure-time feats (23 content feats) | 0.5605 | ✓ first win |
| 002 | + var-CUSUM + Page-Hinkley + window(50) | 0.5613 | ~flat |
| 003 | only sqrt(n)-normalised accumulators | 0.5558 | ✗ keep raw+norm |
| 004 | WindowDist refactor, windows (50,150), 38 feats | **0.5702** | ✓ round-1 best |
| 005 | lambdarank grouped by step | 0.5223 | ✗ rejected |
| 006 | hyperparam sweep (pointwise select) | 0.566–0.570 | signal-limited |
| 007 | multi-scale windows 25/50/100/200 + multi-k CUSUM + var-of-diffs | 0.564 (pointwise sel.) | needs TS-AUC sel. |
| — | **fix: select iters by TS-AUC** (feval) | — | ✓ correctness |
| — | **re-add `log_t`,`t_over_nhist`** (ablation) | 0.5645 → **0.5776** | ✓ big jump |
| 006b | capacity sweep (TS-AUC select) | **0.5796** | leaves31/lr.03/md1000 |
| 006c | elapsed-weight positives clip(elapsed/50,0.2,1) | **0.5812** | ✓ round-2 best |
| 008 | + chi2 dist + lag-2 acf | 0.5777 | ✗ rejected (leaner is better) |

**Current best / submitted: 0.5812.**

---

## Detailed log

### EXP-000 — Baseline (provided)
Streaming EWMA z-score, `tanh` squash (`ALPHA=0.05, KAPPA=3.0`), mean-only.
Quickstarter local TS-AUC **0.4806** (≈ random). Reference to beat.

### EXP-001a — O(1) tracker + LightGBM, WITH time features
27 features incl. absolute time. Pointwise val AUC 0.699 but **VAL TS-AUC 0.5687,
reduced TEST 0.4465 (below random!)**. Top features were `t`, `log_t`,
`t_over_nhist` — pure time. TS-AUC is cross-sectional at fixed `t`, so raw time is
constant across series and useless for ranking; it diverted capacity. → **the
time-feature trap.**

### EXP-001b — drop pure-time features
23 content-only features (`best_iter≈224`). **VAL 0.5605, reduced TEST 0.5138**.
First clear win over baseline. Top gain: `cusum_absmax`, `online_acf1`,
`tail3_excess`, `cum_kurt_diff`, `ks_stat`, `cum_skew`, `acf1_diff`. Weakness:
cumulative features dilute post-break signal (mix pre/post online points).

### EXP-002 — + variance CUSUM + Page-Hinkley + trailing-window(50)
34 features. Pointwise AUC 0.686, **VAL 0.5613**, reduced TEST 0.4940.
`cusum_var_absmax` became dominant. Flat vs 001b.

### EXP-003 — only sqrt(n)-normalised accumulators
Dropped raw `cusum_*`/`ph_mean`, kept only `/sqrt(n)` versions. **VAL 0.5558
(worse).** Raw accumulators carry genuine within-step magnitude signal; the GBT
can't use `t` directly so raw growth doesn't hurt cross-sectional ranking.
→ **keep BOTH raw and normalised.**

### EXP-004 — WindowDist refactor (round-1 best)
Two windows (50, 150) with localized acf + 19-knot eCDF; 38 features; keep
raw+norm accumulators. Pointwise AUC 0.694, **VAL 0.5702 (round-1 best)**, reduced
TEST 0.5132. Window features rank low in gain (correlated with var-CUSUM). Gap
pointwise(0.694) ≫ TS-AUC(0.57) = cross-time order the metric ignores.

### EXP-005 — lambdarank grouped by online step (REJECTED)
group = online-step index, binary relevance, `objective=lambdarank`.
**VAL 0.5223 (worse)**; diverged after iter 1. LambdaRank optimizes NDCG
(top-of-list), not full-ranking AUC → misaligned with TS-AUC. Pointwise binary is
the right objective. (`scripts/train_rank.py`)

### EXP-006 — hyperparameter sweep, pointwise select (`scripts/sweep.py`)
`leaves{31,63,127,255} × min_data{200,500,1000} × lr{.03,.05} × ff{.6,.8}`.
All cluster at **0.566–0.570**. Best 0.5703 ≈ EXP-004. → **signal/feature-limited,
not capacity-limited.**

### EXP-007 — multi-scale windows + var-of-diffs + multi-k CUSUM
57 features (windows 25/50/100/200, multi-k mean CUSUM, var CUSUM, Page-Hinkley,
first-difference variance/abs ratios). With **pointwise** early stop: val 0.564
(worse than 004) — but this was a selection artifact, see next.

### FIX — select iterations by TS-AUC (custom `feval`)
Added a LightGBM `feval` computing TS-AUC on val each round (fast Mann-Whitney in
`src/sb/metric.py`). Pointwise AUC peaks early then fits cross-time noise the
metric ignores; TS-AUC selection picks a genuinely better iteration. **Correctness
fix**, half of the round-2 gain.

### BREAKTHROUGH — re-add `log_t` & `t_over_nhist` (`scripts/ablation.py`)
The earlier "drop ALL time features" was confounded by pointwise selection. With
TS-AUC selection, these two let the model **normalise accumulating statistics**
via interactions (constant within a step → no cross-sectional leak). Ablation:
- content-only (drop time): **0.5645**
- content + `log_t` + `t_over_nhist`: **0.5776** ← big jump
- content + ALL time (incl `t`, `log_n_hist`): 0.5747 (raw `t`/`log_n_hist` add noise)

→ **drop only `{t, log_n_hist}`; keep `log_t` & `t_over_nhist`.**

### EXP-006b — capacity sweep, TS-AUC select (`scripts/sweep2.py`)
Shallower + more regularised wins: **leaves31, lr.03, md1000 → 0.5796**
(leaves255 overfits; ff.5 hurts).

### EXP-006c — weighting (`scripts/sweep3.py`)
Down-weight near-undetectable fresh-break positives,
`weight = clip(elapsed/50, 0.2, 1)`: **0.5812 (best)**. `scale_pos_weight` and
aggressive `feature_fraction` hurt.

### EXP-008 — chi-square dist-distance + lag-2 acf (REJECTED)
Added per-window chi2 + cumulative chi2 + `online_acf2`/`acf2_diff`: **val 0.5777
(worse than 0.5812)**. The model consistently prefers a leaner feature set; these
are dropped in the shipped model (`DROP_FEATURES` in `main.py`).

---

## Current best (round 2) — SUBMITTED

- **Feature set:** EXP-007 multi-scale (windows 25/50/100/200, multi-k CUSUM, var
  CUSUM, Page-Hinkley, var-of-diffs, EWMA, cumulative moments/eCDF) + `log_t` +
  `t_over_nhist`. Drop `{t, log_n_hist, chi2*, acf2*}` → 62 raw, fewer used.
- **Model:** LightGBM binary, `num_leaves=31, lr=0.03, min_data_in_leaf=1000,
  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1`, `288` rounds
  (frozen from TS-AUC selection), elapsed-weighting (ramp=50), `deterministic=True`.
- **Held-out VAL TS-AUC = 0.5812** (vs 0.4806 baseline, 0.5702 round-1).
- Reduced local test (100 series, noisy): 0.52–0.53.
- `crunch test`: train ~2.5 min, infer fast, **determinism PASS @1e-8**, ~1.6 GB RAM.
- **Pushed as submission #1** (project `chinchilla`, userId 13086).

### Honest gap
Held-out VAL 0.581 vs top-10 cutoff ~0.6135 → need ~+3.2 pts. We are on a plateau
for the single-GBT + handcrafted-feature architecture (sweeps and many feature
variants all cluster 0.57–0.58).

---

## Ideas not yet tried (to break past ~0.58)
1. **Energy distance / MMD** trailing-window-vs-historical (stronger than decile
   KS for the 68% subtle distributional breaks). *Most promising.*
2. Higher-lag autocorrelation, partial-acf, and **spectral-slope** change features
   (for the 20.6% dependence breaks).
3. **Per-break-type expert models + gating** (mean/var/acf/distribution); 2025
   winners leaned on epistemic diversity.
4. **Step-importance sample weighting** by `n_pos(t)*n_neg(t)` and/or focal loss
   on hard rows near `tau` (cheap; aligns loss with the metric).
5. **TSFM embeddings** (Chronos/Moirai/TimesFM) as drift features — bundle weights
   (no runtime internet on the competition accelerator); heavy, do last.
