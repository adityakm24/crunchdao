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

## 2. Current model (round 5 — best, SHIPPED; round 6 kept it after 6 negatives)

- **Held-out VAL TS-AUC = 0.6160** (2000-series internal split) — a flat tie with
  round-4's 0.6161 (the neural sub-ensemble is **saturated**; see below), up from
  round-3's 0.6041 and the EWMA baseline 0.4806 (**+13.5 pts**). Still **clears the
  leaderboard top-10 cutoff (~0.6135)** on the internal split.
- **Round 5 is a CORRECTNESS release, not a metric gain.** A float64 parity test
  (`scripts/test_seq_parity.py`) against `torch.nn.GRU` caught a bug in the
  *served* `StreamingGRU`: the candidate (`n`) gate must gate the hidden bias
  `b_hn` by the reset gate `r` — `n = tanh(W_in·x + b_in + r·(W_hn·h + b_hn))` —
  but the shipped recurrence folded `b_hn` into the input bias, a `(1−r)·b_hn`
  ≈ 1.3e-2/step error that compounded (max |logit diff| 0.675). The trainer's
  `numpy_gru_forward` (which *measured* 0.6161) was always correct, so round-4
  **served ≠ evaluated**. Round 5 keeps `bih`/`bhh` separate (parity ~5e-17) so
  the deployed model scores what we evaluated, and nudges `W_GRU` 0.40→0.45 (flat
  on VAL, marginally better OOS reduced 0.5598 vs round-4's 0.5595). Generator:
  `scripts/make_submission5.py`. **Rejected this round** (all via the OOS reduced
  test, which exposed VAL-halves optimism): LSTM members (severe overfit — blend
  collapses to base), more GRU seeds (dilute), a raw-stream GRU (weak *and*
  rank-corr 0.912 — redundant with the base GBTs). **Verdict: recurrent members on
  these calibrated features are input-bottlenecked and saturated; the gap to #1
  (~0.6322) needs a different signal class, not more nets.**
- **Round 6 (negative-results round, ships nothing — keeps `model_029`).** Tested
  the round-5 hypothesis ("need a different signal class") across **six independent
  avenues**, each measured for *incremental* lift onto the shipped 0.6160 stack
  with honest halves and a hard OOS gate: (1) 10 subtle-break detectors — subtle
  cohort **≤0.50** (structural floor); (2) distributional distances (AD/energy/CvM/
  Wasserstein/Hurst) — **+0.0005**, weight 0; (3) Shiryaev–Roberts mixture —
  **+0.001**, weight 0; (4) ExtraTrees/RandomForest bagging — ET lifts the bare
  base +0.0009 but is **flat on the shipped stack** (rank-corr **0.958** to the
  GRU — redundant); (5) raw windowed matched filters — best **+0.0017**, weight 0;
  (6) a learned causal dilated **1-D CNN** over the raw stream (1-ch std 0.5191 and
  3-ch z/z²/|z| std 0.5212) — the *most decorrelated member ever* (rank-corr
  **0.77**) yet **flat→declining** at every blend weight, both halves down.
  **Verdict: comprehensively saturated on the given inputs.** Every signal that
  decorrelates from the GBTs is already captured by the 3-GRU sub-ensemble, and the
  residual (subtle mean/distribution shifts) is at the detection floor. Scripts:
  `eda_subtle.py`, `eda_incremental.py`, `eda_sr.py`, `diverse_members.py`,
  `finalize_et.py`, `eda_window.py`, `train_cnn.py`, `eval_cnn.py`, `reduced_cnn.py`;
  artifacts `model_030_extratrees`, `model_031_cnn` (both rejected, kept for repro).
  **Beating ~0.6322 needs a fundamentally different data source or label structure
  (e.g. a bundled offline time-series foundation-model embedding) — the single
  highest-EV remaining idea and a large, separate effort.**
- **Model:** `submission/main.py` (generated by `scripts/make_submission5.py`) —
  self-contained `train()` + `infer()`, deterministic.
  - **O(1)-per-step streaming extractor with per-series empirical-null
    calibration** (`src/sb/features2/3/4.py`, 162 features inlined; v2/v3/v4 are a
    nested superset, exact-parity verified). See §5 for *why* this was the win.
  - **Base ensemble** (logit space): 4 LightGBM members — 3 on the v2 feature
    subset (seeds 42/7/2026, rounds 374/231/306) + 1 on the v4 superset
    (seed 42, 348 rounds) — blended `0.8·mean(GBT logits) + 0.2·logistic logit`
    with a logistic-regression member on the v2 subset (model-class diversity).
  - **GRU neural sub-ensemble (the round-4 lift, corrected serve in round 5):** 3
    single-layer GRUs (hidden 128/96/160, seeds 0/1/2) over the 152-feature
    calibrated stream, their step-logits **averaged** into one neural member and
    mixed in as `final = 0.55·base + 0.45·mean(GRU logits)`. The recurrent net
    integrates break evidence over time — a different inductive bias from trees
    (rank-corr ~0.85 vs
    the 0.97–0.99 GBT–GBT), so it decorrelates and lifts the blend. Trained
    **offline** (PyTorch/MPS, `scripts/train_seq.py`); weights exported to numpy
    and **embedded as base64**; inference is an **exact float64-numpy GRU
    recurrence** (no torch, no RNG) → deterministic, O(H) per step. `train()` still
    rebuilds the GBTs + logistic deterministically (the cloud has no torch).
  - Positives down-weighted near the break (`weight = clip(elapsed/50, 0.2, 1)`).
- **Local verification:** smoke test **determinism max|diff| = 0.0** (≪1e-8);
  ~0.39 ms/point GBT base + O(H) GRU → full 10k-series test well inside the 15 h
  budget. Full `submission/local_test.py` mirrors the cloud runner end-to-end.
- **Prior live submission:** #1 (round-2 model, VAL 0.5812) to project
  `chinchilla` (userId 13086). The round-4 model above supersedes round 3 — push
  it next (see §3 "Submitting again").

> NOTE: `crunch push` uploads `main.py` *and* whatever is in `resources/`. The
> cloud re-runs our deterministic `train()` regardless, so a stale `resources/`
> model is harmless, but be aware it is shipped.

---

## 3. How to work in this repo

Environment: **`uv`** venv, **Python 3.13** (cloud runtime 3.13.x). Always use
`uv run` / `uv add`. Core deps: numpy, pandas, scipy, scikit-learn, lightgbm,
pyarrow. **Data note:** the parquet/npz live in Git LFS; on a fresh clone where
LFS pull is blocked, fetch them from the GitHub media CDN, e.g.
`curl -sL https://media.githubusercontent.com/media/<owner>/<repo>/main/Dataset/X_train.parquet -o Dataset/X_train.parquet`
(then `git update-index --skip-worktree` the big files so they don't show as
dirty). Feature matrices (`features/train_features_v*.npz`) are git-ignored —
**regenerate** them with the build scripts, don't commit.

```
Dataset/                 competition parquet (X/y train + reduced test)
src/sb/                  data.py (loader), metric.py (fast TS-AUC),
                         features.py (v1), features2/3/4.py (round-3 extractors)
scripts/                 eda, eda2, build_features{,2,3,4}, train, train2,
                         diagnose, ensemble_eval, ensemble_search, combo_eval,
                         stack_eval, final_blend, make_submission{,2,3}
artifacts/models/        versioned iteration artifacts: model_001 … model_018
submission/main.py       self-contained train()/infer() — THE submission artifact
submission/local_test.py mimics the Crunch runner + determinism check
```

### The feature extractors are the single source of truth
`submission/main.py` is **auto-generated** by `scripts/make_submission3.py`, which
textually inlines `src/sb/features2.py` + `features3.py` + `features4.py` (the
cloud can't import our package) and appends the train()/infer() ensemble.
v2⊂v3⊂v4 is a **nested feature superset** (bit-exact parity is verified). **Edit
the `features*.py` source, then regenerate** — never hand-edit `main.py`.

### Typical round-3 loop
```bash
uv run python scripts/eda2.py                 # deeper EDA (break structure)
# Build a feature matrix (sharded to fit RAM — run 2 shards at a time, not 4):
for k in 0 1; do uv run python scripts/build_features4.py --shard $k --n-shards 4 & done; wait
for k in 2 3; do uv run python scripts/build_features4.py --shard $k --n-shards 4 & done; wait
uv run python scripts/build_features4.py --merge --n-shards 4
uv run python scripts/train2.py --model-id model_0XX --features features/train_features_v4.npz --no-retrain
uv run python scripts/diagnose.py --model-id model_0XX --features features/train_features_v4.npz
uv run python scripts/ensemble_search.py      # pick the diverse blend
uv run python scripts/final_blend.py          # add logistic member, save params
uv run python scripts/make_submission4.py     # regenerate submission/main.py (GBT+logistic+GRU)
uv run python submission/local_test.py        # end-to-end + determinism
```
> RAM: building the 5M×162 matrix with **4** concurrent shards OOMs the 15 GB box.
> Run **2 shards at a time** (each loads the full parquet ~3 GB). `train2.py`
> `--no-retrain` keeps the val-split booster (`lgbm_valsplit.txt`) + `meta.json`
> for evaluation; drop the flag to also retrain on all data into `lgbm.txt`.

### Iteration artifact rule (must follow)
- Every meaningful training run gets a new model id folder under
  `artifacts/models/` — never overwrite a prior one.
- Round-3 map: `model_002` reproduces round 2; `model_003`=v2; `010-012`=v3;
  `013/014`=shallow; `015/016`=v4; `017_logistic`=logistic params;
  `019`=v5 (LRV calib, rejected); `model_018_ensemble`=the round-3 FINAL.
- Round-4 map: `model_020_gru`/`021_gru`/`022_gru`=the 3 GRU seeds
  (hidden 128/96/160); **`model_023_ensemble`=the round-4 FINAL** (base +
  3-seed GRU sub-ensemble recipe, VAL 0.6161).

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

## 5. The findings that mattered most

### Round 3 (this handoff): 0.5812 → 0.6041
1. **Per-series empirical-null calibration is THE lever (+1.8 pts single-model).**
   TS-AUC ranks series *cross-sectionally at a fixed step*, so a statistic's raw
   magnitude is meaningless unless comparable across series. EDA
   (`scripts/eda2.py`) showed the H0 spread of a sliding-window mean-z varies
   **0.53→1.83 across series** (heterogeneous serial dependence/tails). Fix: for
   every window statistic, **measure the series' own null loc/scale on its
   break-free historical segment** (dyadic scales 8…512) and emit calibrated
   units. The per-series null-descriptor constants (`null_slv64`, `null_mem_slope`,
   `kurt_h`, `acf1_h`, …) became the top features by gain. (`src/sb/features2.py`.)
2. **AR(1)-prewhitened CUSUM/EWMA bank** + **multiscale scan** (max calibrated
   stat over window sizes) + its **running-max** all rank highly — they encode
   "has clear, persistent, scale-matched deviation appeared".
3. **Equal mean-logit averaging beats learned stacking** (which scored 0.55–0.59
   out of sample — it overfits the *ranking* metric, `scripts/stack_eval.py`).
   Useful diversity comes from **different feature sets** (v2/v3/v4) and a
   **different model class** (a logistic member, corr 0.93 vs 0.97–0.99 GBT–GBT),
   not more seeds.

### Round 2 (still true, do not regress)
4. **Time-feature trap:** drop raw `t` / `log_n_hist` (constant within a step →
   pure ranking noise) **but keep `log_t` & `t_over_nhist`** (they let the GBT
   normalise growing accumulators via interactions).
5. **Select iterations by TS-AUC, not pointwise AUC** (custom `feval`,
   `src/sb/metric.py`). 0.5702 (round 1) → 0.5812 (round 2) → **0.6041 (round 3)**.

---

## 6. The honest ceiling (read before optimizing)

Round 3 broke the round-2 plateau (0.57–0.58 → 0.6041) via per-series null
calibration. Round 4 broke the **GBT-ensembling ceiling** (all 10 trained GBTs
correlate 0.97–0.99, so averaging them caps ~0.602; the logistic pushed to 0.604)
by adding a **GRU neural member** — a different model class (rank-corr ~0.83) that
integrates evidence over time, lifting the blend to **0.6161** (clears the ~0.6135
top-10 cutoff on VAL). Structural reasons the metric stays hard:

- `tau ~ Uniform` over the online segment → at any step a large share of "broken"
  series broke *just before* `t`; **fresh breaks (elapsed<25) ≈ 0.52 AUC** — a
  genuine ceiling (`scripts/diagnose.py`, `scripts/diag_cohorts.py`).
- **Subtle/distribution-only breaks are 68% of breaks and only ≈0.56 AUC**
  (median KS ≈ 0.12, near the two-sample detection limit); v4's calibrated
  skew/kurt nudged but didn't crack it. The GRU helps most on **mature/mid-life**
  breaks (it accumulates evidence), not these fresh/subtle ones.

**Leaderboard context:** top-10 cutoff ≈ **0.6135**, #1 ≈ 0.6322. Round 4 reaches
**0.6161** on VAL (top-10 territory); closing the gap to #1 needs either a
stronger sequence model (temporal CNN/Transformer or a TSFM drift embedding) or a
per-break-type expert for the subtle-break majority.

---

## 7. Highest-leverage directions to try next (ranked by EV)

> Note: the obvious "calibrate the cumulative detectors" idea was tried as **v5**
> (LRV calibration, `src/sb/features5.py`, model_019) and **REJECTED** (0.5879).
> **Round 4 implemented direction #1 below (a GRU sequence member): VAL
> 0.6041 → 0.6161.** **Round 5 fixed a serve-side GRU bug (served == evaluated)
> and found the neural sub-ensemble is SATURATED at ~0.6161** — more seeds, an
> LSTM, and a raw-stream member were all tried and rejected via the OOS reduced
> test (see experiments.md R5). So directions #2/#3 (a different *signal class*)
> are now the real path to #1, not more nets on the same calibrated input.

### Round 6 — start here (housekeeping first, then a new signal class)

0. **⚠️ Security remediation (do first).** The crunch CLI auth **token** was
   committed to history (`structural-break-real-time-chinchilla/.crunchdao/token`,
   live on `origin/main` since before round 5). The **non-destructive** part is
   DONE in round 5's follow-up: `.crunchdao/` is now `.gitignore`d and the token +
   `project.json` were `git rm --cached` (untracked, still on disk). **Still TODO
   (destructive — owner action):** (a) **rotate the token** in the CrunchDAO
   dashboard (assume compromised); (b) **purge it from history** —
   `git filter-repo --path structural-break-real-time-chinchilla/.crunchdao/token --invert-paths`
   then `git push --force`. Also consider purging the large tracked parquets under
   `structural-break-real-time-chinchilla/data/` (X_train.parquet ~218 MB) to
   de-bloat the repo. Coordinate the force-push since it rewrites shared history.

1. **✅ DONE (round 4) — a neural sequence member.** A single-layer GRU over the
   calibrated feature stream, trained offline and run as exact float64-numpy at
   serve time. 3 seeds (hidden 128/96/160) averaged, mixed at W_GRU=0.40.
   Standalone ~0.605, rank-corr ~0.83 vs the GBT base → blend **0.6161**. The GRU
   *overfits past ~epoch 7* (VAL peaks then decays as train loss falls), so
   best-epoch-by-VAL selection is essential; 12–15 epochs suffice next time.
   **Round 5 update:** this lever is now SATURATED — a 2nd architecture (LSTM),
   more seeds, and a different input (raw-stream GRU) were all tried and rejected
   (the LSTM overfits OOS, extra seeds dilute, the raw stream is weak AND
   rank-corr 0.91). Do NOT keep adding nets on the same calibrated input.
2. **TSFM drift embeddings** (Chronos/Moirai/TimesFM), historical-vs-trailing
   embedding distance as features. Bundle weights (no cloud internet); INT8/ONNX
   for the 15 h budget. Biggest potential lift, biggest effort.
3. **Per-break-type experts + gate** (mean/var/acf/subtle), à la the 2025 winners'
   epistemic diversity. The subtle-break expert is the one that matters (68% of
   breaks, 0.56 AUC) — the GRU integrates evidence but doesn't specialise.
4. **Energy distance / MMD + spectral-slope**, calibrated per series. Lower EV:
   KS/L1/skew/kurt are already calibrated and correlated with these.

### Settled — do NOT retry (evidence in experiments.md)
Learned stacking (of GBT logits); leaves>31 / aggressive feature_fraction /
per-step weighting / scale_pos_weight; score postprocessing (running-max/
smoothing); raw uncalibrated distance/higher-moment features; **LRV-calibrated
cumulative detectors (v5)**; **training the GRU >15 epochs** (it overfits the
ranking metric — VAL peaks ~epoch 7). All measured to hurt or no-op.
**Round-5 additions (all rejected via the OOS reduced test):** LSTM sequence
members (look +0.0007 on VAL halves but COLLAPSE out-of-sample); **more GRU seeds**
beyond 3 (dilute the average); a **raw-stream GRU** on single-point `[z, z², |z|]`
(weak 0.5465 AND rank-corr 0.91 — redundant with the base GBTs); **per-online-step
bucketed blend weight** (`phase0_squeeze.py` — overfits both honest halves).
The neural sub-ensemble is **saturated**; spend round-6 effort on a new signal
class (§7 #2/#3), not more nets.

---

## 8. Gotchas / constraints

- **Determinism is enforced** (`crunch test` re-runs infer on 10% at 1e-8
  tolerance). No RNG in the feature path; LightGBM `deterministic=True,
  force_row_wise=True`. The GRU member ships as an **exact float64-numpy
  recurrence** (weights frozen, base64-embedded) — no torch, no RNG — so it is
  bit-deterministic too. Keep it that way.
- **No look-ahead, no series re-read** in `infer()`; everything must be O(1)
  per step (the GRU adds O(H) per step). Full 10k-series test runs comfortably
  inside the 15 h budget. GRUs are `reset()` per series.
- **Forbidden-library check** on push. Stick to numpy/scipy/lightgbm/joblib/
  sklearn. **torch is LOCAL-ONLY** (training) — it must NOT appear in
  `structural-break-real-time-chinchilla/requirements.txt`; the served GRU is
  pure numpy. `scripts/train_seq.py` stays in the repo for the Out-of-Sample
  retraining phase.
- The competition accelerator (if you go the GPU/TSFM route) has **no internet** —
  bundle any model weights as a Kaggle/crunch resource, don't pip-install at runtime.
- `submission/main.py` is generated — edit `src/sb/features*.py` (or the GRU via
  `scripts/train_seq.py`) + regenerate with `scripts/make_submission5.py`
  (`SB_SEQ_DIRS=...,...`, `SB_W_GRU=0.45`; `make_submission4.py` is the round-4
  generator, kept for reference).
