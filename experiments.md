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
| **R3 — round 3 (this handoff): per-series empirical-null calibration** | | | |
| model_002 | reproduce round-2 best (v1 features) | 0.5812 | ✓ baseline reproduced exactly |
| model_003 | **v2 features**: per-series null-calibrated windows + multiscale scan + running-max + AR(1)-prewhitened CUSUM bank + series-null constants (117 feats, seed 42) | **0.5988** | ✓ **+1.76 pts, breaks the plateau** |
| model_004 | v2, bigger capacity (leaves63, md500) | 0.5889 | ✗ overfits (as in round 2) |
| model_005 | v2 + per-step metric weighting `n_pos·n_neg` | 0.5901 | ✗ rejected |
| model_006 | v2, lean (drop <5k-gain feats → 82) | 0.5925 | ✗ full set better |
| model_007 | v2, elapsed ramp=25 | 0.5985 | ~flat vs ramp50 |
| model_008 | v2, seed 7 | 0.5969 | seed variance ~±0.0018 |
| model_009 | v2, seed 2026 | 0.6004 | best single v2 |
| ens-v2 | mean-logit ensemble {42,7,2026} | **0.6018** | ✓ +0.0014 over best single |
| model_010/011/012 | **v3 features** (= v2 + calibrated derived streams \|z\|/Δz/median-crossing), seeds 42/7/2026 | 0.5950 / 0.5925 / 0.5869 | derived streams hurt single-model but decorrelate (corr 0.98) |
| model_013/014 | shallow (leaves15, md2000) v2 / v3, seeds 13/99 | 0.5960 / 0.5913 | shallow ≈ a touch under main; ensemble fodder |
| model_015/016 | **v4 features** (= v3 + calibrated higher-moment skew/kurt windows), seeds 42/2026 | 0.5961 / 0.5915 | calibrated skew/kurt rank top-12 gain; v4 is the best decorrelator (corr 0.97) |
| model_017 | logistic regression on v2 feats (different model class) | 0.5776 | weaker but **corr 0.93** with GBTs → real diversity |
| ens-search | all 10 GBTs are corr **0.97–0.99** → ensembling caps ~0.602 | — | feature-set diversity ≫ seed diversity |
| model_019 | v5 features (v4 + LRV-calibrated cumulative CUSUM/PH/mean-z), seed 42 | 0.5879 | ✗ redundant with prewhitened bank |
| **model_018** | **FINAL: {003,008,009 (v2) + 015 (v4)} GBT mean-logit + 0.2·logistic** | **0.6041** | ✓ **round-3 best, SHIPPED** |
| **R4 — round 4: GRU neural sub-ensemble (model-class diversity)** | | | |
| model_020/021/022_gru | 1-layer GRU seeds 0/1/2 (hidden 128/96/160) over 152 calibrated feats, best-epoch-by-VAL | 0.6048 / 0.5998 / 0.6046 | ✓ rank-corr ~0.83 vs base (most decorrelated member yet) |
| ens-gru | base + **3-seed-AVG** GRU member, W_GRU=0.40 | — | AVG standalone 0.6069; flat plateau 0.6157–0.6161 over W_GRU 0.35–0.50 |
| **model_023** | **FINAL: 0.6·base(4 GBT + 0.2·logistic) + 0.4·mean(3 GRU)** | **0.6161** | ✓ **round-4 best, SHIPPED — clears ~0.6135 top-10 cutoff** |
| **R5 — round 5: GRU serve-bug fix (correctness) + W_GRU 0.40→0.45** | | | |
| bug | served `StreamingGRU` folded `b_hn` into the input bias → n-gate missed `r·b_hn` (~1.3e-2/step, compounding; max\|logit diff\| 0.675). Trainer `numpy_gru_forward` (which measured 0.6161) was always correct → **served ≠ evaluated** | — | ✓ fixed: `bih`/`bhh` separate in both copies; parity vs torch float64 **5e-17** (`test_seq_parity.py`) |
| reject: LSTM | model_024/025_lstm (h128/h96); VAL honest-halves looked +0.0007 (3-GRU+2-LSTM min-half 0.6119) | OOS reduced **collapses** (standalone 0.5429, blend = base 0.5490) | ✗ severe overfit (rank-corr 0.88 > GRU 0.85) — **OOS is the truth, VAL halves lied** |
| reject: 5-GRU | + seeds 3/4 (h112/h144, model_026/027) | min-half 0.6101 < 3-GRU 0.6112 | ✗ weak new seeds dilute the average |
| reject: raw-stream GRU | model_028 on single-point raw `[z, z², \|z\|]` | 0.5465 | ✗ weak **and** rank-corr 0.912 (redundant — base GBTs extract mean/var better) |
| reject: per-step blend W | `phase0_squeeze.py` per-online-step bucketed W | in-sample 0.6171 | ✗ overfits (fit-A hurts half-B −0.0009, fit-B hurts half-A −0.0024) |
| **model_029** | **FINAL: same 3 GRU + base, CORRECTED serve, W_GRU=0.45** | **0.6160** | ✓ **round-5 SHIPPED — correctness; served == evaluated; OOS reduced 0.5598 ≥ round-4 0.5595** |
| **R6 — round 6: aggressive new-signal hunt (6 avenues, all negative)** | | | |
| reject: subtle detectors | `eda_subtle.py` 10 detectors (AD, energy/CvM, spectral/perm entropy, Wasserstein, Hurst, turning-pt, IQR) at 6 snapshots × cum/trailing | subtle-cohort **≤0.50** for ALL | ✗ subtle breaks are a true ≤0.50 ceiling (68% of breaks, KS≈0.12 < detection limit) |
| reject: distributional incr. | `eda_incremental.py` best 5 (AD/energy/KS/Wasserstein/Hurst) blended onto real base | **+0.0005**, optimal weight **0.00** | ✗ orthogonal info already in base; rank-corr 0.10–0.16 but signal too weak |
| reject: Shiryaev-Roberts | `eda_sr.py` SR mixture over mean/var/joint deltas + AR(1)-prewhiten | standalone 0.52, **+0.001**, weight 0 | ✗ sequential CP statistic adds nothing over the calibrated bank |
| reject: ExtraTrees/RF | `diverse_members.py`+`finalize_et.py`; ET(300,leaf200) standalone **0.5930** lifts base +0.0009 | onto **shipped** stack: flat (rank-corr to GRU **0.958**) | ✗ ET redundant with GRU; RF (0.5872) weak — bagging variance already covered |
| reject: windowed filters | `eda_window.py` edge/varenv/acfenv/transient/ramp matched filters at L=20/40/80/160 | best acfenv80 **+0.0017**, weight **0.00** | ✗ linear shape filters near-dead vs shipped |
| reject: 1D-CNN (raw) | `train_cnn.py` causal dilated CNN over raw z-stream, R=63: **1-ch** (32ch, std 0.5191) + **3-ch** z,z²,\|z\| (48ch, std 0.5212) | blend onto shipped **flat→declining** every Wc (0.6160→0.6142), both halves down | ✗ most decorrelated member ever (rank-corr **0.77**) but tops out ~0.52 standalone — below the lift threshold |
| **SHIPPED stays model_029** | round 6 = thorough negative-results round; no member beats VAL 0.6160 | **0.6160** | ✓ **keep round-5; saturation confirmed across 6 signal classes** |
| **R7 — round 7: TSFM embedding member (Chronos-T5-mini) — the 7th negative** | | | |
| reject: TSFM distance | `eda_tsfm.py` Chronos-T5-mini (384-dim) embed hist vs cum/trail; cosine/L2, EOS-token pooling, 6 snapshots | all features **~0.50** standalone, max blend lift **+0.0018** | ✗ a single distance scalar discards the embedding's directional info |
| reject: TSFM sup. head | `eda_tsfm_head.py` logistic on 1152-dim [cum−hist, trail−hist, trail−early], grouped 5-fold CV | head AUC **~0.515**, blend **+0.0008–0.0011**, rank-corr **0.40–0.52** | ✗ tiny signal is **already captured** by the shipped stack (redundant, not orthogonal) |
| reject: TSFM mean-pool | `eda_tsfm_mean.py` length-bucketed exact mean-pooling, re-run distance + head | distance lift **+0.0000–0.0003**; head AUC **0.49–0.50**, lift **+0.0000** | ✗ even flatter than EOS — flat across {distance,supervised}×{EOS,mean} |
| **SHIPPED stays model_029** | round 7 = TSFM negative; Chronos embedding tiny AND redundant | **0.6160** | ✓ **keep round-5; saturation confirmed across 7 signal classes** |
| **R8 — round 8: metric-aligned RANK objective — the FIRST POSITIVE in 8 rounds** | | | |
| reject: lambdarank | `train_rank2.py` LightGBM lambdarank grouped by online-step on v4 feats, trunc=200 (model_032_rank) | standalone **0.5833**; rank-corr to shipped **0.007** (most decorrelated member EVER) | ✗ honest-halves gate FAILS — halfB declines monotonically (NDCG truncation is top-heavy, misaligned with uniform-pair AUC) |
| **accept: rank_xendcg** | `train_rank2.py --xendcg` (model_033_xendcg), listwise CE objective, best_iter 74 | standalone **0.5955**, rank-corr **0.459**; VAL **both halves up** (min +0.0003 @w0.05); **OOS reduced 0.5600→0.5606** across the WHOLE weight grid, both integration paths | ✓ **first member to pass VAL halves AND OOS reduced — xendcg's smooth listwise loss fixes lambdarank's instability** |
| **model_033 shipped** | round-5 stack + **rank_xendcg as a 5th base GBT** (gw=0.15), frozen booster embedded, `rank_const.npz` rescale computed at train() | reduced OOS **0.5606** (+0.0006), full-pipeline retrain, **determinism PASS @1e-8** | ✓ **round-8 SHIPPED — pure LightGBM, zero new deps; first banked gain since round 5** |
| **R9 — round 9: step-conditional feature standardization — VAL mirage, OOS reject** | | | |
| reject: stepnorm | `eda_stepnorm.py`/`reduced_stepnorm.py`: per-online-step cross-sectional NULL (y==0) mean/std baked from TRAIN, smoothed over t; `x→(x−μ₀(t))/σ₀(t)`. Single GBT, same split | VAL full **0.5928→0.6013 (+0.0085)** but halves **asymmetric** (halfA +0.0006 / halfB +0.0159); **OOS reduced 0.5495→0.5457 (−0.0038)** | ✗ **cross-series population leakage** — μ₀(t)/σ₀(t) encode TRAIN series composition (top feat `log_t`=absolute position); transfers on VAL (same id-pool), miscalibrated on OOS. VAL lift was a mirage; OOS gate caught it. Feature-normalization avenue CLOSED |
| **SHIPPED stays model_033** | round 9 = stepnorm OOS-rejected; confirms any train-cross-sectional transform fails OOS | **(OOS 0.5606)** | ✓ **keep round-8; cross-series-population standardization added to the do-not-retry list** |
| **R9b — round 9b: drop `log_t` everywhere — looked like the biggest OOS win, but REGRESSED the REAL leaderboard (reverted R10)** | | | |
| diagnosis | stepnorm autopsy fingered `log_t = log(n_hist+online_step)` (the #1 feature): at a fixed online-step it varies **only with history length** = pure cross-sectional length signal; re-added round 1 (a VAL jump) **before the OOS harness existed**, never OOS-gated → prime driver of the VAL 0.616 vs OOS 0.560 gap | — | the leaky feature hiding in plain sight |
| ablation: drop log_t | `reduced_droppos.py` single GBT, same 80% split, stream reduced OOS once: baseline VAL 0.5928 / halfA 0.5990 / halfB 0.5864 / **OOS 0.5495**; **drop log_t** → halfA 0.5806 / halfB 0.5811 (**symmetric!**) / **OOS 0.5620 (+0.0125)**; drop log_t+t_over_nhist → OOS 0.5530 | **OOS +0.0125, halves symmetric** | ✓ **drop `log_t` ONLY** — `t_over_nhist` (relative position) transfers, **keep it**; symmetric halves = robustness signature (inverse of stepnorm) |
| blend confirm | `reduced_nolog.py` reconstructs ramp weights from cache, 4 GBTs with vs without log_t (logistic+GRUs held fixed, rank gw=0): mean4 0.5518→0.5656, base blend **0.5600→0.5768** | GBTs-only +0.0168 | ✓ confirms at the blend level before full retrain |
| retrain members | rank → `model_034_xendcg` (151 feats, val 0.5945); 3 GRUs → `model_02{0,1,2}_gru_nolog` (151 feats, VAL 0.6066/0.6010/0.5948) | members re-fit clean | ✓ all members log_t-free |
| **model_034 shipped** | full `local_test.py` retrain: rank model_034 + 3 nolog GRUs + generator `_DEFAULT_DROP += log_t` (4 GBTs + logistic) | **reduced OOS 0.5606→0.5809 (+0.0203)**, train 603 s, infer 36 s (~59 min/10k ≪ 15 h), **determinism PASS @1e-8** | ✓ **round-9b SHIPPED — ~34× the rank member; biggest validated jump in the project; zero new deps** |
| **R10 — round 10: REAL leaderboard inverts the gate — `log_t` restored, blend re-tuned, ceiling confirmed** | | | |
| ⚠️ regression | submission #6 = round 9b (log_t dropped) on the REAL public leaderboard | **real 0.5987 → 0.5959** (−0.0028) | ✗ the reduced-OOS gate (100 series) was wrong; **VAL said log_t HELPS and the real test agreed** |
| methodology | **VAL (2000 ser) tracks real (real ≈ VAL − 0.018); reduced-100 is NOISE.** Re-gate everything on VAL full + honest halves | — | ✓ inverts the old "OOS-reduced is truth" belief |
| restore log_t | `_DEFAULT_DROP` − `log_t` (GBTs + logistic keep it); rank/GRU stay nolog (nolog GRU 0.6072 > withlog 0.6048) | blend VAL → **0.6166** | ✓ recovers the regression |
| blend re-tune | `val_blend_search.py` + `val_stack.py`: logistic OVERWEIGHTED (neg. stack weight, corr 0.93); W_LIN 0.20→0.10, RANK_GW 0.15→0.10 | VAL 0.6166 → **0.6170**, both halves up | ✓ shipped weights |
| reject: stepnorm (blend) | `val_stepnorm_blend.py`: single-GBT +0.0051 but the GRU already encodes step structure | blend final **+0.0010** (halfA −0.0008) | ✗ washes out at the blend |
| reject: meta-stacker | `val_stack.py` honest cross-half ridge stack of all 8 members | **0.6160 < 0.6170** hand blend | ✗ nested hand structure generalizes better |
| reject: rank upgrades | `val_rank_ensemble.py`: m033 (xendcg+log_t, 0.5955), m032 (lambdarank, corr −0.05) + ensembles | none beat shipped m034 @ gw0.10 | ✗ rank member also at ceiling |
| reject: spectral + entropy new-signal probe | `val_spectral_probe.py` adds FFT spectral entropy/centroid/band-ratio + permutation entropy (causal hist-vs-online contrasts), honest cross-half logistic on top of shipped blend, **all 2000 VAL series**, 25,194 probe points | standalone AUCs near 0.5; honest deltas **−0.0047 / −0.0120** (min-half −0.0120) | ✗ no robust incremental signal; per-series space appears saturated |
| **model_035 shipped** | restore `log_t` + W_LIN 0.10 + RANK_GW 0.10; nolog GRU/rank unchanged | **VAL 0.6170 (~0.600 real)** — recovers #6 regression, ≈ prior best | ✓ **round-10 SHIPPED — cached re-blend ceiling; drastic needs NEW signal** |
| crunch submission (2026-06-16) | `uv run crunch push` from deploy dir with round-10 recovery package (main.py + resources) | upload completed successfully; run accepted client-side | ✓ submitted; awaiting public leaderboard score |
| **R11 — round 11: DGP-matched synthetic augmentation (washout) + causal-Transformer member (FIRST POSITIVE since R8)** | | | |
| rules check | fetched hub.crunchdao.com + docs (2026-06): methodology **explicitly lists "Deep learning models" + "Foundation models fine-tuned on labeled data"**; FAQ sanctions train-local/ship-weights; cloud has NO network at infer (bundle weights); scipy OK (`SCIPY_ARRAY_API=1`); determinism re-run on 30% @1e-8; 15h/wk | — | ✓ a torch deep member is rules-compliant (re-impl numpy forward OR torch-in-reqs+frozen weights) |
| synth generator | `scripts/gen_synth.py` DGP-matched AR(1)+Student-t (`kurt_to_df`), `--tilt` oversamples moderate-KS frontier; realism classifier AUC 0.943 (separable but imperfect: null tables too narrow, tails/ACF too clean) | — | faithful enough to probe |
| single-GBT A/B | `aug_sweep.py` 2 seeds × 3 src × 2 dose on real VAL; tilt_s2 full-dose | seed42 **+0.0112**, seed7 **+0.0088** single-GBT full-VAL (BOTH seeds) | ✓ first signal-ADDING lever all session at the SINGLE-model level |
| reject: GBT aug @ blend | `aug_base_blend.py` retrains 4 base GBTs real vs real+synth, runs full round-10 blend | AUG members lift (003 +.0045, 008 +.0049, 015 +.0083) but **blend Δ full +0.0002, halfA +0.0025, halfB −0.0024** | ✗ **washes out** — 4 GBTs are 0.97–0.99 corr; blend strength = DECORRELATION (GRU 45%+rank), not GBT accuracy; improving redundant members can't move the blend |
| reject: GRU aug @ blend | `train_seq.py --synth` (TRAIN-only inject, VAL stays real); ctrl vs aug GRU (h96 s1) | ctrl best **0.6009 @ep6**, aug best **0.5994 @ep8** (−0.0015); synth makes GRU peak later (more data) not higher | ✗ augmentation does NOT survive the blend for EITHER member class — per-series signal saturated |
| **accept: causal Transformer** | `scripts/train_attn.py` — `nn.TransformerEncoder` (norm_first, gelu) over the 151 calibrated feats, **causal mask + key-padding**, NO positional encoding (rel-pos feats + mask supply position; abs-pos = time-trap), per-step BCE+ramp, best-VAL-epoch select; 3 seeds (1/2/3), layers3 d128 h8 ff512 maxlen320 | standalone **0.5903 / 0.5864 / 0.5883**, **rank-corr 0.841 / 0.826 / 0.842** to GRU mean (decorrelated like the round-4 GRU's 0.83); each peaks @ep5–7 then overfits | ✓ **FIRST decorrelated new member since R8 — self-attention was the one evidence-integration mechanism never tried (GRU/LSTM/CNN/TCN/TSFM all saturated)** |
| weight-search gate | `scripts/attn_blend_search.py` (pure-numpy on cached logits): attn sub-ensemble rescaled to GRU logit scale, shares the neural pillar `neural=(1−wa)·gru+wa·attn`, search wa (+ joint wa×WG), require lift on **full AND both honest halves** | **2-seed sub-ens** standalone 0.5945, rank-corr 0.854 → **VAL 0.6170→0.6192 (+0.0022)**, halfA +0.0017 halfB +0.0017 (Mode B wa0.40 WG0.50); robust across the whole wa curve | ✓ **validated lift; 2 seeds is the sweet spot** |
| seed-averaging tradeoff | 1/2/3 seeds: standalone 0.5903/0.5945/0.5973 but rank-corr **0.841/0.854/0.869** (RISES) → lift +0.0020/+0.0022/+0.0015 | averaging raises accuracy but converges toward the GRU (shared signal dominates, idiosyncratic decorrelated noise averages out) | ⚠️ **don't stack many seeds; raise the SINGLE-model standalone instead** (push: maxlen 512 + more reg, in progress) |
| **R11b — new INPUT channels (user pivot: attack the input bottleneck, not the architecture)** | | | |
| new streams | `build_raw_stream.py` (z, z², \|z\| Gaussian-parametric), `build_pit_stream.py` (**pit=F̂(xₜ), pit_c, surprise — distribution-free PIT**, targets the 68% subtle non-Gaussian breaks z-standardization misses), `build_rawpit_stream.py` (6-ch merge); all aligned to v4 (sid,t); `train_attn.py --stream PATH --raw {concat,only}` | streams built in 3–5s each (5.04M rows) | infra for the different-INPUT test |
| reject: raw-z **concat** | attn over 151 calib + 3 raw z-channels (154-ch), same config | standalone **0.5877** (< calib 0.5903), rank-corr **0.847** (> calib 0.841 — MORE correlated), blend Δ full −0.0000 halfA +0.0003 halfB −0.0003 | ✗ **raw z is redundant on top of calibrated** — the calib moment-summaries already extract mean/var/tail (same as model_028); adding raw hurts both axes |
| reject: PIT-**only** | attn over 3 PIT + 4 light-context channels (7-ch pure learned detector) | standalone peak **0.5399** @ep13 (monotone-slow, NO overfit — 7-ch can't memorize) | ✗ **too weak standalone** — even a distribution-free detector tops out ~0.54; the subtle majority sits at the genuine two-sample detection floor |
| 3-way stack gate | `attn_stack_search.py gru+calib+pit` — rank-corr matrix gru–calib **0.841**, gru–pit **0.761**, calib–pit **0.760**; honest weight search (full + both halves) | PIT is the **most decorrelated member ever built** (orthogonal to BOTH the GRU and the calib-attn, beats the round-6 CNN's 0.77) — yet the search sets **pit = 0.00** (calib 0.35, gru 0.65 → +0.0020, identical to calib-only) | ✗ **orthogonal NOISE can't lift; you need orthogonal SIGNAL** — at 0.54 any positive PIT weight drags both halves down |
| reject: raw+PIT **combined** (richest different-input) | attn over 6 stream + 4 light-context channels (10-ch: z,z²,\|z\|,pit,pit_c,surprise + ctx), the most expressive learned detector tried | standalone peak **0.5533** @ep15 (highest of all pure learned detectors, monotone-slow), rank-corr **0.774** (orthogonal), equal-weight 4th-neural gate Δ full **−0.0004** halfA −0.0003 halfB −0.0004 | ✗ **still ~0.04 below calib's 0.59 — same weak regime; zeros out like PIT.** 5th consistent negative ⇒ INPUT-bottleneck hypothesis **CLOSED**: the ceiling is information (subtle breaks at the two-sample detection floor), not representation |
| **R11c — CROSS-SERIES structure (user pivot: exploit the within-step cross-sectional metric)** | | | |
| proof: serve recalibration = no-op | `val_xsec_noop.py` — reconstruct shipped blend on VAL, apply 6 cross-sectional recalibrations (per-step z-score / rank / quantile→N(0,1) / min-max, global rank, global sigmoid) | TS-AUC **identical to 1e-12** (Δ = 0.00e+00 for ALL six) | ✗ **mathematically proven**: TS-AUC is within-step Mann-Whitney (rank-based); any transform monotone-in-pred within a step leaves every AUCₜ unchanged. Serve-time cross-series recalibration CANNOT move the metric — only REORDERING series within a step (= better per-series discrimination) can |
| prior art recap | the legit cross-sectional levers were already done: **xendcg rank objective SHIPPED** (R8, listwise CE grouped by online-step, model_033, decorrelated 0.459, at ceiling per R10); **stepnorm cross-sectional feature std CLOSED** (R9, population leakage, failed OOS + real LB) | — | only ONE cross-series cell left untried: a **neural** member with a within-step rank loss (all neural members use pointwise BCE; only the GBT got the rank objective) |
| **model_041_attn SHIPPED (submission #9, 2026-06-18)** | retrain s1/s2 w/ `--save-weights`; `serve_attn.py` = float64 numpy causal KV-cache member (O(T²)/series), **real-weights parity 2.00e-15** vs torch; baked blend-rescale A=1.112/B=+0.290 into head; wired into `make_submission5.py` as additive opt-in 4th neural member (`SB_ATTN_DIRS`); 2-seed mean | VAL 0.6170→**0.6187** (full +0.0017, halfA +0.0024, halfB +0.0012, **robust +0.0012**); local det **0.00e+00**, cloud `crunch test` det **passed @1e-8**, latency 1.70ms/pt (~141min ≪ 15h) | ✓ **SHIPPED — real public 0.6049 (60.49%) vs prior #8 0.5996 = +0.0053 real (≈3× the VAL +0.0017 full — the member GENERALISED BETTER on the hidden test than on VAL); new PB; refines calib to real≈VAL−0.014 (was −0.018); gap to #1 (0.6322) → ~0.027** |
| reject: **rank-loss neural member** (the ONE untried cross-series cell — `train_attn.py --loss rank` within-step pairwise RankNet, coded-never-run; all neural used BCE, only GBT got rank) | model_042_attn_rank s1, same config as shipped; standalone **0.5414** @ep5 (loss→0.002 fast overfit); `attn_stack_search.py` rank-corr to gru/bce1/bce2 = **0.002 / 0.001 / 0.047** (most orthogonal member EVER) | honest stack search picked rank=0.05 (full +0.0024) BUT **ablation** (same search, rank removed) = full +0.0023 → rank's TRUE marginal **+0.0002 full / −0.0014 halfA / +0.0017 halfB (robust −0.0014)** | ✗ **DEAD — orthogonal NOISE not signal (PIT/R11b regime); 0.00-corr = barely-above-random, trades halfA↔halfB, full flat. B closed; rank-loss WAS the cross-series attack so C closed too** |
| note: bce re-weight bonus | same bce-only ablation found gru0.60/bce1 0.25/bce2 0.15/WG0.50 | **+0.0006 uniform** over shipped-A (full/halfA/halfB all +0.0006/+0.0006/+0.0007) | ⚠️ real but **sub-threshold + 3-param gate-fit risk** (stepnorm lesson) — NOT re-shipped |
| **R12 — round 12: independent feature block (2025-winners' "epistemic diversity" playbook) — NEGATIVE, reconfirms the wall** | | | |
| context | research: 2025 winners (Brandão/Alphabot 2nd, L.Morin 3rd — several now competing live: `brandao` #7, `farukcan-saglam` #5) won via **independent feature blocks** (wavelet/dense-rank/%-change views + F-test/Levene/KS) + TabPFN + 2-level stacking. A public cross-dataset benchmark shows **regularized-simple ≫ complex-stack on out-of-dataset stability** (0.74→0.77 @96% vs 0.79→0.65 @82%) — mirrors our VAL 0.619 → real ~0.594 gap (overfit, not just missing signal) | — | TabPFN rejected as a path: gated HF weights + no-internet cloud = undeployable |
| new block `features_xs.py` | the two views v2 **under-covers**: (A) **distribution-free PIT moments** `p=F̂_hist(x)` mean-z/logvar/skew/tail-mass over windows {120,480}+cum (robust to the heavy non-Gaussian tails of the 68% subtle cohort); (B) **multi-lag dependence** acf-diff lags 1–5 vs the series' own historical acf (v2 only has acf1/acf2) + turning-rate + cumslope; per-series calibrated, O(1)/step, deterministic. 38 feats, build 5.04M×38 in ~21s/shard | — | genuinely different representation, not a v2 reskin |
| model_043_xs (with log_t) | LightGBM, TS-AUC select, best_iter 118 | standalone **0.5546**, rank-corr to base **0.906**; base-blend peak **+0.0001 @w0.05** then declines, halves flat | ✗ **correlated AND weak** — `log_t` dominates gain (4.5M) and inflates corr; PIT/acf signal already in v2's calibrated feats |
| model_044_xs_nolog | drop log_t+t_over_nhist (pure dynamics/distribution), best_iter 188 | standalone **0.5453**, rank-corr **0.754** (decorrelated, ≈ round-6 CNN's 0.77); base-blend full +0.0014 @w0.10 BUT **halfA declines 0.5967→0.5945 while halfB rises 0.6110→0.6157** | ✗ **non-robust asymmetric-halves signature** (= stepnorm / rank-loss-neural red flag); the +0.0014 full is a half-trade mirage, FAILS the both-halves gate |
| model_045_wave (with log_t) | `features_wave.py` — causal EWMA-cascade (à trous-style) **multi-resolution band-energy** (the last untried winners' wavelet view): detail_j=s_{j-1}−s_j over 6 dyadic scales, online trailing-window band energy vs the series' OWN historical band energy as log-ratio; windows {120,480}+cum, per-series calibrated, 28 feats | standalone **0.5735** (strongest new block), rank-corr **0.922**; base-blend full +0.0007 @w0.15 but **halfA declines 0.5967→0.5961** | ✗ strong⇒too correlated; non-robust halves |
| model_046_wave_nolog | drop log_t+t_over_nhist (pure band-energy shape) | standalone **0.5534**, rank-corr **0.618** (most decorrelated GBT block yet); base-blend halfA flat→declines, full declines after w0.05 | ✗ decorrelated⇒too weak + non-robust; same dead zone as the round-6 CNN / round-10 spectral |
| **SHIPPED stays model_041 / round-10 blend** | round 12 = 8th & 9th independent signal classes (PIT+dependence, wavelet band-energy) confirmed saturated. **Consistent killer: halfA declines in EVERY new-block blend** — the blocks help only the easy halfB (idiosyncratic noise), never robust cross-sectional signal. Decorrelated⇒too weak (≤0.55), strong⇒too correlated (0.92). The 68% subtle cohort is at the two-sample detection floor (KS≈0.12) | **VAL 0.6187 / real ~0.594** | ✓ **feature-block avenue CLOSED with evidence; remaining real-score EV is the generalization gap (regularization/robustness leaderboard bet), not new per-series signal** |
| **R13 — round 13: from-scratch HIGH-VOLUME + REGULARIZED + generalization-gated GBT (winners'-recipe rebuild; the "robustness" lever)** | | | |
| premise | R12 said feature-volume is dead as *blend add-ons* but never tested the winners' actual recipe: high feature **volume + aggressive selection + strong regularization + a generalization-first gate**. Built `scripts/train_v6.py`: concatenates the 3 independent blocks already on disk (same lexsorted row order — **no rebuild**) → **213 unique feats** (v4 162 + xs 38 + wave 28, dedup shared `log_t`/`t_over_nhist`/`acf1_h`), and gates on a nested **5-fold *series-disjoint* CV** instead of the over-used seed-42 VAL split | — | the honest cross-series generalization protocol |
| CV: v4-only (reg) vs full-bank (reg) | LightGBM leaves31 lr0.03 ff0.6 bag0.8 **l1=1 l2=1 min_data=2000**; 5-fold series-disjoint | v4-only **0.5949 ± 0.0107**; **full-bank 0.5971 ± 0.0109** (folds 0.579–0.613) | ✓ **xs+wave DO help (+0.0022) as FOUNDATIONS** even though they washed out as R12 add-ons — features redundant-as-add-ons can be load-bearing in a from-scratch model (winners' "epistemic diversity" validated; I was testing it wrong) |
| KEY finding — overfitting thesis CONFIRMED | single regularized GBT honest CV **0.597** vs the shipped 12-round blend's **real ~0.594** | — | ✓ **the elaborate blend generalizes WORSE than a simple regularized GBT**; VAL 0.619 was seed-42-split optimism. Honest cross-series ceiling ≈ **0.595–0.597 regardless of config** (3rd independent confirmation) — matches real LB |
| seed-42 VAL sanity | full-bank reg 3-seed on seed-42 VAL = **0.5899** (< base-GBT 0.6041 same split) | — | the strong reg costs in-sample VAL but the seed-42 split is "easy"; the 5-fold mean (0.597) is the trustworthy estimate and aligns with real |
| **submission #10 r13** `scripts/make_submission6.py` → `submission/main.py` (58 KB, no embedded weights) | **stripped ALL neural/logistic/rank** machinery → pure **3-seed (42/7/2026) regularized LightGBM** on the 213-feat bank; combined streaming detector inlines features{2,3,4}+xs+wave; train() rebuilds in-cloud, 250 rounds/seed, deterministic | local CV est. **~0.597**; `crunch test` **PASS** (determinism @1e-8, train 13 min, infer 0.22 s/series ≪ 15 h); **real score PENDING** | ⏳ **SHIPPED #10** as the ROBUST candidate — a public A/B of "simple-robust ≥ overfit-complex" AND the safer **private-LB final-selection** bet (final ranking is a different draw). 0.63 NOT reachable in this paradigm → pivot to (B) a materially stronger single model class |
| **R13b — (B1) Time-series FOUNDATION-MODEL probe (Chronos-Bolt-small) — NEGATIVE, 10th representation to hit the wall** | | | |
| setup | `scripts/probe_chronos.py` + `probe_chronos_fcst.py`. chronos-bolt-small (190 MB) runs offline on CPU ~6.5 ms/embed (env fix: removed `tabpfn`, pinned `huggingface-hub<1.0`; NetApp-MITM SSL → fetched weights via `curl -k` to `models/`). 2500–4000 train series sampled | — | feasible to deploy (CPU, bundled weights) IF signal existed |
| **(A) embedding-distance** | pooled encoder-embedding distance hist-vs-trailing-online window (cos & L2), W_on=96/W_hist=192, RAW + **per-series empirical-null calibration** (the v2 technique) | RAW **0.502/0.507**; **CALIBRATED 0.508/0.510** — random; spearman to mean/var-shift **0.01–0.04** (orthogonal but it's orthogonal NOISE) | ✗ Chronos **instance-normalizes each context** → blind to the mean/variance LEVEL shifts that dominate; residual shape/dynamics distance doesn't separate broken/unbroken |
| **(B) forecast-residual** | forecast online from break-free historical context; normalized |actual−median|/spread per step + running-max integration; DOES see level shifts | resid **0.517**, **cummax 0.545**, naive mean_shift **0.522** (early-break slice) | ✗ marginally > naive baseline, <0.55, and our AR-prewhitened-CUSUM features already capture calibrated predictive-residual signal — redundant |
| **(B1) verdict** | per-series detection signal is now confirmed SATURATED across **trees, logistic, GRU, LSTM, causal-Transformer, rank-loss, PIT, wavelet-MRA, and pretrained TSFM (embed + forecast)** — 10 representations, one wall | honest per-series ceiling **≈ 0.597** ≈ near-Bayes-optimal for the 68% subtle cohort (KS≈0.12 floor) | ✓ **B1 CLOSED.** The leaders' +0.025 is NOT per-series representation signal. Remaining levers: cross-series structure (legit, untested — stepnorm leaked), LB-draw variance (fold σ=0.034), or a better-fit extractor (low EV given saturation). Next: await #10 real score; consider B3 (principled BOCPD) only as a last low-EV lever |
| config-invariance control | full-bank **base params** (min_data1000/ff0.8/no l1l2) vs **strong reg** (#10) under the same 5-fold series CV | base **0.5965 ± 0.0076**, strong-reg **0.5971 ± 0.0109** — identical mean, both at the ceiling | ✓ GBT reg config is irrelevant on the honest gate; **#10 is optimal, no better #11 to ship**. Strong reg only lowered the seed-42-VAL *display* (the "easy" split), not the honest estimate |
| **cross-series avenue assessment** | `infer()` processes ONE series at a time (no access to the step-t cross-section) → legitimate serve-time cross-series features are **structurally impossible**; train-time cross-series **leaks** (stepnorm, R9) | — | ✓ the only uncapped avenue is effectively closed by the interface + leak evidence → the ~0.597 ceiling is real and the +0.025 gap is most likely LB-draw variance + small accumulated leader fit-quality, not a single recoverable lever |




**Round-5 verdict — the neural sub-ensemble is SATURATED at ~0.6161.** Full-VAL is
flat (0.6161/0.6160/0.6157 at W_GRU 0.40/0.45/0.50) and min-half barely moves
(0.6112/0.6114/0.6115). Recurrent members are **input-bottlenecked**: LSTM (a
different architecture) is *more* correlated with the base (0.88) than the GRU
(0.85), and a raw-stream GRU is *both* weak and redundant (corr 0.912). Round 5
ships purely to make the **deployed** model faithful to the **evaluated** one (the
round-4 served GRU was bugged) and to take the marginally-better OOS W=0.45.
Closing the gap to #1 (~0.6322) needs a different **signal class** (a time-series
foundation-model embedding or a dedicated subtle-mean-shift expert), not more nets
on the same calibrated input.

## R6 — round 6: aggressive new-signal hunt (six avenues, all negative)

Round 5 ended on the hypothesis that the gap to #1 needs a **different signal
class**. Round 6 tested that hypothesis hard: six independent new representations,
each measured for *incremental* lift onto the **shipped** stack (not standalone),
with honest two-halves and — for any VAL winner — a hard OOS reduced-test gate.
**All six failed.** The result is a clean, well-evidenced verdict: the blended
model is **saturated on the information present in the given inputs**.

| # | Avenue | Script | Best incremental result | Why it failed |
| --- | --- | --- | --- | --- |
| 1 | Subtle-break detectors (10) | `eda_subtle.py` | subtle-cohort AUC **≤0.50** for every detector | Subtle/distribution-only breaks (68% of all breaks, median KS≈0.12) sit below the two-sample detection limit — a **structural** ceiling, not a modelling gap. |
| 2 | Distributional incremental | `eda_incremental.py` | **+0.0005**, optimal blend weight **0.00** | AD/energy/CvM/Wasserstein/Hurst decorrelate (rank-corr 0.10–0.16) but are individually too weak; the base GBTs already extract what little they add. |
| 3 | Shiryaev–Roberts mixture | `eda_sr.py` | standalone 0.52, **+0.001**, weight 0 | A sequential change-point statistic over mean/var/joint deltas (+AR(1) prewhiten) adds nothing over the calibrated CUSUM/PH bank. |
| 4 | Bagging (ExtraTrees/RF) | `diverse_members.py`, `finalize_et.py` | ET standalone **0.5930**, lifts *base* +0.0009 but **flat on shipped** | ET's rank-corr to the **GRU** is **0.958** — the temporal-integration variance reduction bagging offers is already supplied by the 3-GRU sub-ensemble. RF (0.5872) is just weaker. |
| 5 | Raw windowed matched filters | `eda_window.py` | best acfenv80 **+0.0017**, weight **0.00** | Linear edge/variance-envelope/ACF-envelope/transient/ramp filters over the raw window are near-dead once the shipped neural member is present. |
| 6 | Learned 1-D CNN (raw stream) | `train_cnn.py`, `eval_cnn.py` | **flat→declining** at every blend weight | See below — the decisive test. |

### The CNN test (avenue 6, the one untested lever)
The single-point raw GRU (round 5) was weak *and* redundant, but a **windowed**
convolution is a genuinely different input: a causal dilated 1-D CNN whose
receptive field (R=63, dilations 1,2,4,8,16) sees local *shape*, not summary
statistics. Two configurations were trained to convergence (PyTorch/MPS, exported
to exact float64 numpy, parity ~1e-6 → deterministic at ship time):

- **1-channel** (raw z only, 32 ch, 25 ep): converged hard at VAL standalone
  **0.5191** from epoch ~12 (loss flat 0.476); rank-corr to shipped **0.769** —
  the *most decorrelated member ever produced*.
- **3-channel** (z, z², |z|, 48 ch, 30 ep): VAL standalone **0.5212** (+0.002 from
  the extra channels), rank-corr **0.773**.

Both blended **identically** onto the shipped 0.6160 stack: flat at Wc=0.05
(0.6159/0.6160) then **monotone declining** through Wc=0.40 (0.6142), with *both*
honest halves down at every weight. The decorrelation is real (lower rank-corr
than the GRU's 0.85) but the standalone signal (~0.52) is **below the threshold
where decorrelation can lift a saturated blend** — so by the pre-agreed hard gate
(must beat VAL 0.6160 *and* OOS 0.5598), the CNN **does not ship** and the OOS
test is moot. `scripts/reduced_cnn.py` (the OOS evaluator) is built and ready but
was correctly never needed.

### Round-6 verdict
Six independent signal classes — hand-crafted subtle detectors, distributional
distances, a sequential change-point statistic, tree bagging, linear shape filters,
and a learned convolutional representation — **all confirm the same thing**: every
signal that decorrelates from the gradient-boosted trees is **already captured by
the 3-GRU sub-ensemble**, and the residual (subtle mean/distribution shifts) is at
the information-theoretic detection floor. **The model is comprehensively saturated
at VAL 0.6160 / OOS 0.5598 on the given inputs.** Round 6 ships **nothing** and
keeps round-5 `model_029`. Beating the #1 leaderboard score (~0.6322) from 0.6160
would require a fundamentally **different data source or label structure** (e.g. a
pretrained time-series foundation-model embedding bundled offline), not another
member built on the same calibrated feature stream — which is the single highest-EV
remaining idea but a large, separate effort. All negative results are preserved as
artifacts (`model_030_extratrees`, `model_031_cnn`) and scripts for reproducibility.


(Round-2 was 0.5812 → **+2.3 pts**; EWMA baseline 0.4806 → **+12.4 pts**.)
Honest 2-fold-on-VAL of the final blend: halves 0.6120 / 0.5968 (±~0.007 noise on 2000 series).

### Why v2 worked (the round-3 breakthrough)
EDA (`scripts/eda2.py`) showed the **null spread of sliding-window stats varies
2–3× across series** (std of window-50 mean-z ranges 0.53→1.83 series to series)
because the DGPs have heterogeneous serial dependence and tails. TS-AUC is a
*cross-sectional* ranking at fixed `t`, so a raw |z|=3 from a wandering series
must not outrank |z|=3 from a quiet one. **v2 measures each series' own null
loc/scale for every window statistic on its break-free historical segment (at
dyadic scales 8…512) and reports calibrated z/p units.** The top features by
gain are exactly these per-series null descriptors (`null_slv64`,
`null_mem_slope`, `null_sacf64`, `kurt_h`, `acf1_h`) plus the AR(1)-prewhitened
CUSUM bank and multiscale scan maxima — confirming the hypothesis directly.

### Round-3 findings that shaped the final model
1. **Per-series empirical-null calibration is THE lever** (+1.8 pts). Everything
   else is small by comparison. The mechanism: it removes the cross-sectional
   scale heterogeneity that TS-AUC punishes. Implemented in `src/sb/features2.py`.
2. **v3/v4 extra feature families help the *ensemble*, not the single model.**
   Calibrated derived streams (v3: |z|, Δz, median-crossing) and calibrated
   higher moments (v4: skew/kurt) each *lower* the single-model VAL by ~0.003
   (more noisy features) but are the best **decorrelators** (corr 0.97 vs 0.99
   among v2 seeds), so they lift the blend.
3. **Learned stacking is a trap here** (`scripts/stack_eval.py`): a logistic or
   GBT meta-learner on base logits scored **0.59 / 0.55** out of sample vs
   **0.6025** for equal mean-logit. TS-AUC is a *ranking* metric; per-row logloss
   meta-fitting misaligns with it and overfits. → **equal mean-logit blending.**
4. **Model-class diversity beats more seeds.** A logistic-regression member
   (VAL 0.578, but correlation only 0.93 vs 0.97–0.99 GBT–GBT) lifts the blend
   0.6031 → **0.6041** at weight 0.2. Cheap, deterministic, kept.
5. **Capacity, step-weighting, and aggressive feature_fraction all hurt** — same
   as round 2. Shallow trees (leaves15) are competitive and add diversity.
6. **Postprocessing (running-max / EWMA smoothing) does not help** (`diagnose.py`):
   raw per-step scores are already well-ordered; forcing monotonicity costs ~0.003.

### Where the model is still weak (diagnostics, `scripts/diagnose.py`)
- **By break age**: fresh breaks (elapsed < 25 steps) ≈ 0.52–0.53 AUC —
  near-undetectable, a *structural* ceiling (tau ~ Uniform means a large share of
  "broken" rows broke just before `t`). Mature breaks (elapsed > 250) ≈ 0.65.
- **By break type**: variance 0.66, acf 0.67, but **subtle/distribution-only
  ≈ 0.56** and it is 68% of breaks — the dominant addressable weakness. v4's
  calibrated skew/kurt nudged this but did not crack it (subtle breaks have
  median KS ≈ 0.12, near the two-sample detection limit even at W=400).
- **Early-mid steps** (t<100) ≈ 0.52–0.55 but carry less metric weight.

## R4 — round 4: GRU neural sub-ensemble (model-class diversity)

The round-3 ceiling was GBT saturation (all 10 GBTs corr 0.97–0.99). The logistic
member (corr 0.93, +0.001) proved *model-class* decorrelation is the remaining
lever, so round 4 adds a recurrent net — a genuinely different inductive bias that
**integrates break evidence over time** rather than scoring each step i.i.d.

| Exp | Change | VAL TS-AUC | Decision |
| --- | --- | --- | --- |
| model_020_gru | 1-layer GRU **hidden 128**, seed 0, over the 152-feat calibrated stream (drop pure-time/chi2), elapsed-ramp BCE, best-epoch-by-VAL, cosine LR | standalone **0.6048** (rank-corr 0.829) | ✓ decorrelated member |
| model_021_gru | GRU **hidden 96**, seed 1, dropout 0.1 | standalone 0.5998 (corr 0.831) | ✓ seed/arch diversity |
| model_022_gru | GRU **hidden 160**, seed 2, dropout 0.2 | standalone 0.6046 (corr 0.836) | ✓ seed/arch diversity |
| blend base+020 | base 0.6041 + single GRU at w=0.35 | **0.6169** | ✓ +0.0128, honest halves 0.6238/0.6101 both up |
| AVG member | mean of the 3 GRU step-logits (= ship config) | standalone **0.6069** (corr 0.847) | ✓ averaging denoises the neural member |
| W_GRU sweep | base + AVG member, weight 0.15→0.55 | flat **0.6157–0.6161** @ 0.35–0.50 | ✓ robust plateau, ship **W_GRU=0.40** |
| **model_023** | **FINAL: base (4 GBT + 0.2·logistic) `0.6·base + 0.4·mean(3 GRU)`** | **0.6161** | ✓ **round-4 best, SHIPPED** |

**Round-4 best: blended VAL 0.6161 (clears the ~0.6135 top-10 cutoff).**
(Round-3 was 0.6041 → **+1.2 pts**; EWMA baseline 0.4806 → **+13.5 pts**.)
Honest 2-fold-on-VAL of the AVG blend at W_GRU=0.40: halves **0.6211 / 0.6112**
(both well above the 0.6041 base — a real lift, not in-sample noise).
**Out-of-sample check** (`scripts/reduced_ab.py`, 100-series reduced test, ids ≥
10000 the GRU never trained on): base-only **0.5490** → base+GRU@0.40 **0.5595**
(**+0.0105**) — same direction and magnitude as the VAL lift, so the gain
generalises beyond the VAL split. Determinism (`crunch test` runner re-run):
max|diff| = **0.0**.

### Round-4 findings
1. **A recurrent member breaks the GBT ceiling.** The GRU rank-corr to the base is
   **~0.83** (vs 0.97–0.99 GBT–GBT, 0.93 logistic) — the most decorrelated member
   yet — because integrating evidence over time is a different mechanism from a
   per-step tree. Single seed lifts the blend +0.0128; that is the round-4 win.
2. **The GRU overfits the ranking metric past ~epoch 7.** All three seeds peak at
   VAL TS-AUC ~0.60–0.605 around epoch 7, then **decay to ~0.53 by epoch 40** while
   train loss keeps falling. Selecting the best epoch *by VAL TS-AUC* (not loss) is
   essential; 12–15 epochs suffice next time (saves compute for the 15 h/week OOS
   phase). This is the neural analogue of the round-2 "select by TS-AUC" lesson.
3. **Averaging seeds denoises the neural member.** The 3-seed average standalone
   (0.6069) beats every single seed (0.5998–0.6048) — neural nets are noisier than
   GBTs, so averaging recovers signal. It raises rank-corr to base slightly (0.847),
   so its *blended* peak (0.6161) is a hair under the single best seed's (0.6169)
   but with a **higher worst-half (0.6112 vs 0.6101)** and lower variance → the
   robust, defensible ship choice for unseen leaderboard data.
4. **The blend weight is not fragile.** base+AVG is flat at 0.6157–0.6161 across
   W_GRU 0.35–0.50 and the worst-half keeps rising through 0.50 — so W_GRU=0.40
   (the full optimum, mid-plateau) is safe.
5. **Determinism via numpy.** The GRU is trained in PyTorch (MPS) but **shipped as
   an exact single-layer GRU recurrence in float64 numpy** (weights base64-embedded
   in `main.py`); numpy↔torch parity ~5e-7, but since we *ship the numpy path*,
   determinism is exact (smoke re-run max|diff| = 0.0). `train()` still rebuilds the
   GBTs+logistic deterministically (the cloud has no torch).

### Round-4 pipeline (reproduce)
```bash
# train the 3 GRU seeds offline (PyTorch/MPS), ~12-15 epochs is enough:
uv run python scripts/train_seq.py --hidden 128 --seed 0 --out artifacts/models/model_020_gru --val-out features/val_seq_logits.npz
uv run python scripts/train_seq.py --hidden 96  --seed 1 --out artifacts/models/model_021_gru --val-out features/val_seq_logits_s1.npz
uv run python scripts/train_seq.py --hidden 160 --seed 2 --out artifacts/models/model_022_gru --val-out features/val_seq_logits_s2.npz
uv run python scripts/eval_base.py                                  # cache base 0.6041 VAL logits
uv run python scripts/blend_seq.py --avg features/val_seq_logits*.npz   # AVG member blend
uv run python scripts/tune_wgru.py                                  # W_GRU sensitivity (pick 0.40)
SB_W_GRU=0.40 SB_GRU_DIRS="artifacts/models/model_020_gru,artifacts/models/model_021_gru,artifacts/models/model_022_gru" \
  uv run python scripts/make_submission4.py                         # regenerate submission/main.py
uv run python submission/local_test.py                             # end-to-end + determinism (1e-8)
```

## Experiment artifact structure rule

- Save every trained iteration under `artifacts/models/model_XXX/` (incremental id).
- Example progression:
  - `artifacts/models/model_001/lgbm.txt`
  - `artifacts/models/model_002/lgbm.txt`
  - `artifacts/models/model_003/lgbm.txt`
- Use:
  - `uv run python scripts/train.py --model-id model_00X`
- `model/lgbm.txt` is kept as a latest-compatibility copy, but canonical history is
  the versioned folders in `artifacts/models/`.

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

## Round 3 detailed log (per-series null calibration → 0.6041)

### v2 (`src/sb/features2.py`, model_003) — per-series empirical-null calibration
For every trailing-window statistic (mean-z, log-var, acf1-diff, KS, eCDF-L1) we
measure the **series' own null loc/scale** on the break-free historical segment
at dyadic window scales (8,16,…,512), then report the online statistic in
calibrated units, interpolated at the window's current fill. Plus: multiscale
**scan** (max calibrated stat over window sizes) and its **running-max**;
**AR(1)-prewhitened** CUSUM/EWMA/variance bank (correctly calibrated for
autocorrelated series); and per-series **null-descriptor constants** (`acf1_h`,
`kurt_h`, null spreads at scale 64, long-memory slope). 117 used feats, seed 42:
**VAL 0.5988** (+1.76 over round-2's 0.5812). Build is sharded
(`scripts/build_features2.py --shard k --n-shards 4`) to stay in RAM.

### v3 (`src/sb/features3.py`, model_010-012) — calibrated derived streams
Adds calibrated trailing-window mean/acf of the |z|, Δz and median-crossing
streams (volatility clustering, smoothness, oscillation-rate changes) + scan +
running-max + 2 constants. Single-model **VAL 0.595/0.593/0.587** (seeds
42/7/2026) — *below* v2, but decorrelated.

### v4 (`src/sb/features4.py`, model_015-016) — calibrated higher moments
Adds null-calibrated window **skewness & excess kurtosis** (windows 100/200/400/
800) targeting the 68% subtle/shape breaks + scan + running-max + a tail-heaviness
constant. Single-model **VAL 0.596/0.592**; the calibrated skew/kurt features
rank in the **top-12 by gain**, so the signal is real even if the net single-model
effect is small.

### Stacking vs averaging (`scripts/stack_eval.py`, honest 2-fold-on-VAL)
equal mean-logit **0.6025**; logreg meta **0.5919**; gbm meta **0.5522**.
→ learned stacking overfits the ranking metric; **equal mean-logit** chosen.

### Ensemble search (`scripts/ensemble_search.py`, `scripts/combo_eval.py`)
All 10 GBTs correlate **0.97–0.99**. Best diverse blend = **{003,008,009 (v2) +
015 (v4)} = 0.6031** (v4 is the best decorrelator). Adding weaker v3 seeds dilutes.

### Logistic member (`scripts/final_blend.py`, model_017)
Logistic on v2 feats: VAL 0.578 but **corr 0.93** with GBTs. Blend
`0.8·mean(GBT logits) + 0.2·logistic logit` → **VAL 0.6041** (honest halves
0.612/0.597). Deterministic (frozen scaler+coef). This is the shipped model.

### v5 (`src/sb/features5.py`, model_019) — LRV-calibrated cumulative detectors (REJECTED)
Hypothesis: `log_t` dominates feature gain because the cumulative detectors
(cum mean-z, CUSUM bank, Page-Hinkley) grow with both time and dependence;
calibrate them by the per-series **long-run variance** `LRV = 1+2Σ wₖ·acfₖ`
(Bartlett-tapered, from historical) so `cum_mean_z/√LRV ~ N(0,1)` regardless of
autocorrelation. Single-model **VAL 0.5879 — worse than v2 (0.5988)**. The
**AR(1)-prewhitened CUSUM/EWMA bank already in v2 captures the dependence
calibration**, so the explicit LRV features are redundant and add noise.
→ **rejected; not in the shipped model.** (Negative result kept for the record.)

### FINAL (model_018) — shipped
`submission/main.py` (generated by `scripts/make_submission3.py`) inlines the full
v2→v3→v4 extractor (162 feats, exact parity verified) and a 4-GBT + logistic
mean-logit ensemble. train() rebuilds everything deterministically in-cloud;
infer() streams O(1) per step. **Smoke test: determinism max|diff| = 0.0**,
~0.4 ms/point → ~48 min for the full 10k-series test (≪ 15 h).
**Memory bug found & fixed:** the logistic step `X[:,keep].astype(float64)` on the
full 5M×117 matrix OOM-killed the 15 GB box; now it trains on a deterministic
stride-5 row subsample (~1M rows) in float32 → peak **4.2 GB** (verified). The
4 GBTs train comfortably (peak ~6 GB).

---

## Current best (round 2) — SUPERSEDED by round 3

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

### Honest gap (round 3)
Final VAL **0.6041** vs top-10 cutoff ~0.6135 → **~+0.9 pt short**. Round 3 closed
most of the round-2 gap (+2.3 pts, 0.581 → 0.604) via per-series null calibration.
The remaining gap is small but real; GBT ensembling is exhausted (correlations
0.97–0.99) so the next pts must come from a genuinely different representation.

---

## Ideas not yet tried (to break past ~0.604 toward 0.6135+)
*Ranked by expected value given the round-3 diagnostics.*
1. **Calibrate the CUMULATIVE detectors** (CUSUM / Page-Hinkley / cumulative
   mean-z) by their per-series H0 growth, the same way v2 calibrated *windows*.
   These are the highest-gain features yet still rely on `log_t` to discount
   their growth. Measure the null running-max trajectory on historical blocks and
   standardise. Same proven family as the +1.8-pt v2 win → **highest EV, ~O(1).**
2. **A neural sequence member** (tiny temporal CNN/GRU over the calibrated
   feature stream, or a 1-D CNN on the raw online window). Very different
   inductive bias from GBTs/linear → the decorrelation that actually moves a
   saturated ensemble (logistic already showed corr 0.93 helps). Deterministic
   inference if seeded + single-threaded.
3. **TSFM drift embeddings** (Chronos/Moirai/TimesFM): historical-vs-trailing
   embedding distance. Bundle weights (no cloud internet); INT8/ONNX for the
   15 h budget. Biggest potential lift, biggest effort — do last.
4. **Energy distance / MMD** and **spectral-slope** change features, calibrated
   per series (subtle 68% cohort and the 20% acf cohort). Lower EV — KS/L1/skew/
   kurt already calibrated and correlated with these.
5. **Per-break-type expert models + gate** (mean/var/acf/subtle) — epistemic
   diversity, as the 2025 winners used; more plumbing than the above.

### What NOT to retry (settled, with evidence)
- Learned stacking / meta-learners (overfit the ranking metric, §stack_eval).
- Bigger trees / leaves>31, aggressive feature_fraction, per-step metric
  weighting, scale_pos_weight (all hurt, rounds 2 & 3).
- Score postprocessing (running-max / smoothing) — hurts.
- Raw (uncalibrated) higher-order or distance features — only the *calibrated*
  versions carry cross-sectional rank signal.
- **LRV-calibrated cumulative detectors (v5, model_019 = 0.5879)** — redundant
  with the AR(1)-prewhitened bank; adds noise. (idea #1 in the old list — tried.)
- **Step-conditional / cross-sectional-population standardization** (round 9
  stepnorm) — baking train per-step null μ/σ leaks the train population; VAL
  mirage (+0.0085), OOS regresses (−0.0038). Any transform encoding train
  cross-sectional structure fails OOS.
- **`log_t` (and any absolute-position/length feature)** — REMOVED round 9b.
  At a fixed online-step it varies only with history length = a pure
  cross-sectional length signal that does not transfer (OOS +0.0203 once
  dropped from every member). **Lesson: any feature whose value at a fixed
  online-step depends on absolute position/length must be OOS-gated; prefer
  relative-position features like `t_over_nhist`.**
