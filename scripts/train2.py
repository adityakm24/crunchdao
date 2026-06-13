"""Train v2 LightGBM break detector with TS-AUC selection + ablation switches.

Same protocol as train.py (group split by series, TS-AUC feval selection,
elapsed-ramp positive weighting, full retrain at the end) with switches for
the round-3 experiments:

  --features      path to the feature npz (default v2 matrix)
  --drop          extra comma-separated feature names to drop
  --keep-only     regex; keep only matching features (after the default drop)
  --step-weight   also multiply weights by the per-step metric weight
                  n_pos(t)*n_neg(t) (normalised to mean 1) computed on train
  --ramp          elapsed ramp length (default 50; 0 disables)
  --seed          LightGBM/split seed (default 42; split stays 42 always so
                  experiments remain comparable)
  --leaves/--lr/--min-data  capacity overrides
  --no-retrain    skip the full retrain (ablation runs)

Run:  uv run python scripts/train2.py --model-id model_003
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import lightgbm as lgb

from sb.metric import ts_auc_grouped

MODEL_DIR = "model"
ARTIFACTS_ROOT = os.path.join("artifacts", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_ROOT, exist_ok=True)
SPLIT_SEED = 42

# Same default drop as v1: pure-time features carry no cross-sectional rank
# information; chi2 / lag-2 acf were rejected in EXP-008.
DEFAULT_DROP = {
    "t", "log_n_hist",
    "chi2_cum", "online_acf2", "acf2_diff",
    "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2",
}

NUM_ROUNDS = 1500
EARLY_STOP = 120


def elapsed_weights(sid, y, t_online, ramp):
    w = np.ones(len(y), dtype=np.float64)
    if ramp <= 0:
        return w
    order = np.lexsort((t_online, sid))
    s_s, y_s, t_s = sid[order], y[order], t_online[order]
    w_s = np.ones(len(order))
    i, nrow = 0, len(order)
    while i < nrow:
        j = i
        while j < nrow and s_s[j] == s_s[i]:
            j += 1
        yy, tt = y_s[i:j], t_s[i:j]
        if yy.any():
            tau = tt[yy.argmax()]
            mask = yy == 1
            ww = np.ones(j - i)
            ww[mask] = np.clip((tt[mask] - tau) / ramp, 0.2, 1.0)
            w_s[i:j] = ww
        i = j
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return w_s[inv]


def step_metric_weights(y, t_online):
    """Per-row weight = n_pos(t)*n_neg(t) of the row's step, mean-normalised."""
    t = t_online.astype(np.int64)
    n_max = int(t.max()) + 1
    pos = np.bincount(t, weights=(y == 1).astype(np.float64), minlength=n_max)
    tot = np.bincount(t, minlength=n_max).astype(np.float64)
    neg = tot - pos
    w_step = pos * neg
    w_row = w_step[t]
    m = w_row.mean()
    return w_row / m if m > 0 else np.ones_like(w_row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--features", default="features/train_features_v2.npz")
    ap.add_argument("--drop", default="")
    ap.add_argument("--keep-only", default="")
    ap.add_argument("--step-weight", action="store_true")
    ap.add_argument("--ramp", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--min-data", type=int, default=1000)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--no-retrain", action="store_true")
    args = ap.parse_args()

    params = dict(
        objective="binary", metric="None", boosting_type="gbdt",
        num_leaves=args.leaves, learning_rate=args.lr,
        feature_fraction=args.feature_fraction,
        bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=args.min_data, max_depth=-1, num_threads=0,
        seed=args.seed, deterministic=True, force_row_wise=True, verbosity=-1,
    )

    model_dir_iter = os.path.join(ARTIFACTS_ROOT, args.model_id)
    os.makedirs(model_dir_iter, exist_ok=True)

    d = np.load(args.features, allow_pickle=True)
    X, y, sid, t_online = d["X"], d["y"], d["series_id"], d["t_online"]
    names = [str(n) for n in d["feature_names"]]

    drop = set(DEFAULT_DROP) | {s for s in args.drop.split(",") if s}
    keep_idx = [i for i, n in enumerate(names) if n not in drop]
    if args.keep_only:
        pat = re.compile(args.keep_only)
        keep_idx = [i for i in keep_idx if pat.search(names[i])]
    keep_names = [names[i] for i in keep_idx]
    X = X[:, keep_idx]
    print(f"X={X.shape} pos_rate={y.mean():.4f}  using {len(keep_names)} features")

    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    n_val = int(0.2 * len(uniq))
    val_ids = set(uniq[:n_val].tolist())
    va = np.isin(sid, list(val_ids))
    tr = ~va
    print(f"train rows={tr.sum():,}  val rows={va.sum():,}")

    w_all = elapsed_weights(sid, y, t_online, args.ramp)
    if args.step_weight:
        w_all = w_all * step_metric_weights(y, t_online)
        print("applied per-step metric weights")

    dtrain = lgb.Dataset(X[tr], label=y[tr], weight=w_all[tr],
                         feature_name=list(keep_names))
    dval = lgb.Dataset(X[va], label=y[va], reference=dtrain)

    tva = t_online[va]
    yva_i = y[va].astype(np.int64)

    def ts_auc_feval(preds, _):
        return ("ts_auc", ts_auc_grouped(tva, yva_i, preds), True)

    t0 = time.time()
    booster = lgb.train(
        params, dtrain, num_boost_round=NUM_ROUNDS,
        valid_sets=[dval], valid_names=["val"], feval=ts_auc_feval,
        callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True),
                   lgb.log_evaluation(100)],
    )
    print(f"trained in {time.time()-t0:.1f}s, best_iter={booster.best_iteration}")
    booster.save_model(os.path.join(model_dir_iter, "lgbm_valsplit.txt"),
                       num_iteration=booster.best_iteration)

    pred_va = booster.predict(X[va], num_iteration=booster.best_iteration)
    va_tsauc = ts_auc_grouped(t_online[va], y[va].astype(np.int64), pred_va)
    print(f"\n>>> Held-out VAL TS-AUC : {va_tsauc:.4f}")

    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]
    print("\nTop features by gain:")
    for i in order[:25]:
        print(f"  {keep_names[i]:24s} {imp[i]:.0f}")

    meta = dict(
        model_id=args.model_id, features=args.features,
        n_features=len(keep_names), drop=sorted(drop),
        keep_only=args.keep_only, step_weight=args.step_weight,
        ramp=args.ramp, seed=args.seed, leaves=args.leaves, lr=args.lr,
        min_data=args.min_data, feature_fraction=args.feature_fraction,
        best_iter=int(booster.best_iteration or NUM_ROUNDS),
        val_ts_auc=round(float(va_tsauc), 5),
    )
    with open(os.path.join(model_dir_iter, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if args.no_retrain:
        print("(--no-retrain: skipping full retrain)")
        return

    print("\nRetraining on ALL series for final model ...")
    best_iter = booster.best_iteration or NUM_ROUNDS
    dall = lgb.Dataset(X, label=y, weight=w_all, feature_name=list(keep_names))
    final = lgb.train(params, dall, num_boost_round=best_iter)
    iter_model_path = os.path.join(model_dir_iter, "lgbm.txt")
    final.save_model(iter_model_path)
    shutil.copy2(iter_model_path, os.path.join(MODEL_DIR, "lgbm.txt"))
    print(f"Saved {iter_model_path} (n_trees={final.num_trees()})")


if __name__ == "__main__":
    main()
