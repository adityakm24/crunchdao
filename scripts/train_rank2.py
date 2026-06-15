"""Round 8 — LightGBM LambdaRank grouped by online step, on the v4 features.

The official metric (TS-AUC) is the per-online-step cross-sectional AUC. The 4
shipped GBTs are trained POINTWISE (objective=binary, BCE) — they optimise
calibration, not ranking. A LambdaRank model whose query group is the online
step index optimises within-step ranking of the break label, a direct surrogate
for the metric. This trains on the SAME proven v4 calibrated features as the
shipped v4 GBT (model_015), with TS-AUC early stopping, then caches VAL rank
scores aligned for the incremental-blend gate (scripts/eval_rank.py).

Deployable with ZERO new dependencies (still pure LightGBM).

Run: uv run python scripts/train_rank2.py [--xendcg] [--trunc 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import lightgbm as lgb

from sb.metric import ts_auc_grouped

ARTIFACTS_ROOT = os.path.join("artifacts", "models")
SPLIT_SEED = 42
FEATURES = "features/train_features_v4.npz"

# Same default drop as the pointwise v2/v4 GBTs (pure-time + rejected feats).
DEFAULT_DROP = {
    "t", "log_n_hist",
    "chi2_cum", "online_acf2", "acf2_diff",
    "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2",
}
NUM_ROUNDS = 1500
EARLY_STOP = 120


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="model_032_rank")
    ap.add_argument("--xendcg", action="store_true", help="use rank_xendcg objective")
    ap.add_argument("--trunc", type=int, default=200, help="lambdarank truncation level")
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--min-data", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-logt", action="store_true",
                    help="also drop log_t (cross-sectional length leakage, round 9b)")
    args = ap.parse_args()

    params = dict(
        objective="rank_xendcg" if args.xendcg else "lambdarank",
        metric="None",
        boosting_type="gbdt",
        num_leaves=args.leaves,
        learning_rate=args.lr,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=args.min_data,
        lambdarank_truncation_level=args.trunc,
        label_gain=list(range(2)),  # [0, 1] binary relevance
        num_threads=0,
        seed=args.seed,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )

    drop = set(DEFAULT_DROP) | ({"log_t"} if args.no_logt else set())

    d = np.load(FEATURES, allow_pickle=True)
    X, y, sid, t_online = d["X"], d["y"], d["series_id"], d["t_online"]
    names = [str(n) for n in d["feature_names"]]
    keep_idx = [i for i, n in enumerate(names) if n not in drop]
    keep_names = [names[i] for i in keep_idx]
    X = X[:, keep_idx]
    print(f"X={X.shape} pos_rate={y.mean():.4f}  using {len(keep_names)} feats  "
          f"obj={params['objective']} trunc={args.trunc}")

    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    n_val = int(0.2 * len(uniq))
    val_ids = set(uniq[:n_val].tolist())
    va = np.isin(sid, list(val_ids))
    tr = ~va

    def sort_by_step(mask):
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(t_online[idx], kind="stable")]
        _, counts = np.unique(t_online[order], return_counts=True)
        return order, counts.astype(np.int64)

    tr_order, gtr = sort_by_step(tr)
    va_order, gva = sort_by_step(va)
    print(f"train rows={len(tr_order):,} ({len(gtr)} step-groups)  "
          f"val rows={len(va_order):,} ({len(gva)} step-groups)")

    dtrain = lgb.Dataset(X[tr_order], label=y[tr_order], group=gtr,
                         feature_name=list(keep_names))
    dval = lgb.Dataset(X[va_order], label=y[va_order], group=gva, reference=dtrain)

    tva = t_online[va_order]
    yva_i = y[va_order].astype(np.int64)

    def ts_auc_feval(preds, _):
        return ("ts_auc", ts_auc_grouped(tva, yva_i, preds), True)

    t0 = time.time()
    booster = lgb.train(
        params, dtrain, num_boost_round=NUM_ROUNDS,
        valid_sets=[dval], valid_names=["val"], feval=ts_auc_feval,
        callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True),
                   lgb.log_evaluation(50)],
    )
    print(f"trained in {time.time()-t0:.1f}s, best_iter={booster.best_iteration}")

    out_dir = os.path.join(ARTIFACTS_ROOT, args.model_id)
    os.makedirs(out_dir, exist_ok=True)
    booster.save_model(os.path.join(out_dir, "lgbm_rank.txt"),
                       num_iteration=booster.best_iteration)

    pred_va = booster.predict(X[va_order], num_iteration=booster.best_iteration)
    va_tsauc = ts_auc_grouped(tva, yva_i, pred_va)
    print(f"\n>>> [RANK] Held-out VAL TS-AUC : {va_tsauc:.4f}")

    # cache val rank scores with (series_id, t_online) for alignment in eval_rank
    np.savez("features/val_rank_logits.npz",
             series_id=sid[va_order], t_online=t_online[va_order],
             rank_score=pred_va.astype(np.float64))

    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(dict(model_id=args.model_id, objective=params["objective"],
                       trunc=args.trunc, n_features=len(keep_names),
                       best_iter=int(booster.best_iteration or NUM_ROUNDS),
                       val_ts_auc=round(float(va_tsauc), 5)), fh, indent=2)

    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]
    print("\nTop features by gain:")
    for i in order[:15]:
        print(f"  {keep_names[i]:24s} {imp[i]:.0f}")


if __name__ == "__main__":
    main()
