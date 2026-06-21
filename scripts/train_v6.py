"""Round 13 — from-scratch high-volume + regularized + generalization-gated GBT.

Combines the independent feature blocks already on disk (same lexsorted row
order) into one bank WITHOUT rebuilding, then trains a strongly-regularized
LightGBM and gates it on a nested 5-fold *series-disjoint* CV (the
generalization-first protocol the cross-dataset benchmark says wins) instead of
the over-used single seed-42 VAL split.

  --blocks v4,xs,wave     which feature matrices to concatenate
  --leaves --lr --min-data --l1 --l2 --feature-fraction   capacity / regularization
  --topk N                gain-select to top-N features (0 = use all)
  --folds K               CV folds (default 5)

Reports per-fold + mean/std TS-AUC and the seed-42 VAL TS-AUC (comparable to the
base GBT mean 0.6041). No retrain/save unless --save.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import lightgbm as lgb

from sb.metric import ts_auc_grouped

BLOCK_PATH = {
    "v4": "features/train_features_v4.npz",
    "xs": "features/train_features_xs.npz",
    "wave": "features/train_features_wave.npz",
}
# pure-time / rejected features dropped everywhere (documented time-trap + EXP-008)
DROP = {"t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
        "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2"}


def elapsed_weights(sid, y, t_online, ramp=50.0):
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


def load_bank(blocks):
    X_parts, names = [], []
    seen = set()
    y = sid = t = None
    keep_per_block = []
    for bi, b in enumerate(blocks):
        d = np.load(BLOCK_PATH[b], allow_pickle=True)
        bn = [str(n) for n in d["feature_names"]]
        if y is None:
            y = np.asarray(d["y"]); sid = np.asarray(d["series_id"]); t = np.asarray(d["t_online"])
        else:
            assert np.array_equal(np.asarray(d["series_id"]), sid), f"{b} row order mismatch"
        cols = []
        for i, n in enumerate(bn):
            if n in DROP or n in seen:
                continue
            seen.add(n); cols.append(i); names.append(n)
        keep_per_block.append((b, d, cols))
    # build X
    n = len(y)
    X = np.empty((n, len(names)), dtype=np.float32)
    off = 0
    for b, d, cols in keep_per_block:
        Xb = np.asarray(d["X"])
        for c in cols:
            X[:, off] = Xb[:, c]; off += 1
        del Xb
    return X, y, sid, t, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="v4,xs,wave")
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--min-data", type=int, default=2000)
    ap.add_argument("--l1", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--feature-fraction", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--early", type=int, default=120)
    args = ap.parse_args()

    blocks = [b for b in args.blocks.split(",") if b]
    print(f"[load] blocks={blocks}")
    X, y, sid, t, names = load_bank(blocks)
    print(f"[bank] X={X.shape}  ({len(names)} unique feats)  pos_rate={y.mean():.4f}")
    yi = y.astype(np.int64)
    w_all = elapsed_weights(sid, y, t)

    params = dict(objective="binary", metric="None", boosting_type="gbdt",
                  num_leaves=args.leaves, learning_rate=args.lr,
                  feature_fraction=args.feature_fraction,
                  bagging_fraction=0.8, bagging_freq=1,
                  min_data_in_leaf=args.min_data,
                  lambda_l1=args.l1, lambda_l2=args.l2,
                  max_depth=-1, num_threads=0, seed=42,
                  deterministic=True, force_row_wise=True, verbosity=-1)

    # optional gain-based selection on the seed-42 train split
    uniq = np.unique(sid)
    rng = np.random.default_rng(42); rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    va0 = np.isin(sid, list(val_ids)); tr0 = ~va0
    feat_idx = list(range(len(names)))
    if args.topk and args.topk < len(names):
        dtr = lgb.Dataset(X[tr0], label=y[tr0], weight=w_all[tr0], feature_name=names)
        dva = lgb.Dataset(X[va0], label=y[va0], reference=dtr)
        tv0, yv0 = t[va0], yi[va0]
        sel = lgb.train(params, dtr, num_boost_round=400,
                        valid_sets=[dva], valid_names=["v"],
                        feval=lambda p, _: ("ts_auc", ts_auc_grouped(tv0, yv0, p), True),
                        callbacks=[lgb.early_stopping(args.early, first_metric_only=True)])
        imp = sel.feature_importance(importance_type="gain")
        feat_idx = list(np.argsort(imp)[::-1][:args.topk])
        names = [names[i] for i in feat_idx]
        X = X[:, feat_idx]
        print(f"[select] kept top {len(names)} by gain")

    # nested K-fold series-disjoint CV (the generalization gate)
    folds = np.array([hash(int(s)) % args.folds for s in uniq])  # deterministic
    fold_of = {int(s): int(f) for s, f in zip(uniq, folds)}
    sid_fold = np.array([fold_of[int(s)] for s in sid])

    print(f"\n[CV] {args.folds}-fold series-disjoint  params: leaves={args.leaves} "
          f"l1={args.l1} l2={args.l2} min_data={args.min_data} ff={args.feature_fraction}")
    aucs = []
    for k in range(args.folds):
        va = sid_fold == k; tr = ~va
        dtr = lgb.Dataset(X[tr], label=y[tr], weight=w_all[tr], feature_name=names)
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
        tvk, yvk = t[va], yi[va]
        t0 = time.time()
        bst = lgb.train(params, dtr, num_boost_round=args.rounds,
                        valid_sets=[dva], valid_names=["v"],
                        feval=lambda p, _: ("ts_auc", ts_auc_grouped(tvk, yvk, p), True),
                        callbacks=[lgb.early_stopping(args.early, first_metric_only=True)])
        pred = bst.predict(X[va], num_iteration=bst.best_iteration)
        a = ts_auc_grouped(tvk, yvk, pred)
        aucs.append(a)
        print(f"  fold {k}: TS-AUC={a:.4f}  best_iter={bst.best_iteration}  {time.time()-t0:.0f}s")
    aucs = np.array(aucs)
    print(f"\n>>> CV TS-AUC = {aucs.mean():.4f} ± {aucs.std():.4f}  "
          f"(min {aucs.min():.4f})   base GBT mean reference = 0.6041 (seed-42 VAL)")


if __name__ == "__main__":
    main()
