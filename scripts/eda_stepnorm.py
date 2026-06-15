"""Round 8 probe — STEP-CONDITIONAL NULL standardization of the v4 features.

Hypothesis: per-series null calibration removes cross-SERIES heterogeneity, but a
trailing-window feature's sampling distribution still tightens as the online step
t grows (more data -> smaller null variance). A GBT pools rows across all t, so a
single split threshold is a compromise across these drifting per-step null scales.

Fix (serve-compatible, O(1)): learn from TRAIN the cross-sectional NULL mean/std
of each feature AT each online step t (rows with y==0 = break-not-yet-occurred),
smoothed over t, and transform x -> (x - mu0(t)) / sd0(t). This expresses every
feature in per-step null-sigma units, consistent across t. Using ONLY y==0 rows
avoids shrinking the break signal (broken rows don't inflate the scale). At serve
time it's a t-indexed lookup baked from train -> no cross-series info at inference.

This is NOT a within-step monotonic no-op: it aligns features across t, changing
what the pooled GBT learns. Probe trains a GBT with vs without the transform on
the SAME split and compares VAL TS-AUC. Clear lift -> productionize as a feature
version; flat -> drop and document.

Run: uv run python scripts/eda_stepnorm.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

import numpy as np
import lightgbm as lgb
from scipy.ndimage import uniform_filter1d

from sb.metric import ts_auc_grouped

FEATURES = "features/train_features_v4.npz"
SPLIT_SEED = 42
DEFAULT_DROP = {
    "t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
    "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2",
}
PARAMS = dict(
    objective="binary", metric="None", boosting_type="gbdt",
    num_leaves=31, learning_rate=0.03, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=1000,
    num_threads=0, seed=42, deterministic=True, force_row_wise=True,
    verbosity=-1,
)
NUM_ROUNDS = 600
EARLY_STOP = 80
SMOOTH = 25  # rolling window (in steps) for the per-step null stats


def step_null_stats(X, t, ynull_mask, T, n_feat):
    """Per-step cross-sectional NULL mean/std (y==0 rows), smoothed over t."""
    tn = t[ynull_mask]
    cnt = np.bincount(tn, minlength=T).astype(np.float64)
    cnt_safe = np.maximum(cnt, 1.0)
    mu = np.zeros((T, n_feat), dtype=np.float64)
    sd = np.ones((T, n_feat), dtype=np.float64)
    Xn = X[ynull_mask]
    for j in range(n_feat):
        col = Xn[:, j].astype(np.float64)
        s = np.bincount(tn, weights=col, minlength=T)
        sq = np.bincount(tn, weights=col * col, minlength=T)
        m = s / cnt_safe
        v = np.maximum(sq / cnt_safe - m * m, 0.0)
        mu[:, j] = m
        sd[:, j] = np.sqrt(v)
    # smooth across t (nearest-edge) to denoise sparse high-t steps
    mu = uniform_filter1d(mu, size=SMOOTH, axis=0, mode="nearest")
    sd = uniform_filter1d(sd, size=SMOOTH, axis=0, mode="nearest")
    sd[sd < 1e-6] = 1.0  # dead/constant-across-series feats (e.g. log_t) -> ~0
    return mu, sd


def train_eval(Xtr, ytr, Xva, yva, tva, names, tag):
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=list(names))
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    yva_i = yva.astype(np.int64)

    def feval(preds, _):
        return ("ts_auc", ts_auc_grouped(tva, yva_i, preds), True)

    t0 = time.time()
    bst = lgb.train(PARAMS, dtr, num_boost_round=NUM_ROUNDS,
                    valid_sets=[dva], valid_names=["val"], feval=feval,
                    callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True),
                               lgb.log_evaluation(0)])
    pred = bst.predict(Xva, num_iteration=bst.best_iteration)
    auc = ts_auc_grouped(tva, yva_i, pred)
    print(f"  [{tag}] VAL TS-AUC = {auc:.4f}  (best_iter {bst.best_iteration}, "
          f"{time.time()-t0:.0f}s)")
    return auc, bst


def main():
    d = np.load(FEATURES, allow_pickle=True)
    X, y, sid, t = d["X"], d["y"], d["series_id"], d["t_online"]
    names_all = [str(n) for n in d["feature_names"]]
    keep = [i for i, n in enumerate(names_all) if n not in DEFAULT_DROP]
    names = [names_all[i] for i in keep]
    X = np.ascontiguousarray(X[:, keep])
    y = y.astype(np.int32)
    T = int(t.max()) + 1
    print(f"X={X.shape}  feats={len(names)}  T={T}  pos_rate={y.mean():.4f}")

    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    va = np.isin(sid, list(val_ids))
    tr = ~va
    tva = t[va].astype(np.int64)
    print(f"train rows={tr.sum():,}  val rows={va.sum():,}")

    print("\n=== BASELINE (raw v4 features) ===")
    base_auc, _ = train_eval(X[tr], y[tr], X[va], y[va], tva, names, "raw")

    print("\n=== STEP-CONDITIONAL NULL STANDARDIZATION ===")
    t0 = time.time()
    null_tr = tr & (y == 0)
    mu, sd = step_null_stats(X, t, null_tr, T, X.shape[1])
    print(f"  computed per-step null stats from {null_tr.sum():,} y==0 train rows "
          f"({time.time()-t0:.0f}s); smoothing window {SMOOTH}")
    Xn = (X - mu[t]) / sd[t]
    np.clip(Xn, -12.0, 12.0, out=Xn)
    Xn = Xn.astype(np.float32)
    sn_auc, bst = train_eval(Xn[tr], y[tr], Xn[va], y[va], tva, names, "stepnorm")

    print("\n=== STACKED (raw ++ stepnorm, GBT picks) ===")
    Xc = np.concatenate([X, Xn], axis=1)
    names_c = names + [n + "__sn" for n in names]
    st_auc, _ = train_eval(Xc[tr], y[tr], Xc[va], y[va], tva, names_c, "stack")

    print(f"\nSUMMARY:  raw {base_auc:.4f} | stepnorm {sn_auc:.4f} "
          f"({sn_auc-base_auc:+.4f}) | stack {st_auc:.4f} ({st_auc-base_auc:+.4f})")
    imp = bst.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]
    print("Top stepnorm feats:", ", ".join(names[i] for i in order[:12]))


if __name__ == "__main__":
    main()
