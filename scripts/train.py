"""Train LightGBM break detector and evaluate with TS-AUC.

Steps:
  1. Load the prebuilt feature matrix.
  2. Group-split by series_id (no series leaks between train/val).
  3. Train LightGBM (binary) with early stopping on grouped val TS-AUC proxy.
  4. Report held-out val TS-AUC.
  5. Build features for the reduced local test set (100 series) and report its
     TS-AUC -- the closest local proxy to the public leaderboard.
  6. Retrain on ALL training series and save the final model to model/.

Run:  uv run python scripts/train.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import lightgbm as lgb

from sb.data import iter_series, load_test_targets
from sb.features import StreamingDetector, FEATURE_NAMES, N_FEATURES
from sb.metric import ts_auc_grouped

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
SEED = 42

# Pure-time features carry NO cross-sectional information (they are identical
# across all series at a fixed step t), so they cannot help TS-AUC and only
# divert model capacity. Drop them and keep only series-content signal.
# Drop raw `t` and `log_n_hist` (keep log_t & t_over_nhist for time-normalisation
# via interactions), and the chi2 / lag-2-acf additions which empirically hurt.
DROP_FEATURES = {
    "t", "log_n_hist",
    "chi2_cum", "online_acf2", "acf2_diff",
    "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2",
}
KEEP_IDX = [i for i, n in enumerate(FEATURE_NAMES) if n not in DROP_FEATURES]
KEEP_NAMES = [FEATURE_NAMES[i] for i in KEEP_IDX]

PARAMS = dict(
    objective="binary",
    metric="None",  # selection is driven by the custom TS-AUC feval below
    boosting_type="gbdt",
    num_leaves=31,
    learning_rate=0.03,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=1000,
    max_depth=-1,
    num_threads=0,
    seed=SEED,
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,
)
NUM_ROUNDS = 1500
EARLY_STOP = 120


RAMP = 50.0  # down-weight near-undetectable fresh-break positives


def elapsed_weights(sid, y, t_online):
    """Weight positives by min(1, elapsed_since_break/RAMP) (floor 0.2); neg=1."""
    w = np.ones(len(y), dtype=np.float64)
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
            ww[mask] = np.clip((tt[mask] - tau) / RAMP, 0.2, 1.0)
            w_s[i:j] = ww
        i = j
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return w_s[inv]


def build_test_features():
    """Stream the reduced test set through the extractor -> (X, sid, t)."""
    det = StreamingDetector()
    Xs, ids, ts = [], [], []
    for s in iter_series("test"):
        det.calibrate(s.x_hist)
        T = s.n_online
        feats = np.empty((T, N_FEATURES), dtype=np.float32)
        for i, x in enumerate(s.x_online):
            feats[i] = det.update(float(x))
        Xs.append(feats)
        ids.append(np.full(T, s.series_id, dtype=np.int32))
        ts.append(np.arange(T, dtype=np.int32))
    return np.vstack(Xs), np.concatenate(ids), np.concatenate(ts)


def test_set_ts_auc(booster):
    Xte, sid_te, t_te = build_test_features()
    pred = booster.predict(Xte[:, KEEP_IDX], num_iteration=booster.best_iteration)
    ytdf = load_test_targets().reset_index()  # columns: id, time, target
    ytdf = ytdf.sort_values(["id", "time"])
    ytdf["t_online"] = ytdf.groupby("id").cumcount()
    # align: our (sid_te, t_te) order matches iter_series order; build lookup
    key = ytdf.set_index(["id", "t_online"])["target"]
    y = key.loc[list(zip(sid_te.tolist(), t_te.tolist()))].to_numpy()
    return ts_auc_grouped(t_te, y.astype(np.int64), pred)


def main() -> None:
    d = np.load("features/train_features.npz", allow_pickle=True)
    X, y, sid, t_online = d["X"], d["y"], d["series_id"], d["t_online"]
    X = X[:, KEEP_IDX]
    print(f"Loaded X={X.shape} pos_rate={y.mean():.4f}")
    print(f"Using {len(KEEP_NAMES)} content features (dropped time features: {sorted(DROP_FEATURES)})")

    uniq = np.unique(sid)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    n_val = int(0.2 * len(uniq))
    val_ids = set(uniq[:n_val].tolist())
    val_mask = np.isin(sid, list(val_ids))
    tr, va = ~val_mask, val_mask
    print(f"train rows={tr.sum():,}  val rows={va.sum():,}  "
          f"({len(uniq)-n_val} train series / {n_val} val series)")

    w_all = elapsed_weights(sid, y, t_online)
    dtrain = lgb.Dataset(X[tr], label=y[tr], weight=w_all[tr], feature_name=list(KEEP_NAMES))
    dval = lgb.Dataset(X[va], label=y[va], reference=dtrain)

    # Select iterations by TS-AUC (the competition metric), NOT pointwise AUC.
    tva = t_online[va]
    yva_i = y[va].astype(np.int64)

    def ts_auc_feval(preds, _dataset):
        return ("ts_auc", ts_auc_grouped(tva, yva_i, preds), True)

    t0 = time.time()
    booster = lgb.train(
        PARAMS, dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dval], valid_names=["val"],
        feval=ts_auc_feval,
        callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True),
                   lgb.log_evaluation(100)],
    )
    print(f"trained in {time.time()-t0:.1f}s, best_iter={booster.best_iteration}")

    # held-out val TS-AUC
    pred_va = booster.predict(X[va], num_iteration=booster.best_iteration)
    va_tsauc = ts_auc_grouped(t_online[va], y[va].astype(np.int64), pred_va)
    print(f"\n>>> Held-out VAL TS-AUC : {va_tsauc:.4f}")

    # reduced test set TS-AUC (closest leaderboard proxy)
    te_tsauc = test_set_ts_auc(booster)
    print(f">>> Reduced TEST TS-AUC : {te_tsauc:.4f}  (baseline EWMA = 0.4806)")

    # feature importance
    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]
    print("\nTop features by gain:")
    for i in order[:15]:
        print(f"  {KEEP_NAMES[i]:24s} {imp[i]:.0f}")

    # ---- retrain on ALL series for the final model ----
    print("\nRetraining on ALL series for final model ...")
    best_iter = booster.best_iteration or NUM_ROUNDS
    dall = lgb.Dataset(X, label=y, weight=w_all, feature_name=list(KEEP_NAMES))
    final = lgb.train(PARAMS, dall, num_boost_round=best_iter)
    final.save_model(os.path.join(MODEL_DIR, "lgbm.txt"))
    print(f"Saved {MODEL_DIR}/lgbm.txt (n_trees={final.num_trees()})")


if __name__ == "__main__":
    main()
