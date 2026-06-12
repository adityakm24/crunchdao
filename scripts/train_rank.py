"""Train a LightGBM LambdaRank model grouped by online step.

Each "query group" is one online step index t: all (series) rows sharing that
step compete. Optimizing within-group ranking of the binary break label is a
direct surrogate for the per-step cross-sectional AUC that TS-AUC averages.

Compares against the pointwise model and evaluates held-out + reduced-test TS-AUC.

Run:  uv run python scripts/train_rank.py
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

DROP_FEATURES = {"t", "log_t", "log_n_hist", "t_over_nhist"}
KEEP_IDX = [i for i, n in enumerate(FEATURE_NAMES) if n not in DROP_FEATURES]
KEEP_NAMES = [FEATURE_NAMES[i] for i in KEEP_IDX]

PARAMS = dict(
    objective="lambdarank",
    metric="auc",
    boosting_type="gbdt",
    num_leaves=63,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=200,
    lambdarank_truncation_level=50,
    label_gain=[0, 1],   # binary relevance
    num_threads=0,
    seed=SEED,
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,
)
NUM_ROUNDS = 1000
EARLY_STOP = 60


def sort_by_group(X, y, t_online):
    """Return arrays sorted by online step plus group sizes."""
    order = np.argsort(t_online, kind="stable")
    Xs, ys, ts = X[order], y[order], t_online[order]
    _, counts = np.unique(ts, return_counts=True)
    return Xs, ys, ts, counts.astype(np.int64)


def build_test_features():
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
    pred = booster.predict(Xte[:, KEEP_IDX])
    ytdf = load_test_targets().reset_index().sort_values(["id", "time"])
    ytdf["t_online"] = ytdf.groupby("id").cumcount()
    key = ytdf.set_index(["id", "t_online"])["target"]
    y = key.loc[list(zip(sid_te.tolist(), t_te.tolist()))].to_numpy()
    return ts_auc_grouped(t_te, y.astype(np.int64), pred)


def main() -> None:
    d = np.load("features/train_features.npz", allow_pickle=True)
    X, y, sid, t_online = d["X"][:, KEEP_IDX], d["y"], d["series_id"], d["t_online"]
    print(f"Loaded X={X.shape} pos_rate={y.mean():.4f}  ({len(KEEP_NAMES)} content feats)")

    uniq = np.unique(sid)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    n_val = int(0.2 * len(uniq))
    val_ids = set(uniq[:n_val].tolist())
    va = np.isin(sid, list(val_ids))
    tr = ~va

    Xtr, ytr, ttr, gtr = sort_by_group(X[tr], y[tr], t_online[tr])
    Xva, yva, tva, gva = sort_by_group(X[va], y[va], t_online[va])
    print(f"train rows={Xtr.shape[0]:,} ({len(gtr)} step-groups)  "
          f"val rows={Xva.shape[0]:,} ({len(gva)} step-groups)")

    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=list(KEEP_NAMES))
    dval = lgb.Dataset(Xva, label=yva, group=gva, reference=dtrain)

    t0 = time.time()
    booster = lgb.train(
        PARAMS, dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(50)],
    )
    print(f"trained in {time.time()-t0:.1f}s, best_iter={booster.best_iteration}")

    pred_va = booster.predict(Xva)
    va_tsauc = ts_auc_grouped(tva, yva.astype(np.int64), pred_va)
    print(f"\n>>> [RANK] Held-out VAL TS-AUC : {va_tsauc:.4f}")
    te_tsauc = test_set_ts_auc(booster)
    print(f">>> [RANK] Reduced TEST TS-AUC : {te_tsauc:.4f}  (baseline 0.4806)")

    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]
    print("\nTop features by gain:")
    for i in order[:15]:
        print(f"  {KEEP_NAMES[i]:24s} {imp[i]:.0f}")


if __name__ == "__main__":
    main()
