"""Quick hyperparameter sweep for the pointwise model, scored by held-out TS-AUC.

Loads the feature matrix once, fixes the group split, and evaluates a handful of
LightGBM configs. Reports val TS-AUC for each.

Run:  uv run python scripts/sweep.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

import numpy as np
import lightgbm as lgb

from sb.features import FEATURE_NAMES
from sb.metric import ts_auc_grouped

SEED = 42
DROP = {"t", "log_t", "log_n_hist", "t_over_nhist"}
KEEP_IDX = [i for i, n in enumerate(FEATURE_NAMES) if n not in DROP]

d = np.load("features/train_features.npz", allow_pickle=True)
X, y, sid, t_online = d["X"][:, KEEP_IDX], d["y"], d["series_id"], d["t_online"]

uniq = np.unique(sid)
rng = np.random.default_rng(SEED)
rng.shuffle(uniq)
n_val = int(0.2 * len(uniq))
va = np.isin(sid, list(set(uniq[:n_val].tolist())))
tr = ~va
dtrain = lgb.Dataset(X[tr], label=y[tr])
dval = lgb.Dataset(X[va], label=y[va], reference=dtrain)
tva, yva = t_online[va], y[va].astype(np.int64)

BASE = dict(objective="binary", metric="auc", boosting_type="gbdt",
            learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, num_threads=0, seed=SEED, deterministic=True,
            force_row_wise=True, verbosity=-1)

CONFIGS = {
    "EXP004 (leaves63,md200,lr.05)": dict(num_leaves=63, min_data_in_leaf=200),
    "leaves127,md500,lr.03": dict(num_leaves=127, min_data_in_leaf=500, learning_rate=0.03),
    "leaves255,md1000,lr.03": dict(num_leaves=255, min_data_in_leaf=1000, learning_rate=0.03),
    "leaves31,md1000,lr.03": dict(num_leaves=31, min_data_in_leaf=1000, learning_rate=0.03),
    "leaves63,md500,lr.03,ff.6": dict(num_leaves=63, min_data_in_leaf=500, learning_rate=0.03, feature_fraction=0.6),
}

for name, over in CONFIGS.items():
    p = {**BASE, **over}
    t0 = time.time()
    bst = lgb.train(p, dtrain, num_boost_round=2000,
                    valid_sets=[dval], valid_names=["val"],
                    callbacks=[lgb.early_stopping(80, verbose=False)])
    pred = bst.predict(X[va])
    tsauc = ts_auc_grouped(tva, yva, pred)
    print(f"{name:34s} best_iter={bst.best_iteration:4d} "
          f"val_auc={bst.best_score['val']['auc']:.4f} TS-AUC={tsauc:.4f} ({time.time()-t0:.0f}s)")
