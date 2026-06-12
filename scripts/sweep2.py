"""Capacity sweep on the winning feature set (drop {t, log_n_hist}), TS-AUC select."""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

import numpy as np
import lightgbm as lgb

from sb.features import FEATURE_NAMES
from sb.metric import ts_auc_grouped

SEED = 42
DROP = {"t", "log_n_hist"}
KEEP = [i for i, n in enumerate(FEATURE_NAMES) if n not in DROP]

d = np.load("features/train_features.npz", allow_pickle=True)
Xall, y, sid, t_online = d["X"][:, KEEP], d["y"], d["series_id"], d["t_online"]
uniq = np.unique(sid)
rng = np.random.default_rng(SEED)
rng.shuffle(uniq)
va = np.isin(sid, list(set(uniq[: int(0.2 * len(uniq))].tolist())))
tr = ~va
tva, yva_i = t_online[va], y[va].astype(np.int64)
dtrain = lgb.Dataset(Xall[tr], label=y[tr])
dval = lgb.Dataset(Xall[va], label=y[va], reference=dtrain)


def feval(preds, _ds):
    return ("ts_auc", ts_auc_grouped(tva, yva_i, preds), True)


BASE = dict(objective="binary", metric="None", boosting_type="gbdt",
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            num_threads=0, seed=SEED, deterministic=True, force_row_wise=True,
            verbosity=-1)

CONFIGS = {
    "leaves63 lr.05 md200": dict(num_leaves=63, learning_rate=0.05, min_data_in_leaf=200),
    "leaves127 lr.03 md500": dict(num_leaves=127, learning_rate=0.03, min_data_in_leaf=500),
    "leaves31 lr.03 md1000": dict(num_leaves=31, learning_rate=0.03, min_data_in_leaf=1000),
    "leaves63 lr.03 md500 ff.6": dict(num_leaves=63, learning_rate=0.03, min_data_in_leaf=500, feature_fraction=0.6),
    "leaves255 lr.02 md1000": dict(num_leaves=255, learning_rate=0.02, min_data_in_leaf=1000),
    "leaves63 lr.03 md2000": dict(num_leaves=63, learning_rate=0.03, min_data_in_leaf=2000),
}

for name, over in CONFIGS.items():
    p = {**BASE, **over}
    t0 = time.time()
    bst = lgb.train(p, dtrain, num_boost_round=2500, valid_sets=[dval], feval=feval,
                    callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    pred = bst.predict(Xall[va], num_iteration=bst.best_iteration)
    ts = ts_auc_grouped(tva, yva_i, pred)
    print(f"{name:30s} best_iter={bst.best_iteration:4d} TS-AUC={ts:.4f} ({time.time()-t0:.0f}s)", flush=True)
