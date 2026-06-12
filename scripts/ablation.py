"""Feature-group ablation with TS-AUC-based early stopping.

Loads the 57-column matrix once (which still contains the time columns) and
trains LightGBM on different feature subsets, selecting iterations by TS-AUC.

Run:  uv run python scripts/ablation.py
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
NAME2IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
TIME = ["t", "log_t", "log_n_hist", "t_over_nhist"]
# the EXP-007-only additions (multi window + multi-k cusum + diff)
EXTRA = [n for n in FEATURE_NAMES if (
    n.startswith(("w25_", "w100_", "w200_", "cusum_absmax_k"))
    or n in ("dvar_ratio_cum", "dvar_ratio_win", "dabs_ratio_cum"))]

d = np.load("features/train_features.npz", allow_pickle=True)
Xall, y, sid, t_online = d["X"], d["y"], d["series_id"], d["t_online"]

uniq = np.unique(sid)
rng = np.random.default_rng(SEED)
rng.shuffle(uniq)
n_val = int(0.2 * len(uniq))
va = np.isin(sid, list(set(uniq[:n_val].tolist())))
tr = ~va
tva, yva_i = t_online[va], y[va].astype(np.int64)

BASE = dict(objective="binary", metric="None", boosting_type="gbdt",
            num_leaves=63, learning_rate=0.05, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=200,
            num_threads=0, seed=SEED, deterministic=True, force_row_wise=True,
            verbosity=-1)


def run(name, drop):
    keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in set(drop)]
    Xtr, Xva = Xall[tr][:, keep], Xall[va][:, keep]
    dtrain = lgb.Dataset(Xtr, label=y[tr])
    dval = lgb.Dataset(Xva, label=y[va], reference=dtrain)

    def feval(preds, _ds):
        return ("ts_auc", ts_auc_grouped(tva, yva_i, preds), True)

    t0 = time.time()
    bst = lgb.train(BASE, dtrain, num_boost_round=1500, valid_sets=[dval],
                    feval=feval,
                    callbacks=[lgb.early_stopping(120, first_metric_only=True, verbose=False)])
    pred = bst.predict(Xva, num_iteration=bst.best_iteration)
    ts = ts_auc_grouped(tva, yva_i, pred)
    print(f"{name:42s} nfeat={len(keep):3d} best_iter={bst.best_iteration:4d} "
          f"TS-AUC={ts:.4f} ({time.time()-t0:.0f}s)", flush=True)


run("content-only (drop time) [EXP-007]", TIME)
run("content + log_t,t_over_nhist", ["t", "log_n_hist"])
run("content + ALL time", [])
run("EXP-004-like (drop time + EXP007 extras)", TIME + EXTRA)
run("drop time + drop multi-window keep w50/w150-like", TIME + [n for n in FEATURE_NAMES if n.startswith(("w25_", "w100_"))])
