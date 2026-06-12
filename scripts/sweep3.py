"""Probe regularization + sample/class weighting on winning feature set."""
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


def feval(preds, _ds):
    return ("ts_auc", ts_auc_grouped(tva, yva_i, preds), True)


BASE = dict(objective="binary", metric="None", boosting_type="gbdt",
            num_leaves=31, learning_rate=0.03, min_data_in_leaf=1000,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            num_threads=0, seed=SEED, deterministic=True, force_row_wise=True,
            verbosity=-1)

# down-weight near-undetectable fresh-break rows (elapsed small): weight ramps
# from 0.3 at break to 1.0 over `ramp` steps; negatives weight 1.
def make_weights(ramp):
    w = np.ones(len(y), dtype=np.float64)
    # elapsed since break is not directly stored; approximate via consecutive
    # positive run length within each series using t_online resets.
    # Simpler: weight positives by min(1, (t_online - tau_local)/ramp) needs tau.
    return w  # placeholder; see weighted runs below


def run(name, over, weights=None, spw=None):
    p = {**BASE, **over}
    if spw is not None:
        p["scale_pos_weight"] = spw
    dtrain = lgb.Dataset(Xall[tr], label=y[tr],
                         weight=None if weights is None else weights[tr])
    dval = lgb.Dataset(Xall[va], label=y[va], reference=dtrain)
    t0 = time.time()
    bst = lgb.train(p, dtrain, num_boost_round=3000, valid_sets=[dval], feval=feval,
                    callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    pred = bst.predict(Xall[va], num_iteration=bst.best_iteration)
    ts = ts_auc_grouped(tva, yva_i, pred)
    print(f"{name:34s} best_iter={bst.best_iteration:4d} TS-AUC={ts:.4f} ({time.time()-t0:.0f}s)", flush=True)


# elapsed-since-break weights: reconstruct per-row elapsed from positive runs.
def elapsed_weights(ramp):
    w = np.ones(len(y), dtype=np.float64)
    pos = y == 1
    # within each series, positives are a contiguous tail; elapsed = position in run.
    # Use t_online and the first positive t per series.
    order = np.lexsort((t_online, sid))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    s_sorted = sid[order]
    y_sorted = y[order]
    t_sorted = t_online[order]
    w_sorted = np.ones(len(order))
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        yy = y_sorted[i:j]
        tt = t_sorted[i:j]
        if yy.any():
            tau = tt[yy.argmax()]
            elapsed = tt - tau
            ww = np.ones(j - i)
            mask = yy == 1
            ww[mask] = np.clip(elapsed[mask] / ramp, 0.2, 1.0)
            w_sorted[i:j] = ww
        i = j
    w = w_sorted[inv]
    return w


run("baseline leaves31 lr.03 md1000", {})
run("leaves15 lr.02 md2000", dict(num_leaves=15, learning_rate=0.02, min_data_in_leaf=2000))
run("leaves31 lr.02 md1000 ff.5", dict(learning_rate=0.02, feature_fraction=0.5))
run("+scale_pos_weight=2", {}, spw=2.0)
run("+elapsed_weights ramp=20", {}, weights=elapsed_weights(20))
run("+elapsed_weights ramp=50", {}, weights=elapsed_weights(50))
