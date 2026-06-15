"""Round 9b — does the model overfit to absolute POSITION/LENGTH features?

The stepnorm probe found `log_t` is the #1 feature, yet normalising it gave a big
VAL gain that REGRESSED OOS -> the model leans on log_t / t_over_nhist as a
cross-sectional ranking signal. At a fixed online-step these vary ONLY with the
series' history length, so they encode the TRAIN population's (length <-> break)
correlation, which need not transfer. Both were re-added back in round 1 (a VAL
jump) BEFORE the OOS reduced harness existed -> never OOS-validated.

Hypothesis: dropping {log_t, t_over_nhist} HURTS VAL but HELPS (or is flat on) OOS.
If OOS improves, that is a real generalisation gain and explains the VAL/OOS gap;
retrain the base GBTs without them.

Trains one GBT per drop-set on the SAME 80% split, compares full VAL + honest
halves (rng1) + streamed reduced OOS (ids>=10000).

Run: uv run python scripts/reduced_droppos.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "submission")

import numpy as np
import lightgbm as lgb

from sb.metric import ts_auc_grouped

FEATURES = "features/train_features_v4.npz"
SPLIT_SEED = 42
BASE_DROP = {
    "t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
    "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2",
}
DROP_SETS = {
    "baseline": set(),
    "drop log_t": {"log_t"},
    "drop log_t+t_over_nhist": {"log_t", "t_over_nhist"},
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


def fit(Xtr, ytr, Xva, yva, tva, names):
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=list(names))
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)

    def feval(p, _):
        return ("ts_auc", ts_auc_grouped(tva, yva, p), True)

    bst = lgb.train(PARAMS, dtr, num_boost_round=NUM_ROUNDS,
                    valid_sets=[dva], valid_names=["val"], feval=feval,
                    callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True),
                               lgb.log_evaluation(0)])
    return bst


def halves(sid_val):
    uniq = np.unique(sid_val)
    np.random.default_rng(1).shuffle(uniq)
    a = set(uniq[: len(uniq) // 2].tolist())
    mA = np.isin(sid_val, list(a))
    return mA, ~mA


def stream_oos(keep_idx):
    import main as sub
    from sb.data import iter_series, load_test_targets

    det = sub.StreamingDetector()
    yt = load_test_targets().reset_index().sort_values(["id", "time"])
    yt["k"] = yt.groupby("id").cumcount()
    key = yt.set_index(["id", "k"])["target"]
    rows, ys, ts = [], [], []
    for s in iter_series("test"):
        det.calibrate(np.asarray(s.x_hist, dtype=np.float64))
        for i, x in enumerate(s.x_online):
            feats = det.update(float(x))
            rows.append(feats[keep_idx])
            ys.append(int(key.loc[(s.series_id, i)]))
            ts.append(i)
    return (np.asarray(rows, dtype=np.float32),
            np.asarray(ys, dtype=np.int64), np.asarray(ts, dtype=np.int64))


def main():
    d = np.load(FEATURES, allow_pickle=True)
    names_all = [str(n) for n in d["feature_names"]]
    X_all = d["X"]
    y = d["y"].astype(np.int32)
    sid = d["series_id"]
    t = d["t_online"].astype(np.int64)

    uniq = np.unique(sid)
    np.random.default_rng(SPLIT_SEED).shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    va = np.isin(sid, list(val_ids))
    tr = ~va
    tva = t[va]
    yv = y[va].astype(np.int64)
    sidv = sid[va]
    mA, mB = halves(sidv)
    print(f"train rows={tr.sum():,}  val rows={va.sum():,}")

    # stream OOS once with FULL feature vector, then slice per drop-set
    print("streaming reduced OOS (full feats)...")
    t0 = time.time()
    full_keep = np.array([i for i, n in enumerate(names_all) if n not in BASE_DROP])
    full_names = [names_all[i] for i in full_keep]
    name_to_pos = {n: j for j, n in enumerate(full_names)}
    Xo_full, yo, to = stream_oos(full_keep)
    print(f"OOS rows={len(yo):,}  pos_rate={yo.mean():.4f}  ({time.time()-t0:.0f}s)\n")

    print(f"{'drop-set':>26}{'VALfull':>9}{'halfA':>8}{'halfB':>8}{'OOS':>9}{'iter':>6}")
    base_oos = None
    for tag, extra in DROP_SETS.items():
        drop = BASE_DROP | extra
        keep = np.array([i for i, n in enumerate(names_all) if n not in drop])
        names = [names_all[i] for i in keep]
        Xk = np.ascontiguousarray(X_all[:, keep])
        bst = fit(Xk[tr], y[tr], Xk[va], yv, tva, names)
        pv = bst.predict(Xk[va], num_iteration=bst.best_iteration)
        full = ts_auc_grouped(tva, yv, pv)
        aA = ts_auc_grouped(tva[mA], yv[mA], pv[mA])
        aB = ts_auc_grouped(tva[mB], yv[mB], pv[mB])
        # OOS: slice the streamed full matrix to this drop-set's columns
        cols = [name_to_pos[n] for n in names]
        oos = ts_auc_grouped(to, yo, bst.predict(Xo_full[:, cols],
                                                 num_iteration=bst.best_iteration))
        if base_oos is None:
            base_oos = oos
        d_oos = oos - base_oos
        print(f"{tag:>26}{full:>9.4f}{aA:>8.4f}{aB:>8.4f}{oos:>9.4f}"
              f"{bst.best_iteration:>6}  ({d_oos:+.4f} OOS)")


if __name__ == "__main__":
    main()
