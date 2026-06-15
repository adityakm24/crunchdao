"""Round 9 — VAL honest-halves + OOS reduced gate for STEP-CONDITIONAL NULL
standardization of the v4 features.

eda_stepnorm.py showed a big VAL lift (raw 0.5928 -> stepnorm 0.6013, +0.0085).
VAL has historically overstated vs OOS, so this is the decisive gate: does a
single stepnorm GBT beat a single raw GBT on (a) BOTH honest VAL halves and
(b) the reduced OOS test (ids>=10000), the true generalization signal?

Transform (serve-compatible, leak-free): mu0(t)/sd0(t) = per-online-step
cross-sectional mean/std of each feature over TRAIN y==0 rows, smoothed over t;
x -> (x - mu0(t)) / sd0(t), clipped. Stats come from the 80% train portion only,
so VAL and OOS rows never enter their own normalisation -> exactly deployable as
a t-indexed lookup baked from train (O(1), no cross-series info at inference).

Run: uv run python scripts/reduced_stepnorm.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "submission")

import numpy as np
import lightgbm as lgb
from scipy.ndimage import uniform_filter1d

from sb.metric import ts_auc_grouped

FEATURES = "features/train_features_v4.npz"
SPLIT_SEED = 42
SMOOTH = 25
CLIP = 12.0
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


def step_null_stats(X, t, mask, T):
    """Per-step cross-sectional NULL mean/std over rows in `mask`, smoothed in t."""
    n_feat = X.shape[1]
    tn = t[mask]
    cnt = np.maximum(np.bincount(tn, minlength=T).astype(np.float64), 1.0)
    mu = np.empty((T, n_feat))
    sd = np.empty((T, n_feat))
    Xn = X[mask]
    for j in range(n_feat):
        col = Xn[:, j].astype(np.float64)
        m = np.bincount(tn, weights=col, minlength=T) / cnt
        v = np.maximum(np.bincount(tn, weights=col * col, minlength=T) / cnt - m * m, 0.0)
        mu[:, j] = m
        sd[:, j] = np.sqrt(v)
    mu = uniform_filter1d(mu, size=SMOOTH, axis=0, mode="nearest")
    sd = uniform_filter1d(sd, size=SMOOTH, axis=0, mode="nearest")
    sd[sd < 1e-6] = 1.0
    return mu, sd


def apply_norm(X, t, mu, sd):
    Xn = (X - mu[t]) / sd[t]
    np.clip(Xn, -CLIP, CLIP, out=Xn)
    return Xn.astype(np.float32)


def fit(Xtr, ytr, Xva, yva, tva, names, tag):
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=list(names))
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)

    def feval(p, _):
        return ("ts_auc", ts_auc_grouped(tva, yva, p), True)

    t0 = time.time()
    bst = lgb.train(PARAMS, dtr, num_boost_round=NUM_ROUNDS,
                    valid_sets=[dva], valid_names=["val"], feval=feval,
                    callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True),
                               lgb.log_evaluation(0)])
    print(f"  [{tag}] best_iter {bst.best_iteration}  ({time.time()-t0:.0f}s)")
    return bst


def halves(sid_val):
    uniq = np.unique(sid_val)
    np.random.default_rng(1).shuffle(uniq)
    a = set(uniq[: len(uniq) // 2].tolist())
    mA = np.isin(sid_val, list(a))
    return mA, ~mA


def stream_oos(keep_idx):
    """Stream the reduced OOS test through the production StreamingDetector."""
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
    keep_idx = np.array([i for i, n in enumerate(names_all) if n not in DEFAULT_DROP])
    names = [names_all[i] for i in keep_idx]
    X = np.ascontiguousarray(d["X"][:, keep_idx])
    y = d["y"].astype(np.int32)
    sid = d["series_id"]
    t = d["t_online"].astype(np.int64)
    T = int(t.max()) + 1
    print(f"X={X.shape}  feats={len(names)}  T={T}")

    uniq = np.unique(sid)
    np.random.default_rng(SPLIT_SEED).shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    va = np.isin(sid, list(val_ids))
    tr = ~va
    tva = t[va]
    yv = y[va].astype(np.int64)
    print(f"train rows={tr.sum():,}  val rows={va.sum():,}")

    # stats from the 80% TRAIN null rows only (deployable, leak-free)
    mu, sd = step_null_stats(X, t, tr & (y == 0), T)
    Xn = apply_norm(X, t, mu, sd)

    print("\n=== train raw + stepnorm GBT (80% split, VAL early-stop) ===")
    braw = fit(X[tr], y[tr], X[va], yv, tva, names, "raw")
    bsn = fit(Xn[tr], y[tr], Xn[va], yv, tva, names, "stepnorm")

    praw = braw.predict(X[va], num_iteration=braw.best_iteration)
    psn = bsn.predict(Xn[va], num_iteration=bsn.best_iteration)
    sidv = sid[va]
    mA, mB = halves(sidv)

    def auc(mask, p):
        return ts_auc_grouped(tva[mask], yv[mask], p[mask])

    print("\n=== VAL honest halves (rng1) ===")
    print(f"{'':>10}{'full':>9}{'halfA':>9}{'halfB':>9}")
    for tag, p in (("raw", praw), ("stepnorm", psn)):
        full = ts_auc_grouped(tva, yv, p)
        print(f"{tag:>10}{full:>9.4f}{auc(mA,p):>9.4f}{auc(mB,p):>9.4f}")
    dfull = ts_auc_grouped(tva, yv, psn) - ts_auc_grouped(tva, yv, praw)
    print(f"{'delta':>10}{dfull:>+9.4f}{auc(mA,psn)-auc(mA,praw):>+9.4f}"
          f"{auc(mB,psn)-auc(mB,praw):>+9.4f}")

    print("\n=== streaming reduced OOS test (ids>=10000) ===")
    t0 = time.time()
    Xo, yo, to = stream_oos(keep_idx)
    print(f"OOS rows={len(yo):,}  pos_rate={yo.mean():.4f}  ({time.time()-t0:.0f}s)")
    Xon = apply_norm(Xo, to, mu, sd)
    oraw = ts_auc_grouped(to, yo, braw.predict(Xo, num_iteration=braw.best_iteration))
    osn = ts_auc_grouped(to, yo, bsn.predict(Xon, num_iteration=bsn.best_iteration))
    print(f"\nOOS TS-AUC  raw {oraw:.4f} | stepnorm {osn:.4f} ({osn-oraw:+.4f})")

    if osn > oraw + 1e-4 and auc(mA, psn) > auc(mA, praw) and auc(mB, psn) > auc(mB, praw):
        print("\n==> PASS: stepnorm beats raw on BOTH VAL halves AND OOS. Productionize.")
        np.savez("features/stepnorm_stats.npz", mu=mu.astype(np.float32),
                 sd=sd.astype(np.float32), keep_idx=keep_idx,
                 names=np.array(names), smooth=SMOOTH, clip=CLIP)
        print("    saved features/stepnorm_stats.npz")
    else:
        print("\n==> FAIL the gate. Document as negative, keep raw features.")


if __name__ == "__main__":
    main()
