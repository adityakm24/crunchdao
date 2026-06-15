"""Round 9b — confirm the log_t-drop gain survives at the BLEND level (OOS).

Single-GBT OOS jumped +0.0125 by dropping log_t. The shipped model blends 4 GBTs
+ logistic + 3 GRUs (+ rank). This isolates the dominant lever: retrain ONLY the 4
base GBTs WITHOUT log_t on the full 10k train (production ramp weights reconstructed
from the cache), keep the logistic + GRUs exactly as shipped, and compare the
round-5 base blend OOS (rank excluded here; gw=0) with log_t IN vs OUT.

If the blend lifts even with logistic/GRU still carrying log_t, the full fix
(drop log_t everywhere + retrain rank/GRUs) will lift at least as much.

Run: uv run python scripts/reduced_nolog.py
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "submission")

import numpy as np
import lightgbm as lgb
import pandas as pd

from sb.data import iter_series, load_test_targets
from sb.metric import ts_auc_grouped
import main as sub

CACHE = "features/train_features_v4.npz"
MODEL_DIR = "submission/_model"
RAMP = 50.0


def _logit(p):
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def reconstruct_weights(sid, t, y):
    """Rebuild the production ramp weights: pre-break w=1, post-break ramp."""
    df = pd.DataFrame({"sid": sid, "t": t.astype(np.int64), "y": y})
    tau_row = df["t"].where(df["y"] == 1).groupby(df["sid"]).transform("min").to_numpy()
    w = np.ones(len(y), dtype=np.float64)
    pm = y == 1
    w[pm] = np.clip((t[pm] - tau_row[pm]) / RAMP, 0.2, 1.0)
    return w


def keep_without(keep, drop_name):
    col = sub._NAME_TO_COL[drop_name]
    return [i for i in keep if i != col]


def train_gbts(X, y, w, drop_logt):
    boosters, keeps = [], []
    for mi, (seed, rounds, version) in enumerate(sub.ENSEMBLE):
        keep = sub._keep_for(version)
        if drop_logt:
            keep = keep_without(keep, "log_t")
        names = [sub.FEATURE_NAMES[i] for i in keep]
        dtrain = lgb.Dataset(X[:, keep], label=y, weight=w, feature_name=names)
        params = dict(sub.LGB_PARAMS, seed=seed)
        boosters.append(lgb.train(params, dtrain, num_boost_round=rounds))
        keeps.append(np.asarray(keep, dtype=np.int64))
    return boosters, keeps


def main():
    d = np.load(CACHE, allow_pickle=True)
    X = np.ascontiguousarray(d["X"])
    y = d["y"].astype(np.int32)
    sid = d["series_id"]
    t = d["t_online"]
    assert [str(n) for n in d["feature_names"]] == list(sub.FEATURE_NAMES), \
        "cache feature order must match deployed FEATURE_NAMES"
    w = reconstruct_weights(sid, t, y)
    print(f"train rows={len(y):,}  pos_rate={y.mean():.4f}  "
          f"mean w(pos)={w[y == 1].mean():.3f}")

    print("training 4 GBTs WITH log_t ...")
    t0 = time.time()
    bts_with, keeps_with = train_gbts(X, y, w, drop_logt=False)
    print(f"  done ({time.time()-t0:.0f}s)")
    print("training 4 GBTs WITHOUT log_t ...")
    t0 = time.time()
    bts_no, keeps_no = train_gbts(X, y, w, drop_logt=True)
    print(f"  done ({time.time()-t0:.0f}s)")

    # logistic exactly as shipped (still uses log_t — held fixed)
    lg = np.load(os.path.join(MODEL_DIR, "logistic.npz"))
    l_keep, l_mean, l_scale = lg["keep"], lg["mean"], lg["scale"]
    l_coef, l_int = lg["coef"], float(lg["intercept"][0])

    grus = sub._load_grus(sub._NAME_TO_COL)
    calib_grus = [g for g in grus if getattr(g, "is_raw", False) is False]
    raw_grus = [g for g in grus if getattr(g, "is_raw", False)]
    w_seq = 1.0 / len(grus) if grus else 0.0
    w_gbt = (1.0 - sub.W_LIN) / len(bts_with)

    det = sub.StreamingDetector()
    yt = load_test_targets().reset_index().sort_values(["id", "time"])
    yt["k"] = yt.groupby("id").cumcount()
    key = yt.set_index(["id", "k"])["target"]

    m4_with, m4_no, lin_l, seq_l, ys, ts = [], [], [], [], [], []
    print("\nstreaming reduced OOS ...")
    t0 = time.time()
    for s in iter_series("test"):
        det.calibrate(np.asarray(s.x_hist, dtype=np.float64))
        for g in grus:
            g.reset()
        mu_h, sd_h = det.mu_h, det.sd_h
        for i, x in enumerate(s.x_online):
            feats = det.update(float(x))
            aw = sum(_logit(float(b.predict(feats[k].reshape(1, -1))[0]))
                     for b, k in zip(bts_with, keeps_with)) / len(bts_with)
            an = sum(_logit(float(b.predict(feats[k].reshape(1, -1))[0]))
                     for b, k in zip(bts_no, keeps_no)) / len(bts_no)
            xs = np.clip((feats[l_keep] - l_mean) / l_scale, -8.0, 8.0)
            lin = float(np.dot(xs, l_coef)) + l_int
            if grus:
                seq = sum(g.step(feats) for g in calib_grus)
                if raw_grus:
                    rv = sub._raw_vec(float(x), mu_h, sd_h)
                    seq += sum(g.step(rv) for g in raw_grus)
                seq *= w_seq
            else:
                seq = 0.0
            m4_with.append(aw)
            m4_no.append(an)
            lin_l.append(lin)
            seq_l.append(seq)
            ys.append(int(key.loc[(s.series_id, i)]))
            ts.append(i)
    m4_with = np.asarray(m4_with); m4_no = np.asarray(m4_no)
    lin_l = np.asarray(lin_l); seq_l = np.asarray(seq_l)
    ys = np.asarray(ys, dtype=np.int64); ts = np.asarray(ts, dtype=np.int64)
    print(f"  OOS rows={len(ys):,}  pos_rate={ys.mean():.4f}  ({time.time()-t0:.0f}s)\n")

    def blend(m4):
        base = (1.0 - sub.W_LIN) * m4 + sub.W_LIN * lin_l
        return (1.0 - sub.W_GRU) * base + sub.W_GRU * seq_l

    a_with = ts_auc_grouped(ts, ys, blend(m4_with))
    a_no = ts_auc_grouped(ts, ys, blend(m4_no))
    g_with = ts_auc_grouped(ts, ys, m4_with)
    g_no = ts_auc_grouped(ts, ys, m4_no)
    print("                 mean4(GBT)   full blend (rank excl, gw=0)")
    print(f"  WITH log_t      {g_with:.4f}        {a_with:.4f}")
    print(f"  WITHOUT log_t   {g_no:.4f}        {a_no:.4f}")
    print(f"  delta           {g_no-g_with:+.4f}        {a_no-a_with:+.4f}")
    if a_no > a_with + 1e-4:
        print("\n==> BLEND CONFIRMS: drop log_t productionwide (GBTs+logistic+rank+GRUs).")
    else:
        print("\n==> blend gain washed out by logistic/GRU still carrying log_t; "
              "must retrain those too to capture it.")


if __name__ == "__main__":
    main()
