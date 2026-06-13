"""Lock in the final ensemble: best GBT subset (+ optional logistic member),
evaluated on the fixed VAL split, with an honest 2-fold-on-VAL check so the
chosen blend isn't an in-sample fluke.

Trains the logistic deterministically on the v2 train split (same fixed split
as the GBTs) and saves its scaler mean/scale + coef/intercept to
artifacts/models/model_017_logistic/ for the submission to reproduce.

Run: uv run python scripts/final_blend.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sb.metric import ts_auc_grouped

SPLIT = 42
FEAT = {"v2": "features/train_features_v2.npz",
        "v3": "features/train_features_v3.npz",
        "v4": "features/train_features_v4.npz"}
GBT = {"003": "v2", "008": "v2", "009": "v2", "015": "v4"}  # chosen blend
V2_DROP = {"t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
           "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2"}

cache = {}


def arrays(fv):
    if fv not in cache:
        d = np.load(FEAT[fv], allow_pickle=True)
        cache[fv] = (d["X"], [str(n) for n in d["feature_names"]],
                     d["y"], d["series_id"], d["t_online"])
    return cache[fv]


def split(sid):
    u = np.unique(sid)
    rng = np.random.default_rng(SPLIT)
    rng.shuffle(u)
    vi = set(u[: int(0.2 * len(u))].tolist())
    va = np.isin(sid, list(vi))
    return ~va, va


def main() -> None:
    # GBT val logits
    L, yv, tv, sv = {}, None, None, None
    for m, fv in GBT.items():
        X, names, y, sid, t = arrays(fv)
        _, va = split(sid)
        if yv is None:
            yv, tv, sv = y[va].astype(np.int64), t[va], sid[va]
        b = lgb.Booster(model_file=f"artifacts/models/model_{m}/lgbm_valsplit.txt")
        ki = [names.index(n) for n in b.feature_name()]
        L[m] = logit(np.clip(b.predict(X[va][:, ki]), 1e-7, 1 - 1e-7))
        print(f"  GBT model_{m} ({fv}): {ts_auc_grouped(tv, yv, expit(L[m])):.4f}")

    # logistic on v2 features (deterministic), saved for the submission
    X, names, y, sid, t = arrays("v2")
    keep = [i for i, n in enumerate(names) if n not in V2_DROP]
    keep_names = [names[i] for i in keep]
    tr, va = split(sid)
    Xtr = np.nan_to_num(X[tr][:, keep], nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(X[va][:, keep], nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = np.clip(scaler.transform(Xtr), -8, 8)
    Xva_s = np.clip(scaler.transform(Xva), -8, 8)
    lr = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
    lr.fit(Xtr_s, y[tr])
    lin_logit_va = (Xva_s @ lr.coef_[0]) + lr.intercept_[0]
    print(f"  logistic (v2): {ts_auc_grouped(tv, yv, expit(lin_logit_va)):.4f}")

    G = np.vstack([L[m] for m in GBT]).mean(0)
    print(f"\n  GBT blend {list(GBT)}        : {ts_auc_grouped(tv, yv, G):.4f}")
    best = ("gbt", ts_auc_grouped(tv, yv, G), 0.0)
    for w in (0.1, 0.15, 0.2, 0.25, 0.3):
        blend = (1 - w) * G + w * lin_logit_va
        s = ts_auc_grouped(tv, yv, blend)
        if s > best[1]:
            best = (f"+{w}logit", s, w)
        print(f"  GBT + {w:.2f}*logistic        : {s:.4f}")

    # honest 2-fold-on-VAL for the chosen weight
    w = best[2]
    uniq = np.unique(sv)
    rng = np.random.default_rng(1)
    rng.shuffle(uniq)
    A = np.isin(sv, list(set(uniq[: len(uniq) // 2].tolist())))
    gw = (1 - w) * G + w * lin_logit_va
    print(f"\n  chosen: {best[0]} (w={w}) -> {best[1]:.4f}")
    print(f"  honest halves: A={ts_auc_grouped(tv[A], yv[A], gw[A]):.4f}  "
          f"B={ts_auc_grouped(tv[~A], yv[~A], gw[~A]):.4f}")

    outdir = "artifacts/models/model_017_logistic"
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "logistic.npz"),
             mean=scaler.mean_, scale=scaler.scale_,
             coef=lr.coef_[0], intercept=np.array([lr.intercept_[0]]),
             feature_names=np.array(keep_names))
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(dict(val_ts_auc=round(float(ts_auc_grouped(tv, yv, expit(lin_logit_va))), 5),
                       blend_weight=w, blend_val=round(float(best[1]), 5),
                       members=list(GBT)), fh, indent=2)
    print(f"  saved logistic params -> {outdir}")


if __name__ == "__main__":
    main()
