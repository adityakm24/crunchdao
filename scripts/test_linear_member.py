"""Does a linear model add ensemble diversity over the GBT pool?

Trains a logistic regression on the v2 features (train split, same fixed split
as the GBTs), evaluates its own VAL TS-AUC, and checks whether mean-logit
blending it with the GBT ensemble helps out of sample. Linear models have a
very different inductive bias from GBTs, so even a weaker linear member can
decorrelate errors and lift the blend.

Run: uv run python scripts/test_linear_member.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sb.metric import ts_auc_grouped

SPLIT_SEED = 42
GBT = {"model_003": "v2", "model_009": "v2", "model_010": "v3",
       "model_012": "v3", "model_015": "v4", "model_016": "v4"}
FEAT = {"v2": "features/train_features_v2.npz",
        "v3": "features/train_features_v3.npz",
        "v4": "features/train_features_v4.npz"}


def split(sid):
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    va = np.isin(sid, list(val_ids))
    return ~va, va


def main() -> None:
    import os
    d = np.load(FEAT["v2"], allow_pickle=True)
    X, y, sid, t = d["X"], d["y"], d["series_id"], d["t_online"]
    names = [str(n) for n in d["feature_names"]]
    drop = {"t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
            "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2"}
    keep = [i for i, n in enumerate(names) if n not in drop]
    tr, va = split(sid)

    Xtr = np.nan_to_num(X[tr][:, keep], nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(X[va][:, keep], nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -8, 8)
    Xva = np.clip(sc.transform(Xva), -8, 8)
    yv, tv = y[va].astype(np.int64), t[va]

    lr = LogisticRegression(C=0.5, max_iter=500, solver="lbfgs")
    lr.fit(Xtr, y[tr])
    p_lin = lr.predict_proba(Xva)[:, 1]
    print(f"  logistic (v2 feats) VAL TS-AUC: {ts_auc_grouped(tv, yv, p_lin):.4f}")

    # GBT pool logits on the same val rows
    cache = {"v2": (X, names)}
    pool = []
    for mid, fv in GBT.items():
        path = f"artifacts/models/{mid}/lgbm_valsplit.txt"
        if not os.path.exists(path):
            continue
        if fv not in cache:
            dd = np.load(FEAT[fv], allow_pickle=True)
            cache[fv] = (dd["X"], [str(n) for n in dd["feature_names"]])
        XX, nn = cache[fv]
        _, va2 = split(dd["series_id"]) if fv != "v2" else (tr, va)
        b = lgb.Booster(model_file=path)
        ki = [nn.index(n) for n in b.feature_name()]
        p = np.clip(b.predict(XX[va2][:, ki]), 1e-7, 1 - 1e-7)
        pool.append(logit(p))
        print(f"  {mid:12s} ({fv}): {ts_auc_grouped(tv, yv, p):.4f}")

    gbt = np.vstack(pool).mean(0)
    print(f"\n  GBT-only mean-logit         : {ts_auc_grouped(tv, yv, gbt):.4f}")
    llin = logit(np.clip(p_lin, 1e-7, 1 - 1e-7))
    for w in (0.1, 0.2, 0.3):
        blend = (1 - w) * gbt + w * llin
        print(f"  GBT + {w:.1f}*logistic         : {ts_auc_grouped(tv, yv, blend):.4f}")
    print(f"  corr(GBT, logistic) = {np.corrcoef(gbt, llin)[0,1]:.3f}")


if __name__ == "__main__":
    main()
