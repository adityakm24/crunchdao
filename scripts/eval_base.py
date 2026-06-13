"""Reproduce the round-3 final (model_018) VAL TS-AUC from the existing
val-split boosters + logistic, WITHOUT retraining, and cache the per-member VAL
logits so new members (e.g. a neural sequence member) can be blended and
evaluated quickly on the *same* fixed split.

Everything is derived from the single v4 matrix (v2 is a nested column subset),
so we don't need separate v2/v4 npz files.

Outputs features/val_base_logits.npz with:
  y, t_online, series_id           (VAL rows, split seed 42)
  gbt_logits  (n_val, 4)           per-GBT-member logit
  log_logit   (n_val,)             logistic-member logit
  base_logit  (n_val,)             0.8*mean(gbt) + 0.2*logistic  (the shipped blend)

Run: uv run python scripts/eval_base.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np

from sb.metric import ts_auc_grouped

V4 = "features/train_features_v4.npz"
SPLIT_SEED = 42
W_LIN = 0.2
GBT_MEMBERS = ["model_003", "model_008", "model_009", "model_015"]
LOGI = "artifacts/models/model_017_logistic/logistic.npz"
V2_DROP = {"t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
           "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2"}


def val_mask(sid: np.ndarray) -> np.ndarray:
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main() -> None:
    d = np.load(V4, allow_pickle=True)
    X, y, sid, t = d["X"], d["y"], d["series_id"], d["t_online"]
    names = [str(n) for n in d["feature_names"]]
    name_to_col = {n: i for i, n in enumerate(names)}

    va = val_mask(sid)
    Xv = X[va]
    yv = y[va].astype(np.int64)
    tv = t[va]
    sv = sid[va]
    print(f"VAL rows={va.sum():,}  series={len(np.unique(sv))}")

    # GBT members on their own feature-name subsets
    gbt_logits = []
    for m in GBT_MEMBERS:
        b = lgb.Booster(model_file=f"artifacts/models/{m}/lgbm_valsplit.txt")
        cols = [name_to_col[n] for n in b.feature_name()]
        p = b.predict(Xv[:, cols])
        gl = _logit(p)
        gbt_logits.append(gl)
        print(f"  {m}: {ts_auc_grouped(tv, yv, p):.4f}")
    gbt_logits = np.vstack(gbt_logits).T  # (n_val, 4)
    gbt_mean = gbt_logits.mean(axis=1)
    print(f"  GBT mean-logit blend: {ts_auc_grouped(tv, yv, gbt_mean):.4f}")

    # logistic member on the v2 subset
    lg = np.load(LOGI, allow_pickle=True)
    lkeep = [name_to_col[n] for n in [str(n) for n in lg["feature_names"]]]
    Xl = np.nan_to_num(Xv[:, lkeep], nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip((Xl - lg["mean"]) / lg["scale"], -8, 8)
    log_logit = Xs @ lg["coef"] + lg["intercept"][0]
    print(f"  logistic: {ts_auc_grouped(tv, yv, log_logit):.4f}")

    base_logit = (1 - W_LIN) * gbt_mean + W_LIN * log_logit
    base = ts_auc_grouped(tv, yv, base_logit)
    print(f"\n>>> BASE ensemble (shipped model_018) VAL TS-AUC = {base:.4f}")

    os.makedirs("features", exist_ok=True)
    np.savez("features/val_base_logits.npz",
             y=yv, t_online=tv, series_id=sv,
             gbt_logits=gbt_logits, log_logit=log_logit, base_logit=base_logit)
    print("cached -> features/val_base_logits.npz")


if __name__ == "__main__":
    main()
