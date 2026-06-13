"""Evaluate arbitrary model ensembles on the fixed VAL split (split seed 42).

Each model is (model_id, feature_npz). Predicts each booster on its own
feature matrix's VAL rows, then blends by mean-logit (and reports rank-mean).

Run: uv run python scripts/ensemble_eval.py m003:v2 m009:v2 m010:v3 ...
  where vN is shorthand for features/train_features_vN.npz (v1 -> train_features.npz).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from sb.metric import ts_auc_grouped

SPLIT_SEED = 42
FEAT = {
    "v1": "features/train_features.npz",
    "v2": "features/train_features_v2.npz",
    "v3": "features/train_features_v3.npz",
}


def val_mask_for(sid):
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids))


def main() -> None:
    specs = []
    for a in sys.argv[1:]:
        mid, fv = a.split(":")
        if not mid.startswith("model_"):
            mid = "model_" + mid.replace("m", "").zfill(3)
        specs.append((mid, fv))

    cache = {}
    preds = []
    yv = tv = None
    for mid, fv in specs:
        path = FEAT[fv]
        if fv not in cache:
            d = np.load(path, allow_pickle=True)
            names = [str(n) for n in d["feature_names"]]
            va = val_mask_for(d["series_id"])
            cache[fv] = (d["X"][va], names, d["y"][va].astype(np.int64),
                         d["t_online"][va])
        X, names, y, t = cache[fv]
        if yv is None:
            yv, tv = y, t
        b = lgb.Booster(model_file=f"artifacts/models/{mid}/lgbm_valsplit.txt")
        ki = [names.index(n) for n in b.feature_name()]
        p = b.predict(X[:, ki])
        preds.append(p)
        print(f"  {mid:12s} ({fv}): {ts_auc_grouped(t, y, p):.4f}")

    P = np.clip(np.vstack(preds), 1e-7, 1 - 1e-7)
    logit = np.log(P / (1 - P)).mean(0)
    rank = np.vstack([rankdata(p) for p in preds]).mean(0)
    print(f"\n  ENSEMBLE mean-logit : {ts_auc_grouped(tv, yv, logit):.4f}")
    print(f"  ENSEMBLE rank-mean  : {ts_auc_grouped(tv, yv, rank):.4f}")
    print(f"  ENSEMBLE mean-prob  : {ts_auc_grouped(tv, yv, P.mean(0)):.4f}")


if __name__ == "__main__":
    main()
