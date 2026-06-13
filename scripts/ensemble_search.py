"""Pick the final mean-logit ensemble from a model pool, principled (not
exhaustive cherry-picking, which would overfit VAL).

Loads each model's VAL prediction (aligned; split seed fixed at 42), reports
single scores and pairwise correlation, then evaluates a handful of *named*
candidate blends defined by feature-set/seed diversity. Prints the best.

Run: uv run python scripts/ensemble_search.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
from scipy.special import logit

from sb.metric import ts_auc_grouped

SPLIT_SEED = 42
FEAT = {
    "v1": "features/train_features.npz",
    "v2": "features/train_features_v2.npz",
    "v3": "features/train_features_v3.npz",
    "v4": "features/train_features_v4.npz",
}

# pool: model_id -> feature version
POOL = {
    "model_003": "v2", "model_008": "v2", "model_009": "v2",
    "model_010": "v3", "model_011": "v3", "model_012": "v3",
    "model_013": "v2", "model_014": "v3",
    "model_015": "v4", "model_016": "v4",
}

CANDIDATES = {
    "v2x3": ["model_003", "model_008", "model_009"],
    "v3x3": ["model_010", "model_011", "model_012"],
    "v2+v3": ["model_003", "model_009", "model_010", "model_012"],
    "v2+v3+v4": ["model_003", "model_009", "model_010", "model_012",
                 "model_015", "model_016"],
    "all_main": ["model_003", "model_008", "model_009", "model_010",
                 "model_011", "model_012", "model_015", "model_016"],
    "all_main+shallow": ["model_003", "model_008", "model_009", "model_010",
                         "model_011", "model_012", "model_013", "model_014",
                         "model_015", "model_016"],
    "best_per_set": ["model_009", "model_010", "model_015", "model_016"],
}


def val_arrays(fv, cache):
    if fv not in cache:
        d = np.load(FEAT[fv], allow_pickle=True)
        sid = d["series_id"]
        uniq = np.unique(sid)
        rng = np.random.default_rng(SPLIT_SEED)
        rng.shuffle(uniq)
        val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
        va = np.isin(sid, list(val_ids))
        names = [str(n) for n in d["feature_names"]]
        cache[fv] = (d["X"][va], names, d["y"][va].astype(np.int64),
                     d["t_online"][va])
    return cache[fv]


def main() -> None:
    import os
    cache = {}
    L = {}
    yv = tv = None
    for mid, fv in POOL.items():
        path = f"artifacts/models/{mid}/lgbm_valsplit.txt"
        if not os.path.exists(path):
            continue
        X, names, y, t = val_arrays(fv, cache)
        if yv is None:
            yv, tv = y, t
        b = lgb.Booster(model_file=path)
        ki = [names.index(n) for n in b.feature_name()]
        p = np.clip(b.predict(X[:, ki]), 1e-7, 1 - 1e-7)
        L[mid] = logit(p)
        print(f"  {mid:12s} ({fv}): {ts_auc_grouped(t, y, p):.4f}")

    avail = list(L.keys())
    print(f"\nPairwise logit correlation ({len(avail)} models):")
    A = np.vstack([L[m] for m in avail])
    C = np.corrcoef(A)
    print("        " + " ".join(f"{m.split('_')[1]}" for m in avail))
    for i, m in enumerate(avail):
        print(f"  {m.split('_')[1]:>4s}: " + " ".join(f"{C[i,j]:.2f}" for j in range(len(avail))))

    print("\nCandidate blends (mean-logit):")
    best = (None, 0.0)
    for name, members in CANDIDATES.items():
        ms = [m for m in members if m in L]
        if len(ms) < 2:
            continue
        blend = np.vstack([L[m] for m in ms]).mean(0)
        sc = ts_auc_grouped(tv, yv, blend)
        flag = ""
        if sc > best[1]:
            best = (name, sc)
            flag = " *"
        print(f"  {name:20s} ({len(ms)} models): {sc:.4f}{flag}")
    print(f"\nBEST: {best[0]} = {best[1]:.4f}")


if __name__ == "__main__":
    main()
