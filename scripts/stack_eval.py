"""Honest stacking/blend evaluator over base models on the fixed VAL split.

For each base model we predict on its own feature matrix's VAL rows (the VAL
series are identical across models — split seed is fixed at 42 — so the
per-row base predictions align). We then assess three blends with an honest
2-fold-on-VAL protocol (fit the blend on one half of the VAL *series*, score
the other half, both directions, weight by pair count):

  * equal     : mean-logit (no fitting; reference)
  * logreg    : logistic regression on base logits (meta weights)
  * gbm_stack : small LightGBM on base logits + a few raw gate features

This tells us whether a *learned* blend genuinely beats equal averaging out of
sample, before we commit to the extra inference complexity.

Run: uv run python scripts/stack_eval.py m003:v2 m008:v2 m009:v2 m010:v3 ...
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from sb.metric import ts_auc_grouped

SPLIT_SEED = 42
FEAT = {
    "v1": "features/train_features.npz",
    "v2": "features/train_features_v2.npz",
    "v3": "features/train_features_v3.npz",
    "v4": "features/train_features_v4.npz",
}
# raw gate features for the gbm stack (constant-ish per series, cheap signal)
GATE = ["log_t", "t_over_nhist", "acf1_h", "kurt_h", "null_mem_slope"]


def val_arrays(fv):
    d = np.load(FEAT[fv], allow_pickle=True)
    sid = d["series_id"]
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    va = np.isin(sid, list(val_ids))
    names = [str(n) for n in d["feature_names"]]
    return d["X"][va], names, d["y"][va].astype(np.int64), d["t_online"][va], sid[va]


def main() -> None:
    specs = []
    for a in sys.argv[1:]:
        mid, fv = a.split(":")
        if not mid.startswith("model_"):
            mid = "model_" + mid.replace("m", "").zfill(3)
        specs.append((mid, fv))

    cache = {}
    preds, gate = [], None
    yv = tv = sv = None
    for mid, fv in specs:
        if fv not in cache:
            cache[fv] = val_arrays(fv)
        X, names, y, t, s = cache[fv]
        if yv is None:
            yv, tv, sv = y, t, s
            gi = [names.index(g) for g in GATE]
            gate = X[:, gi]
        b = lgb.Booster(model_file=f"artifacts/models/{mid}/lgbm_valsplit.txt")
        ki = [names.index(n) for n in b.feature_name()]
        p = np.clip(b.predict(X[:, ki]), 1e-7, 1 - 1e-7)
        preds.append(p)
        print(f"  {mid:12s} ({fv}): {ts_auc_grouped(t, y, p):.4f}")

    L = np.vstack([logit(p) for p in preds]).T  # [N, M] base logits
    print(f"\n  equal mean-logit (all rows): {ts_auc_grouped(tv, yv, L.mean(1)):.4f}")

    # honest 2-fold over VAL *series*
    uniq = np.unique(sv)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    half = set(uniq[: len(uniq) // 2].tolist())
    foldA = np.isin(sv, list(half))
    out_equal = np.empty(len(yv))
    out_lr = np.empty(len(yv))
    out_gbm = np.empty(len(yv))
    for fit_mask, sc_mask in [(foldA, ~foldA), (~foldA, foldA)]:
        out_equal[sc_mask] = L[sc_mask].mean(1)
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(L[fit_mask], yv[fit_mask])
        out_lr[sc_mask] = lr.decision_function(L[sc_mask])
        # gbm stack on logits + gates
        Z = np.hstack([L, gate])
        dtr = lgb.Dataset(Z[fit_mask], label=yv[fit_mask])
        pr = dict(objective="binary", metric="None", num_leaves=15,
                  learning_rate=0.05, min_data_in_leaf=2000, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, seed=1, deterministic=True,
                  force_row_wise=True, verbosity=-1)
        bst = lgb.train(pr, dtr, num_boost_round=120)
        out_gbm[sc_mask] = bst.predict(Z[sc_mask])

    print("\n  Honest 2-fold-on-VAL out-of-sample TS-AUC:")
    print(f"    equal mean-logit : {ts_auc_grouped(tv, yv, out_equal):.4f}")
    print(f"    logreg stack     : {ts_auc_grouped(tv, yv, out_lr):.4f}")
    print(f"    gbm stack        : {ts_auc_grouped(tv, yv, out_gbm):.4f}")


if __name__ == "__main__":
    main()
