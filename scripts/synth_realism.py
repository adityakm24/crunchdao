"""Round 11 — synthetic realism diagnostic.

Two objective measures of how well synth matches real:
  1. Per-feature standardized gap |mean_s - mean_r| / std_r and std ratio.
  2. A LightGBM real-vs-synth classifier (split by SERIES, no row leakage). If it
     separates trivially (AUC ~1.0) the synth is unrealistic; the top-gain features
     are exactly what to fix. Target: AUC well below ~0.85 before trusting augmentation.

Run: uv run python scripts/synth_realism.py [--synth features/synth_features_v4.npz]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="features/synth_features_v4.npz")
    ap.add_argument("--nreal", type=int, default=200000)
    args = ap.parse_args()

    import lightgbm as lgb

    r = np.load("features/train_features_v4.npz", allow_pickle=True)
    names = [str(x) for x in r["feature_names"]]
    s = np.load(args.synth, allow_pickle=True)

    rng = np.random.default_rng(0)
    ri = rng.choice(len(r["X"]), min(args.nreal, len(r["X"])), replace=False)
    si = rng.choice(len(s["X"]), min(args.nreal, len(s["X"])), replace=False)
    Xr, Xs = r["X"][ri].astype(np.float32), s["X"][si].astype(np.float32)
    sid_r, sid_s = r["series_id"][ri], s["series_id"][si]

    # ---- per-feature gap ----
    mr, sr = Xr.mean(0), Xr.std(0) + 1e-9
    ms, ss = Xs.mean(0), Xs.std(0) + 1e-9
    gap = np.abs(ms - mr) / sr
    ratio = ss / sr
    order = np.argsort(-gap)
    print("=== worst per-feature gaps (|dmean|/std_real, std_ratio synth/real) ===")
    for i in order[:22]:
        print(f"  {names[i]:20s}  gap={gap[i]:6.2f}  std_ratio={ratio[i]:6.2f}  "
              f"(real {mr[i]:+.3f}/{sr[i]:.3f}  synth {ms[i]:+.3f}/{ss[i]:.3f})")

    # ---- classifier (split by series) ----
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xs))])
    sid = np.concatenate([sid_r.astype(np.int64), sid_s.astype(np.int64)])
    uniq = np.unique(sid); rng.shuffle(uniq)
    test_ids = set(uniq[: len(uniq) // 5].tolist())
    te = np.array([x in test_ids for x in sid])
    tr = ~te
    dtr = lgb.Dataset(X[tr], label=y[tr])
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, min_data_in_leaf=200, verbose=-1, seed=0)
    bst = lgb.train(params, dtr, num_boost_round=200)
    pred = bst.predict(X[te])
    # AUC
    pos = pred[y[te] == 1]; neg = pred[y[te] == 0]
    o = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    rk = np.empty(len(o)); rk[o] = np.arange(1, len(o) + 1)
    auc = (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    print(f"\n=== real-vs-synth classifier AUC = {auc:.4f} "
          f"(0.5=indistinguishable, 1.0=trivially separable) ===")
    gains = bst.feature_importance(importance_type="gain")
    go = np.argsort(-gains)
    print("top discriminative features (fix these):")
    for i in go[:15]:
        print(f"  {names[i]:20s}  gain={gains[i]:10.0f}")


if __name__ == "__main__":
    main()
