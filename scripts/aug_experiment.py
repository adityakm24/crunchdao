"""Round 11 — synthetic-augmentation A/B (local, no push).

Decisive falsification test. Train two GBTs that differ ONLY in training data:
  A) real-train rows (the 8000-series VAL-split TRAIN of the v4 matrix)
  B) real-train rows + synthetic rows
Identical params / seed / rounds. Evaluate both on the SAME real VAL (2000 series),
full + honest halves (seed1). VAL/halves stay 100% REAL — synth is train-only.

Gate: augmentation is real only if B beats A on VAL full AND both halves. Real public
tracks VAL-0.018, so a VAL gain that holds on both halves is the minimum bar.

Run: uv run python scripts/aug_experiment.py --synth features/synth_features_v4.npz \
        --rounds 800 [--synthfrac 1.0]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "src")
import lightgbm as lgb                       # noqa: E402
from sb.metric import ts_auc_grouped         # noqa: E402

V4 = "features/train_features_v4.npz"
SPLIT_SEED = 42

PARAMS = dict(
    objective="binary", metric="None", boosting_type="gbdt",
    num_leaves=31, learning_rate=0.03, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=1000,
    num_threads=0, seed=42, force_row_wise=True, verbosity=-1,
)


def val_mask(sid):
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids))


def half_mask(sid_val):
    uniq = np.unique(sid_val)
    rng = np.random.default_rng(1)
    rng.shuffle(uniq)
    a = set(uniq[: len(uniq) // 2].tolist())
    return np.isin(sid_val, list(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="features/synth_features_v4.npz")
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--synthfrac", type=float, default=1.0,
                    help="fraction of synth series to append")
    args = ap.parse_args()

    d = np.load(V4, allow_pickle=True)
    X = np.nan_to_num(d["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y, sid, t = d["y"].astype(np.int64), d["series_id"], d["t_online"]
    va = val_mask(sid); tr = ~va
    Xtr, ytr = X[tr], y[tr]
    Xv, yv, tv, sv = X[va], y[va], t[va], sid[va]
    print(f"real train rows={tr.sum():,}  VAL rows={va.sum():,}")

    s = np.load(args.synth, allow_pickle=True)
    Xs = np.nan_to_num(s["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ys, sids = s["y"].astype(np.int64), s["series_id"]
    if args.synthfrac < 1.0:
        uniq = np.unique(sids)
        keep = set(np.random.default_rng(0).choice(
            uniq, int(args.synthfrac * len(uniq)), replace=False).tolist())
        m = np.isin(sids, list(keep))
        Xs, ys = Xs[m], ys[m]
    print(f"synth rows appended={len(Xs):,}  (pos_rate={ys.mean():.3f})")

    halfA = half_mask(sv)

    def fit_eval(Xtrain, ytrain, tag):
        t0 = time.time()
        dtr = lgb.Dataset(Xtrain, label=ytrain)
        bst = lgb.train(PARAMS, dtr, num_boost_round=args.rounds)
        p = bst.predict(Xv)
        full = ts_auc_grouped(tv, yv, p)
        aA = ts_auc_grouped(tv[halfA], yv[halfA], p[halfA])
        aB = ts_auc_grouped(tv[~halfA], yv[~halfA], p[~halfA])
        print(f"[{tag}] full={full:.4f}  halfA={aA:.4f}  halfB={aB:.4f}  "
              f"min={min(aA,aB):.4f}  ({time.time()-t0:.0f}s)")
        return np.array([full, aA, aB])

    print(f"\ntraining real-only and augmented ({args.rounds} rounds each)...")
    rA = fit_eval(Xtr, ytr, "REAL-only ")
    Xaug = np.vstack([Xtr, Xs]); yaug = np.concatenate([ytr, ys])
    rB = fit_eval(Xaug, yaug, "REAL+SYNTH")
    d_full, d_a, d_b = (rB - rA)
    print(f"\nΔ(aug-real)  full={d_full:+.4f}  halfA={d_a:+.4f}  halfB={d_b:+.4f}")
    ok = (d_full > 0) and (d_a > 0) and (d_b > 0)
    print("VERDICT:", "AUGMENTATION HELPS (lifts full + both halves)" if ok
          else "no gain (fails honest gate) — augmentation does not help")


if __name__ == "__main__":
    main()
