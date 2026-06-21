"""Round 11 — augmentation sweep (robustness + dose + tilt), local, no push.

Amortizes the REAL-only baseline (trained once per GBT seed) and sweeps multiple
synth sources x doses, so we can tell signal from single-GBT noise. Evaluates every
model on the SAME real VAL (full + honest halves seed1). VAL stays 100% REAL.

Decision: a robust augmentation lifts VAL full across BOTH gbt seeds and most doses,
ideally without a real halfA regression. Then we escalate to the blend-level gate.

Run: uv run python scripts/aug_sweep.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")
import lightgbm as lgb                       # noqa: E402
from sb.metric import ts_auc_grouped         # noqa: E402

V4 = "features/train_features_v4.npz"
SPLIT_SEED = 42
SYNTH = [
    ("faith_s0", "features/synth_features_v4.npz"),
    ("faith_s1", "features/synth_v4_s1.npz"),
    ("tilt_s2",  "features/synth_v4_tilt.npz"),
]
FRACS = [0.5, 1.0]
GBT_SEEDS = [42, 7]
ROUNDS = 800


def base_params(seed):
    return dict(objective="binary", metric="None", boosting_type="gbdt",
                num_leaves=31, learning_rate=0.03, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=1000,
                num_threads=0, seed=seed, force_row_wise=True, verbosity=-1)


def val_mask(sid):
    uniq = np.unique(sid); rng = np.random.default_rng(SPLIT_SEED); rng.shuffle(uniq)
    return np.isin(sid, list(set(uniq[: int(0.2 * len(uniq))].tolist())))


def half_mask(sid_val):
    uniq = np.unique(sid_val); rng = np.random.default_rng(1); rng.shuffle(uniq)
    return np.isin(sid_val, list(set(uniq[: len(uniq) // 2].tolist())))


def main():
    d = np.load(V4, allow_pickle=True)
    X = np.nan_to_num(d["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y, sid, t = d["y"].astype(np.int64), d["series_id"], d["t_online"]
    va = val_mask(sid); tr = ~va
    Xtr, ytr = X[tr], y[tr]
    Xv, yv, tv, sv = X[va], y[va], t[va], sid[va]
    hA = half_mask(sv)
    print(f"real train={tr.sum():,}  VAL={va.sum():,}")

    # load synth sources once
    synth = {}
    for tag, path in SYNTH:
        try:
            s = np.load(path, allow_pickle=True)
            Xs = np.nan_to_num(s["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            synth[tag] = (Xs, s["y"].astype(np.int64), s["series_id"])
            print(f"  loaded {tag}: {len(Xs):,} rows")
        except FileNotFoundError:
            print(f"  MISSING {path} (skip {tag})")

    def evl(bst):
        p = bst.predict(Xv)
        return (ts_auc_grouped(tv, yv, p),
                ts_auc_grouped(tv[hA], yv[hA], p[hA]),
                ts_auc_grouped(tv[~hA], yv[~hA], p[~hA]))

    for gs in GBT_SEEDS:
        t0 = time.time()
        b0 = lgb.train(base_params(gs), lgb.Dataset(Xtr, label=ytr), num_boost_round=ROUNDS)
        rf, ra, rb = evl(b0)
        print(f"\n=== GBT seed {gs} ===")
        print(f"  REAL-only      full={rf:.4f}  halfA={ra:.4f}  halfB={rb:.4f}  ({time.time()-t0:.0f}s)")
        for tag in synth:
            Xs, ys, _ = synth[tag]
            for fr in FRACS:
                if fr < 1.0:
                    us = np.unique(synth[tag][2])
                    keep = set(np.random.default_rng(0).choice(
                        us, int(fr * len(us)), replace=False).tolist())
                    m = np.isin(synth[tag][2], list(keep))
                    Xa = np.vstack([Xtr, Xs[m]]); ya = np.concatenate([ytr, ys[m]])
                else:
                    Xa = np.vstack([Xtr, Xs]); ya = np.concatenate([ytr, ys])
                b = lgb.train(base_params(gs), lgb.Dataset(Xa, label=ya), num_boost_round=ROUNDS)
                f, a, bb = evl(b)
                flag = "  <== lifts full+both" if (f > rf and a > ra and bb > rb) else \
                       ("  <- full+B up" if (f > rf and bb > rb) else "")
                print(f"  +{tag:9s} fr={fr:.2f}  full={f:.4f} ({f-rf:+.4f})  "
                      f"halfA={a:.4f} ({a-ra:+.4f})  halfB={bb:.4f} ({bb-rb:+.4f}){flag}")


if __name__ == "__main__":
    main()
