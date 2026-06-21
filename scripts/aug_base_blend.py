"""Round 11 — augmented-BASE blend gate (local, no push).

The decisive shippability test for GBT augmentation. Retrains the actual 4 base
members (model_003/008/009/015 configs: same params, ramp-50 weighting, their
feature subsets + seeds) on real-train-only vs real-train+synth, early-stopped on
real VAL. Builds mean4 each way, then the FULL round-10 blend (real rank + logistic
+ 3 real GRUs), and compares on VAL full + honest halves. VAL stays 100% real.

This catches the stepnorm trap: a single-GBT gain that the GRU already captures
washes out here. A blend lift on full + both halves = a real, shippable gain.

Run: uv run python scripts/aug_base_blend.py --synth features/synth_v4_tilt.npz [--rounds 1200]
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
RAMP = 50.0
RANK_GW, W_LIN, W_GRU = 0.10, 0.10, 0.45
# (member_id, seed) — params are shared (leaves31/lr.03/min1000/ff.8); feature
# subset is read from each saved booster's feature_name().
MEMBERS = [("model_003", 42), ("model_008", 7), ("model_009", 2026), ("model_015", 42)]


def val_mask(sid):
    uniq = np.unique(sid); rng = np.random.default_rng(SPLIT_SEED); rng.shuffle(uniq)
    return np.isin(sid, list(set(uniq[: int(0.2 * len(uniq))].tolist())))


def half_mask(sid_val):
    uniq = np.unique(sid_val); rng = np.random.default_rng(1); rng.shuffle(uniq)
    return np.isin(sid_val, list(set(uniq[: len(uniq) // 2].tolist())))


def elapsed_weights(sid, y, t_online):
    """Positives weighted clip((t-tau)/RAMP,0.2,1); negatives=1 (matches train.py)."""
    order = np.lexsort((t_online, sid))
    s_s, y_s, t_s = sid[order], y[order], t_online[order]
    w_s = np.ones(len(order))
    i, n = 0, len(order)
    while i < n:
        j = i
        while j < n and s_s[j] == s_s[i]:
            j += 1
        yy, tt = y_s[i:j], t_s[i:j]
        if yy.any():
            tau = tt[yy.argmax()]
            m = yy == 1
            w_s[i:j][m] = np.clip((tt[m] - tau) / RAMP, 0.2, 1.0)
        i = j
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    return w_s[inv]


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="features/synth_v4_tilt.npz")
    ap.add_argument("--rounds", type=int, default=1200)
    args = ap.parse_args()

    d = np.load(V4, allow_pickle=True)
    X = np.nan_to_num(d["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y, sid, t = d["y"].astype(np.int64), d["series_id"], d["t_online"]
    names = [str(n) for n in d["feature_names"]]
    col = {n: i for i, n in enumerate(names)}
    va = val_mask(sid); tr = ~va
    yv, tv, sv = y[va], t[va], sid[va]

    s = np.load(args.synth, allow_pickle=True)
    Xs = np.nan_to_num(s["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ys, sids, ts = s["y"].astype(np.int64), s["series_id"], s["t_online"]
    assert [str(n) for n in s["feature_names"]] == names, "synth schema mismatch"
    print(f"real train={tr.sum():,}  VAL={va.sum():,}  synth={len(Xs):,}")

    wtr = elapsed_weights(sid[tr], y[tr], t[tr])
    ws = elapsed_weights(sids, ys, ts)

    # member feature-name subsets from saved boosters
    subsets = {}
    for m, _ in MEMBERS:
        b = lgb.Booster(model_file=f"artifacts/models/{m}/lgbm_valsplit.txt")
        subsets[m] = b.feature_name()

    def train_member(m, seed, augment):
        cols = [col[n] for n in subsets[m]]
        Xtr = X[tr][:, cols]; ytr = y[tr]; wt = wtr
        if augment:
            Xtr = np.vstack([Xtr, Xs[:, cols]]); ytr = np.concatenate([ytr, ys])
            wt = np.concatenate([wt, ws])
        Xval = X[va][:, cols]
        params = dict(objective="binary", metric="None", num_leaves=31,
                      learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, min_data_in_leaf=1000, num_threads=0,
                      seed=seed, force_row_wise=True, verbosity=-1)
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wt)

        def feval(preds, ds):
            return ("tsauc", ts_auc_grouped(tv, yv, preds), True)
        dval = lgb.Dataset(Xval, label=yv, reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=args.rounds,
                        valid_sets=[dval], feval=feval,
                        callbacks=[lgb.early_stopping(120, verbose=False)])
        return _logit(bst.predict(Xval, num_iteration=bst.best_iteration))

    def mean4(augment):
        cols_logits = []
        for m, seed in MEMBERS:
            t0 = time.time()
            gl = train_member(m, seed, augment)
            cols_logits.append(gl)
            print(f"    {m} ({'AUG' if augment else 'real'}) "
                  f"VAL={ts_auc_grouped(tv, yv, gl):.4f}  ({time.time()-t0:.0f}s)")
        return np.vstack(cols_logits).T.mean(axis=1)

    # real rank + logistic + GRUs aligned to VAL base order
    base = np.load("features/val_base_logits.npz")
    assert np.array_equal(base["series_id"], sv) and np.array_equal(base["t_online"], tv)
    log_logit = base["log_logit"].astype(np.float64)
    rk = np.load("features/val_rank_logits.npz")
    rk_key = {(int(a), int(bb)): float(v) for a, bb, v in
              zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank_score = np.array([rk_key[(int(a), int(bb))] for a, bb in zip(sv, tv)])
    gru = np.mean([np.load(f"features/val_seq_logits_{mm}_nolog.npz")["val_logit"]
                   for mm in ("020", "021", "022")], axis=0)
    hA = half_mask(sv)

    def blend(m4):
        rag = (rank_score - rank_score.mean()) / (rank_score.std() + 1e-12)
        rag = rag * m4.std() + m4.mean()
        gbt5 = (1 - RANK_GW) * m4 + RANK_GW * rag
        bl = (1 - W_LIN) * gbt5 + W_LIN * log_logit
        return (1 - W_GRU) * bl + W_GRU * gru

    def rep(tag, m4):
        sh = blend(m4)
        f = ts_auc_grouped(tv, yv, sh)
        a = ts_auc_grouped(tv[hA], yv[hA], sh[hA])
        b = ts_auc_grouped(tv[~hA], yv[~hA], sh[~hA])
        print(f"  [{tag}] blend full={f:.4f}  halfA={a:.4f}  halfB={b:.4f}")
        return np.array([f, a, b])

    print("\n--- real-only base ---")
    r = rep("REAL ", mean4(False))
    print("\n--- augmented base ---")
    aug = rep("AUG  ", mean4(True))
    dd = aug - r
    print(f"\nΔ(aug-real) blend  full={dd[0]:+.4f}  halfA={dd[1]:+.4f}  halfB={dd[2]:+.4f}")
    print("VERDICT:", "AUG LIFTS BLEND (full+both halves)" if (dd > 0).all()
          else ("full+B up" if dd[0] > 0 and dd[2] > 0 else "no robust blend gain"))


if __name__ == "__main__":
    main()
