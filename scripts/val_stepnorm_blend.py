"""Round 10 — re-validate STEP-CONDITIONAL NULL STANDARDIZATION at the BLEND level
on VAL (2000 held-out train series), now that the real leaderboard has proven VAL
(not the 100-series reduced test) tracks the public score.

The round-9 stepnorm probe was a SINGLE GBT (VAL 0.5928 -> 0.6013, +0.0085) and was
WRONGLY rejected on the 100-series reduced test (-0.0038). Submission #6 (log_t
dropped on that same misleading reduced gate) REGRESSED the real score 0.5987 ->
0.5959, confirming VAL is the trustworthy gate. This re-tests stepnorm where it
matters: the full production blend, measured on VAL + honest halves.

Method (leak-free): reconstruct the VAL split (seed 42, 20%). Retrain the 4 base
GBTs (sub.ENSEMBLE, log_t KEPT) on VAL-train with vs without stepnorm; predict VAL.
Hold the OTHER members fixed at their cached HELD-OUT VAL logits (logistic from
val_base_logits, GRU = mean of the 3 cached GRU val logits) so they cancel in the
delta and isolate the GBT-side stepnorm effect on the blend.

stepnorm uses a PURE-NUMPY per-step smoother (no scipy) so the probe validates the
exact transform that will be productionized.

Run: uv run python scripts/val_stepnorm_blend.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "submission")

import numpy as np
import lightgbm as lgb

from sb.metric import ts_auc_grouped
import main as sub

CACHE = "features/train_features_v4.npz"
SPLIT_SEED = 42
SMOOTH = 25  # odd -> exact length-preserving centered moving average
CLIP = 12.0
W_LIN = sub.W_LIN
W_GRU = sub.W_GRU
GRU_CACHES = [
    "features/val_seq_logits_020_nolog.npz",
    "features/val_seq_logits_021_nolog.npz",
    "features/val_seq_logits_022_nolog.npz",
]


def smooth_t(a, size):
    """Centered moving average along axis 0 with edge replication (pure numpy).
    size must be odd so the output length equals the input length exactly."""
    if size <= 1:
        return a.copy()
    pad = size // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    cs = np.cumsum(ap, axis=0)
    cs = np.vstack([np.zeros((1, a.shape[1]), dtype=cs.dtype), cs])
    return (cs[size:] - cs[:-size]) / float(size)


def step_null_stats(X, t, null_mask, T):
    """Per-online-step cross-sectional NULL (y==0) mean/std, smoothed over t."""
    tn = t[null_mask]
    Xn = X[null_mask]
    cnt = np.maximum(np.bincount(tn, minlength=T).astype(np.float64), 1.0)
    nf = X.shape[1]
    mu = np.zeros((T, nf)); sd = np.ones((T, nf))
    for j in range(nf):
        col = Xn[:, j].astype(np.float64)
        s = np.bincount(tn, weights=col, minlength=T)
        sq = np.bincount(tn, weights=col * col, minlength=T)
        m = s / cnt
        v = np.maximum(sq / cnt - m * m, 0.0)
        mu[:, j] = m
        sd[:, j] = np.sqrt(v)
    mu = smooth_t(mu, SMOOTH)
    sd = smooth_t(sd, SMOOTH)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def halves(sid_val):
    uniq = np.unique(sid_val)
    np.random.default_rng(1).shuffle(uniq)
    a = set(uniq[: len(uniq) // 2].tolist())
    mA = np.isin(sid_val, list(a))
    return mA, ~mA


def keep_with_logt(version):
    col = sub._NAME_TO_COL["log_t"]
    return sorted(set(sub._keep_for(version)) | {col})


def train_members(X, y, w, tr, keeps):
    boosters = []
    for (seed, rounds, version), keep in zip(sub.ENSEMBLE, keeps):
        names = [sub.FEATURE_NAMES[i] for i in keep]
        dtr = lgb.Dataset(X[tr][:, keep], label=y[tr], weight=w[tr],
                          feature_name=names)
        boosters.append(lgb.train(dict(sub.LGB_PARAMS, seed=seed), dtr,
                                   num_boost_round=rounds))
    return boosters


def mean4_logit(boosters, keeps, Xva):
    acc = np.zeros(Xva.shape[0])
    for b, keep in zip(boosters, keeps):
        p = b.predict(Xva[:, keep])
        acc += np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1 - 1e-7))
    return acc / len(boosters)


def reconstruct_ramp_weights(sid, t, y):
    import pandas as pd
    df = pd.DataFrame({"sid": sid, "t": t.astype(np.int64), "y": y})
    tau = df["t"].where(df["y"] == 1).groupby(df["sid"]).transform("min").to_numpy()
    w = np.ones(len(y))
    pm = y == 1
    w[pm] = np.clip((t[pm] - tau[pm]) / sub.RAMP, 0.2, 1.0)
    return w


def main():
    d = np.load(CACHE, allow_pickle=True)
    X = np.ascontiguousarray(d["X"])
    y = d["y"].astype(np.int32)
    sid = d["series_id"].astype(np.int64)
    t = d["t_online"].astype(np.int64)
    assert [str(n) for n in d["feature_names"]] == list(sub.FEATURE_NAMES)
    T = int(t.max()) + 1

    uniq = np.unique(sid)
    np.random.default_rng(SPLIT_SEED).shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    va = np.isin(sid, list(val_ids))
    tr = ~va
    print(f"train rows={tr.sum():,}  val rows={va.sum():,}")

    w = reconstruct_ramp_weights(sid, t, y)
    keeps = [keep_with_logt(v) for (_, _, v) in sub.ENSEMBLE]

    # ---- align cached held-out VAL logits (canonical order) to cache-VAL order ----
    vb = np.load("features/val_base_logits.npz")
    vb_t = vb["t_online"].astype(np.int64); vb_y = vb["y"].astype(np.int64)
    canon_key = vb["series_id"].astype(np.int64) * 1000 + vb_t
    lin_canon = vb["log_logit"].astype(np.float64)
    gru_canon = np.mean([np.load(f)["val_logit"].astype(np.float64)
                         for f in GRU_CACHES], axis=0)
    # sanity: cached members must be in val_base_logits' canonical order
    print(f"sanity (canonical): logistic TS-AUC {ts_auc_grouped(vb_t, vb_y, lin_canon):.4f}"
          f"  gru TS-AUC {ts_auc_grouped(vb_t, vb_y, gru_canon):.4f}"
          f"  (expect ~0.578 / ~0.60 if aligned)")
    cache_key = sid[va] * 1000 + t[va]
    oc = np.argsort(canon_key)
    ok = np.argsort(cache_key)
    N = va.sum()
    lin = np.empty(N); gru = np.empty(N); ycanon = np.empty(N, dtype=np.int64)
    lin[ok] = lin_canon[oc]
    gru[ok] = gru_canon[oc]
    ycanon[ok] = vb["y"].astype(np.int64)[oc]
    assert (ycanon == y[va].astype(np.int64)).all(), "VAL alignment mismatch"

    tva = t[va]; yva = y[va].astype(np.int64)
    mA, mB = halves(sid[va])

    # ---- RAW arm ----
    print("\ntraining 4 GBTs (RAW, log_t kept) ...")
    t0 = time.time()
    bts_raw = train_members(X, y, w, tr, keeps)
    m4_raw = mean4_logit(bts_raw, keeps, X[va])
    print(f"  done ({time.time()-t0:.0f}s)")

    # ---- STEPNORM arm ----
    print("computing per-step null stats (pure numpy) + stepnorm transform ...")
    t0 = time.time()
    mu, sd = step_null_stats(X, t, tr & (y == 0), T)
    Xn = np.clip((X - mu[t]) / sd[t], -CLIP, CLIP).astype(np.float32)
    print(f"  transform done ({time.time()-t0:.0f}s); training 4 GBTs (STEPNORM) ...")
    t0 = time.time()
    bts_sn = train_members(Xn, y, w, tr, keeps)
    m4_sn = mean4_logit(bts_sn, keeps, Xn[va])
    print(f"  done ({time.time()-t0:.0f}s)")

    # ---- blends (rank excluded, gw=0, to match reduced_nolog isolation) ----
    def base_of(m4):
        return (1.0 - W_LIN) * m4 + W_LIN * lin

    def final_of(m4):
        return (1.0 - W_GRU) * base_of(m4) + W_GRU * gru

    def row(tag, m4):
        f = final_of(m4)
        return (tag, ts_auc_grouped(tva, yva, m4),
                ts_auc_grouped(tva, yva, base_of(m4)),
                ts_auc_grouped(tva, yva, f),
                ts_auc_grouped(tva[mA], yva[mA], f[mA]),
                ts_auc_grouped(tva[mB], yva[mB], f[mB]))

    r_raw = row("RAW    ", m4_raw)
    r_sn = row("STEPNORM", m4_sn)
    print(f"\n{'arm':>10}{'mean4':>9}{'base':>9}{'final':>9}{'halfA':>9}{'halfB':>9}")
    for r in (r_raw, r_sn):
        print(f"{r[0]:>10}{r[1]:>9.4f}{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>9.4f}{r[5]:>9.4f}")
    d_full = r_sn[3] - r_raw[3]
    d_a = r_sn[4] - r_raw[4]
    d_b = r_sn[5] - r_raw[5]
    print(f"{'delta':>10}{r_sn[1]-r_raw[1]:>9.4f}{r_sn[2]-r_raw[2]:>9.4f}"
          f"{d_full:>9.4f}{d_a:>9.4f}{d_b:>9.4f}")
    verdict = ("WIN: stepnorm lifts the blend on VAL + both halves -> productionize"
               if d_full > 1e-4 and d_a > -1e-4 and d_b > -1e-4
               else "flat/negative on the blend -> do NOT productionize; just restore log_t")
    print(f"\n==> {verdict}")


if __name__ == "__main__":
    main()
