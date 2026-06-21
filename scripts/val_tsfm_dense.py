"""Round 11 — DECISIVE frozen-TSFM gate (strongest possible form).

Round 7 tested Chronos-mini embeddings with only 6 SPARSE snapshot steps and a
LINEAR head, on a per-snapshot z-blend. It was redundant (~0.515 AUC, rankcorr
0.4-0.5). Before concluding anything about the far-more-expensive FINE-TUNING path,
this script runs the strongest *frozen*-embedding test that round 7 lacked:

  1. DENSE per-step embeddings over a fine step grid (not 6 snapshots).
  2. A STRONG nonlinear head (logistic AND LightGBM) on the 1152-d diff vectors.
  3. The REAL gate we use for every member: top-level z-blend onto the *current*
     shipped logit (4 GBT + rank + logistic + 3 nolog GRUs, round-10 weights),
     scored with the true TS-AUC metric on VAL full + BOTH honest halves (seed 1).

Decision: if even this strongest frozen form cannot lift VAL full AND both honest
halves with LOW rankcorr, the frozen-TSFM signal is definitively redundant and a
fine-tuned encoder (which only re-weights the same representation) is very unlikely
to rescue it -> close the lever. If it shows robust orthogonal lift -> escalate to
end-to-end fine-tuning (justifies the deploy-blocker work: determinism + latency).

Run: HF_HUB_OFFLINE=1 uv run python scripts/val_tsfm_dense.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from sb.data import iter_series        # noqa: E402
from sb.metric import ts_auc_grouped   # noqa: E402

SEED = 42
HALF_SEED = 1
STEP_GRID = list(range(40, 401, 6))    # 61 dense steps across the metric-heavy band
TRAIL = 128
CTX = 512
EARLY = 64
CLIP = 8.0
MODEL = "models/chronos-t5-mini"
CACHE = "features/val_tsfm_dense.npz"
RANK_GW = 0.10
W_LIN = 0.10
W_GRU = 0.45


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def val_ids_split():
    import pandas as pd
    idx = pd.read_parquet("Dataset/y_train_index.parquet")
    uniq = np.array(sorted(idx.index.tolist()))
    np.random.default_rng(SEED).shuffle(uniq)
    val = uniq[: int(0.2 * len(uniq))]
    return set(int(v) for v in val), np.array(sorted(int(v) for v in val))


def reconstruct_shipped():
    """Round-10 shipped logit on VAL, keyed by (sid, t_online)."""
    base = np.load("features/val_base_logits.npz")
    sid = base["series_id"].astype(np.int64)
    t = base["t_online"].astype(np.int64)
    y = base["y"].astype(np.int64)
    gbt = base["gbt_logits"].astype(np.float64)       # (N,4)
    log_logit = base["log_logit"].astype(np.float64)
    mean4 = gbt.mean(axis=1)

    # rank member, aligned to base order by (sid,t)
    rk = np.load("features/val_rank_logits.npz")
    rk_key = {(int(s), int(tt)): float(v) for s, tt, v in
              zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank_score = np.array([rk_key[(int(s), int(tt))] for s, tt in zip(sid, t)])
    rank_as_gbt = (rank_score - rank_score.mean()) / (rank_score.std() + 1e-12)
    rank_as_gbt = rank_as_gbt * mean4.std() + mean4.mean()
    gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * rank_as_gbt
    base_logit = (1 - W_LIN) * gbt5 + W_LIN * log_logit

    gru = np.mean([np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"]
                   for m in ("020", "021", "022")], axis=0)
    shipped = (1 - W_GRU) * base_logit + W_GRU * gru

    lut = {(int(s), int(tt)): (float(sh), int(yy))
           for s, tt, sh, yy in zip(sid, t, shipped, y)}
    return lut


def embed_val(val_ids_sorted):
    import os
    if os.path.exists(CACHE):
        print(f"loading cached embeddings from {CACHE}")
        d = np.load(CACHE, allow_pickle=True)
        meta = d["meta"]
        embs = d["embs"]
        return {tuple(m): e for m, e in zip(meta, embs)}

    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="mps", torch_dtype=torch.float32)

    windows, meta = [], []
    n = 0
    for s in iter_series("train", ids=list(val_ids_sorted)):
        xh = s.x_hist.astype(np.float64)
        xo = s.x_online.astype(np.float64)
        if len(xo) <= STEP_GRID[0] + 5:
            continue
        mu, sd = xh.mean(), xh.std() + 1e-9
        zh = np.clip((xh - mu) / sd, -CLIP, CLIP)
        zo = np.clip((xo - mu) / sd, -CLIP, CLIP)
        windows.append(zh[-CTX:].astype(np.float32)); meta.append((s.series_id, "hist", -1))
        windows.append(zo[:EARLY].astype(np.float32)); meta.append((s.series_id, "early", -1))
        for t in STEP_GRID:
            if len(xo) <= t:
                continue
            windows.append(zo[max(0, t - CTX):t].astype(np.float32)); meta.append((s.series_id, "cum", t))
            windows.append(zo[max(0, t - TRAIL):t].astype(np.float32)); meta.append((s.series_id, "trail", t))
        n += 1
    print(f"series embedded: {n}, windows: {len(windows)}")

    embs = np.empty((len(windows), 384), dtype=np.float32)
    bs = 256
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(windows), bs):
            chunk = [torch.tensor(a, dtype=torch.float32) for a in windows[i:i + bs]]
            e, _ = pipe.embed(chunk)                       # (B, Lmax, 384) left-padded
            embs[i:i + len(chunk)] = e[:, -1, :].cpu().numpy().astype(np.float32)
            if (i // bs) % 25 == 0:
                el = time.time() - t0
                print(f"   embed {i + len(chunk)}/{len(windows)}  {el:.0f}s")
    np.savez(CACHE, meta=np.array(meta, dtype=object), embs=embs)
    print(f"cached embeddings -> {CACHE}")
    E = {}
    for m, e in zip(meta, embs):
        E[m] = e
    return E


def build_dataset(E, lut):
    rows_X, rows_y, rows_ship, rows_sid, rows_t = [], [], [], [], []
    sids = sorted({m[0] for m in E if m[1] == "hist"})
    for sid in sids:
        if (sid, "hist", -1) not in E or (sid, "early", -1) not in E:
            continue
        eh = E[(sid, "hist", -1)]
        ee = E[(sid, "early", -1)]
        for t in STEP_GRID:
            kc, kt = (sid, "cum", t), (sid, "trail", t)
            if kc not in E or kt not in E:
                continue
            look = lut.get((sid, t))
            if look is None:
                continue
            ec, et = E[kc], E[kt]
            rows_X.append(np.concatenate([ec - eh, et - eh, et - ee]))  # 1152-d
            rows_y.append(look[1]); rows_ship.append(look[0])
            rows_sid.append(sid); rows_t.append(t)
    return (np.asarray(rows_X, np.float32), np.asarray(rows_y, np.int64),
            np.asarray(rows_ship, np.float64), np.asarray(rows_sid, np.int64),
            np.asarray(rows_t, np.int64))


def oof_logistic(X, y, grp):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=grp):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.05, max_iter=3000)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    return oof


def oof_lgbm(X, y, grp):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=grp):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        params = dict(objective="binary", learning_rate=0.03, num_leaves=31,
                      feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                      min_data_in_leaf=200, verbose=-1, seed=0)
        bst = lgb.train(params, dtr, num_boost_round=300)
        oof[te] = bst.predict(X[te], raw_score=True)
    return oof


def gate(name, head_score, y, ship, sid, t, val_ids_sorted):
    # honest halves by SERIES (rng HALF_SEED)
    u = np.array(sorted(set(int(s) for s in sid)))
    np.random.default_rng(HALF_SEED).shuffle(u)
    halfA = set(int(v) for v in u[: len(u) // 2])
    inA = np.array([int(s) in halfA for s in sid])

    rc = float(np.corrcoef(ship, head_score)[0, 1])
    shz, hz = zscore(ship), zscore(head_score)

    def auc_at(mask, w):
        return ts_auc_grouped(t[mask], y[mask], sigmoid(shz[mask] + w * hz[mask]))

    base_full = ts_auc_grouped(t, y, sigmoid(shz))
    base_A = ts_auc_grouped(t[inA], y[inA], sigmoid(shz[inA]))
    base_B = ts_auc_grouped(t[~inA], y[~inA], sigmoid(shz[~inA]))
    print(f"\n=== HEAD: {name}   rankcorr={rc:+.3f} ===")
    print(f"shipped baseline   full={base_full:.4f}  halfA={base_A:.4f}  halfB={base_B:.4f}")
    best = None
    for w in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        f = auc_at(np.ones(len(y), bool), w)
        a = auc_at(inA, w); b = auc_at(~inA, w)
        flag = "  <-- both halves up" if (a > base_A + 1e-5 and b > base_B + 1e-5) else ""
        print(f"  w={w:4.2f}  full={f:.4f} ({f-base_full:+.4f})  "
              f"halfA={a:.4f} ({a-base_A:+.4f})  halfB={b:.4f} ({b-base_B:+.4f}){flag}")
        robust = min(f - base_full, a - base_A, b - base_B)
        if best is None or robust > best[1]:
            best = (w, robust)
    print(f"  BEST robust min-lift across (full,A,B): {best[1]:+.4f} @ w={best[0]}")
    return best[1]


def main():
    val_set, val_sorted = val_ids_split()
    print(f"VAL series: {len(val_sorted)}")
    lut = reconstruct_shipped()
    print(f"shipped lut: {len(lut)} (sid,t) keys")
    E = embed_val(val_sorted)
    X, y, ship, sid, t = build_dataset(E, lut)
    print(f"dataset: X{X.shape}  pos_rate={y.mean():.3f}  steps={len(set(t))}  series={len(set(sid))}")

    r1 = gate("logistic", oof_logistic(X, y, sid), y, ship, sid, t, val_sorted)
    r2 = gate("lightgbm", oof_lgbm(X, y, sid), y, ship, sid, t, val_sorted)

    print("\n================ VERDICT ================")
    best = max(r1, r2)
    if best > 0.0005:
        print(f"PROMISING: robust min-lift {best:+.4f} -> escalate to fine-tuning.")
    else:
        print(f"REDUNDANT: best robust min-lift {best:+.4f} (<= +0.0005). Frozen-TSFM "
              f"lever CLOSED even in strongest dense form; fine-tuning very unlikely to rescue.")


if __name__ == "__main__":
    main()
