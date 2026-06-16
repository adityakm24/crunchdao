"""Round 10 — test RANK-MEMBER upgrades on VAL (the rank member is the blend's
diversity gold, corr ~0.42 with the GBTs). Three rank boosters exist on disk:
  model_034_xendcg : rank_xendcg, nolog (151)  -- currently shipped, VAL 0.5945
  model_033_xendcg : rank_xendcg, WITH log_t (152), VAL 0.5955  -- rank is a tree
                     like the GBTs, which love log_t -> likely a free upgrade
  model_032_rank   : lambdarank,  WITH log_t (152), VAL 0.5833  -- different
                     objective -> decorrelated, good ensemble partner

We predict each on the held-out VAL rows (from the v4 cache), map to val_base_logits'
canonical row order, and test rank configs inside the production blend (log_t
restored on GBTs/logistic, nolog GRU, W_GRU=0.45, W_LIN=0.10) with a RANK_GW sweep.
Gate: VAL full up AND neither honest half down vs the current m034 blend.

Run: uv run python scripts/val_rank_ensemble.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
import lightgbm as lgb

from sb.metric import ts_auc_grouped

EPS = 1e-9
CACHE = "features/train_features_v4.npz"
SPLIT_SEED = 42
RANK_MODELS = {
    "m034_nolog": "artifacts/models/model_034_xendcg/lgbm_rank.txt",
    "m033_withlog": "artifacts/models/model_033_xendcg/lgbm_rank.txt",
    "m032_lambda": "artifacts/models/model_032_rank/lgbm_rank.txt",
}


def main():
    d = np.load(CACHE, allow_pickle=True)
    names = [str(n) for n in d["feature_names"]]
    name_to_col = {n: i for i, n in enumerate(names)}
    sid = d["series_id"].astype(np.int64)
    t = d["t_online"].astype(np.int64)

    uniq = np.unique(sid)
    np.random.default_rng(SPLIT_SEED).shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    va = np.isin(sid, list(val_ids))
    X = d["X"]
    Xva = X[va]
    sidv = sid[va]
    tv = t[va]
    print(f"VAL rows={va.sum():,}")

    # predict each rank booster on VAL (v4-cache order)
    rank_v4 = {}
    for tag, path in RANK_MODELS.items():
        b = lgb.Booster(model_file=path)
        cols = np.asarray([name_to_col[n] for n in b.feature_name()], dtype=np.int64)
        rank_v4[tag] = b.predict(Xva[:, cols]).astype(np.float64)
        print(f"  {tag:>14}: VAL TS-AUC {ts_auc_grouped(tv, d['y'][va].astype(np.int64), rank_v4[tag]):.4f}")

    # ---- canonical (val_base_logits) order: GBT/logistic/GRU members ----
    vb = np.load("features/val_base_logits.npz")
    y = vb["y"].astype(np.int64)
    tc = vb["t_online"].astype(np.int64)
    sidc = vb["series_id"].astype(np.int64)
    gbt = vb["gbt_logits"].astype(np.float64).mean(axis=1)
    lin = vb["log_logit"].astype(np.float64)
    K = int(max(tc.max(), tv.max())) + 2
    gru = np.mean([np.load(f"features/val_seq_logits_0{i}_nolog.npz")["val_logit"].astype(np.float64)
                   for i in (20, 21, 22)], axis=0)

    # map v4-order rank preds -> canonical order via (sid,t) key
    ckey = sidc * K + tc
    vkey = sidv * K + tv
    o = np.argsort(vkey)
    pos = np.searchsorted(vkey[o], ckey)
    assert np.array_equal(vkey[o][pos], ckey), "rank reindex failed (key not unique?)"
    rank = {tag: rank_v4[tag][o][pos] for tag in rank_v4}

    # sanity: y must match after remap
    assert np.array_equal(d["y"][va][o][pos], y), "VAL y mismatch after remap"

    uniq2 = np.unique(sidc)
    np.random.default_rng(1).shuffle(uniq2)
    A = set(uniq2[: len(uniq2) // 2].tolist())
    mA = np.isin(sidc, list(A)); mB = ~mA

    def auc(v, m=None):
        return ts_auc_grouped(tc, y, v) if m is None else ts_auc_grouped(tc[m], y[m], v[m])

    def rescale(s, r):
        return (s - s.mean()) / (s.std() + EPS) * r.std() + r.mean()

    def blend(rank_vec, rank_gw, w_lin=0.10, w_gru=0.45):
        r = rescale(rank_vec, gbt)
        gbt5 = (1 - rank_gw) * gbt + rank_gw * r
        base = (1 - w_lin) * gbt5 + w_lin * lin
        return (1 - w_gru) * base + w_gru * gru

    # correlations of rank variants with the GBT mean (lower = more diverse)
    print("\nrank-variant corr with GBT mean (lower=more decorrelated):")
    for tag in rank:
        c = np.corrcoef(rank[tag], gbt)[0, 1]
        print(f"   {tag:>14}: {c:+.3f}")

    rank_configs = {
        "m034 (shipped)": rank["m034_nolog"],
        "m033 withlog": rank["m033_withlog"],
        "mean(033,034)": 0.5 * (rescale(rank["m033_withlog"], gbt) + rescale(rank["m034_nolog"], gbt)),
        "mean(033,032)": 0.5 * (rescale(rank["m033_withlog"], gbt) + rescale(rank["m032_lambda"], gbt)),
        "mean(033,034,032)": (rescale(rank["m033_withlog"], gbt) + rescale(rank["m034_nolog"], gbt)
                              + rescale(rank["m032_lambda"], gbt)) / 3.0,
    }

    base_blend = blend(rank["m034_nolog"], 0.10)
    bF, bA, bB = auc(base_blend), auc(base_blend, mA), auc(base_blend, mB)
    print(f"\nbaseline (m034, RANK_GW=0.10): full={bF:.4f} halfA={bA:.4f} halfB={bB:.4f}")

    print(f"\n{'rank config':>20}{'gw':>6}{'full':>8}{'halfA':>8}{'halfB':>8}{'dFull':>8}")
    best = ("m034 (shipped)", 0.10, bF, bA, bB)
    for cfg, rvec in rank_configs.items():
        for gw in (0.10, 0.15, 0.20, 0.25):
            v = blend(rvec, gw)
            f, a, b = auc(v), auc(v, mA), auc(v, mB)
            mark = ""
            if f > bF + 1e-4 and a >= bA - 1e-4 and b >= bB - 1e-4:
                mark = "  <-- robust win"
                if f > best[2]:
                    best = (cfg, gw, f, a, b)
            print(f"{cfg:>20}{gw:>6.2f}{f:>8.4f}{a:>8.4f}{b:>8.4f}{f-bF:>+8.4f}{mark}")

    print(f"\n=== BEST: {best[0]} RANK_GW={best[1]:.2f} -> full={best[2]:.4f} "
          f"(halfA {best[3]:.4f} / halfB {best[4]:.4f}); gain over m034 {best[2]-bF:+.4f} ===")


if __name__ == "__main__":
    main()
