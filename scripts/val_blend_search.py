"""Round 10 — comprehensive VAL blend search over the FULL cached-member palette.

Context: the real leaderboard proved VAL (2000 held-out train series) tracks the
public score; the 100-series reduced test is noise. The production blend weights
were tuned on that noisy gate, so re-optimising the blend on VAL is the highest-EV
cheap lever (every member's held-out VAL logits are already cached -> pure logit
arithmetic, no retraining).

All members are aligned to val_base_logits' canonical row order:
  - gbt0..3, logistic : val_base_logits.npz   (WITH log_t -- the restored config)
  - rank              : val_rank_logits.npz    (reindexed by (series_id, t_online))
  - GRU variants      : val_seq_logits*.npz    (with-log, nolog, seeds, raw)
  - LSTM              : val_seq_logits_lstm{0,1}.npz
  - CNN, ExtraTrees   : val_cnn_logits.npz, val_et_logits.npz

Gate = VAL full TS-AUC up AND neither honest half down (the robustness rule the
real leaderboard taught us). TS-AUC is invariant to the final sigmoid, so the
search works entirely in logit space.

Run: uv run python scripts/val_blend_search.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from sb.metric import ts_auc_grouped

EPS = 1e-9


def zscore(a):
    a = a.astype(np.float64)
    return (a - a.mean()) / (a.std() + EPS)


def halves(sid_val):
    """Same honest-half split as val_stepnorm_blend.py (seed 1, by series)."""
    uniq = np.unique(sid_val)
    np.random.default_rng(1).shuffle(uniq)
    a = set(uniq[: len(uniq) // 2].tolist())
    mA = np.isin(sid_val, list(a))
    return mA, ~mA


def main():
    vb = np.load("features/val_base_logits.npz")
    y = vb["y"].astype(np.int64)
    t = vb["t_online"].astype(np.int64)
    sid = vb["series_id"].astype(np.int64)
    gbt = vb["gbt_logits"].astype(np.float64)   # (N,4) WITH log_t
    lin = vb["log_logit"].astype(np.float64)
    N = len(y)
    K = int(t.max()) + 2
    print(f"VAL rows={N:,}  pos={y.mean():.4f}  steps=0..{t.max()}")

    # ---- rank: reindex (series_id,t_online) -> canonical order ----
    vr = np.load("features/val_rank_logits.npz")
    rkey = vr["series_id"].astype(np.int64) * K + vr["t_online"].astype(np.int64)
    ckey = sid * K + t
    o = np.argsort(rkey)
    pos = np.searchsorted(rkey[o], ckey)
    assert np.array_equal(rkey[o][pos], ckey), "rank reindex failed"
    rank = vr["rank_score"].astype(np.float64)[o][pos]

    # ---- bare canonical-order val_logit members ----
    def load_logit(path):
        return np.load(path)["val_logit"].astype(np.float64)

    members = {
        "gbt_mean": gbt.mean(axis=1),
        "logistic": lin,
        "rank": rank,
        "gru_nolog": np.mean([load_logit(f"features/val_seq_logits_0{i}_nolog.npz")
                              for i in (20, 21, 22)], axis=0),
        "gru_withlog": load_logit("features/val_seq_logits.npz"),
        "gru_seeds": np.mean([load_logit(f"features/val_seq_logits_s{i}.npz")
                              for i in (1, 2, 3, 4)], axis=0),
        "gru_raw": load_logit("features/val_seq_logits_raw.npz"),
        "lstm": np.mean([load_logit(f"features/val_seq_logits_lstm{i}.npz")
                         for i in (0, 1)], axis=0),
        "cnn": load_logit("features/val_cnn_logits.npz"),
        "et": load_logit("features/val_et_logits.npz"),
    }

    mA, mB = halves(sid)

    def auc(v, m=None):
        if m is None:
            return ts_auc_grouped(t, y, v)
        return ts_auc_grouped(t[m], y[m], v[m])

    # ---- per-member diagnostics (alignment sanity + strength) ----
    print(f"\n{'member':>12}{'full':>8}{'halfA':>8}{'halfB':>8}{'corr(gbt)':>10}")
    gm = members["gbt_mean"]
    aucs = {}
    for name, v in members.items():
        aucs[name] = auc(v)
        c = np.corrcoef(zscore(v), zscore(gm))[0, 1]
        flag = "  <-- MISALIGNED?" if aucs[name] < 0.52 else ""
        print(f"{name:>12}{aucs[name]:>8.4f}{auc(v, mA):>8.4f}{auc(v, mB):>8.4f}{c:>10.3f}{flag}")

    # ---- baseline = current production blend (log_t restored on GBTs) ----
    # gbt5 = (1-RANK_GW)*mean4 + RANK_GW*rank_rescaled ; base = (1-W_LIN)*gbt5 +
    # W_LIN*lin ; final = (1-W_GRU)*base + W_GRU*gru_mean.  Rank rescaled to gbt
    # logit scale (z-match), gru_mean = nolog GRUs (shipped set).
    def rescale(src, ref):
        return (src - src.mean()) / (src.std() + EPS) * ref.std() + ref.mean()

    def prod_blend(w_gru, w_lin, rank_gw, gru_key="gru_nolog"):
        mean4 = members["gbt_mean"]
        r = rescale(members["rank"], mean4)
        gbt5 = (1 - rank_gw) * mean4 + rank_gw * r
        base = (1 - w_lin) * gbt5 + w_lin * members["logistic"]
        return (1 - w_gru) * base + w_gru * members[gru_key]

    base_blend = prod_blend(0.45, 0.20, 0.15, "gru_nolog")
    bF, bA, bB = auc(base_blend), auc(base_blend, mA), auc(base_blend, mB)
    print(f"\nshipped-structure blend (log_t restored, nolog GRU, W_GRU=0.45 "
          f"W_LIN=0.20 RANK_GW=0.15):")
    print(f"   full={bF:.4f}  halfA={bA:.4f}  halfB={bB:.4f}")

    # ---- Phase A: re-tune the 3 existing weights + GRU-set choice ----
    print("\n--- Phase A: re-tune existing weights (nested structure) ---")
    bestA = (bF, bA, bB, 0.45, 0.20, 0.15, "gru_nolog")
    for gru_key in ("gru_nolog", "gru_withlog", "gru_seeds"):
        for w_gru in np.arange(0.30, 0.66, 0.05):
            for w_lin in np.arange(0.05, 0.41, 0.05):
                for rank_gw in np.arange(0.0, 0.31, 0.05):
                    v = prod_blend(w_gru, w_lin, rank_gw, gru_key)
                    f = auc(v)
                    if f > bestA[0] + EPS:
                        a, b = auc(v, mA), auc(v, mB)
                        # robust: neither half below the shipped baseline
                        if a >= bA - 1e-4 and b >= bB - 1e-4:
                            bestA = (f, a, b, w_gru, w_lin, rank_gw, gru_key)
    print(f"   best: full={bestA[0]:.4f} halfA={bestA[1]:.4f} halfB={bestA[2]:.4f}"
          f"  | W_GRU={bestA[3]:.2f} W_LIN={bestA[4]:.2f} RANK_GW={bestA[5]:.2f}"
          f" gru={bestA[6]}  (delta full {bestA[0]-bF:+.4f})")

    # ---- Phase B: greedy-add diverse members to the Phase-A blend (z-scored) ----
    print("\n--- Phase B: greedy-add diverse members (z-scored) ---")
    cur = prod_blend(bestA[3], bestA[4], bestA[5], bestA[6])
    cur = zscore(cur)
    curF, curA, curB = auc(cur), auc(cur, mA), auc(cur, mB)
    floorA, floorB = bestA[1], bestA[2]
    pool = ["lstm", "cnn", "et", "gru_raw", "gru_withlog", "gru_seeds", "gru_nolog"]
    added = []
    while True:
        best_step = None
        for name in pool:
            if name in added:
                continue
            zc = zscore(members[name])
            for w in np.arange(0.05, 0.51, 0.05):
                v = cur + w * zc
                f = auc(v)
                if best_step is None or f > best_step[0]:
                    a, b = auc(v, mA), auc(v, mB)
                    best_step = (f, a, b, name, w)
        if best_step is None:
            break
        f, a, b, name, w = best_step
        if f > curF + 1e-4 and a >= floorA - 1e-4 and b >= floorB - 1e-4:
            cur = cur + w * zscore(members[name])
            curF, curA, curB = f, a, b
            added.append(name)
            print(f"   + {name:>12} w={w:.2f}  -> full={f:.4f} halfA={a:.4f} halfB={b:.4f}")
        else:
            print(f"   (stop) best candidate {best_step[3]} w={best_step[4]:.2f} "
                  f"full={best_step[0]:.4f} not robust/insufficient")
            break

    print(f"\n=== SUMMARY ===")
    print(f"shipped-structure (log_t restored): full={bF:.4f} A={bA:.4f} B={bB:.4f}")
    print(f"Phase A (reweighted)              : full={bestA[0]:.4f} A={bestA[1]:.4f} B={bestA[2]:.4f}")
    print(f"Phase B (+ {added})")
    print(f"                                  : full={curF:.4f} A={curA:.4f} B={curB:.4f}")
    print(f"total VAL gain over shipped-structure: {curF-bF:+.4f}")
    print(f"(real public ~= VAL - 0.017; shipped #6 real=0.5959, prior best=0.5987)")


if __name__ == "__main__":
    main()
