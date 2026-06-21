"""Round 11 — weight-search gate for the causal-attention member(s).

The probe (train_attn.py) added attention as a NAIVE equal-weight 4th neural
member: neural = mean(3 GRU + 1 attn) => attn carries only 0.45*1/4 = 0.11 of the
final logit, yet still lifted the blend +0.0017 (full) on VAL + both honest halves.

This script extracts the lever properly. It works ONLY from CACHED logits (no GBT/
GRU/attn retrain) and:
  1. builds an attention SUB-ENSEMBLE (mean of N seeds, rescaled to GRU logit scale),
  2. reconstructs the exact round-10 GBT pillar + GRU pillar,
  3. searches the attention weight in two modes:
       A) share the neural pillar:  neural = (1-wa)*gru_mean + wa*attn_sub
                                    shipped = (1-WG)*base + WG*neural   (WG=0.45)
          (wa=0.25 reproduces the probe's "equal 4th member" anchor.)
       B) optional joint (wa, WG) grid (stronger neural pillar may want WG>0.45).
  4. picks the weight maximising FULL VAL TS-AUC SUBJECT TO lifting BOTH honest
     halves vs the current shipped blend (rejects weights that overfit full VAL).

Run:  uv run python scripts/attn_blend_search.py features/val_attn_s1.npz [s2.npz ...]
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
from sb.metric import ts_auc_grouped

SPLIT_SEED = 42
RANK_GW, W_LIN, W_GRU = 0.10, 0.10, 0.45


def half_mask(sid_val):
    uniq = np.unique(sid_val)
    rng = np.random.default_rng(1)
    rng.shuffle(uniq)
    return np.isin(sid_val, list(set(uniq[: len(uniq) // 2].tolist())))


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt(ra @ ra) * np.sqrt(rb @ rb) + 1e-12))


def z(a):
    return (a - a.mean()) / (a.std() + 1e-12)


def main():
    attn_files = sys.argv[1:] or ["features/val_attn_s1.npz"]

    base = np.load("features/val_base_logits.npz")
    sv, tv = base["series_id"], base["t_online"]
    mean4 = base["gbt_logits"].mean(axis=1)
    log_logit = base["log_logit"]

    # target labels aligned to the cached VAL order
    d = np.load("features/train_features_v4.npz", allow_pickle=True)
    sid, t, y = d["series_id"], d["t_online"], d["y"]
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    va = np.isin(sid, list(set(uniq[: int(0.2 * len(uniq))].tolist())))
    yv_lookup = {(int(a), int(b)): int(c) for a, b, c in zip(sid[va], t[va], y[va])}
    yv = np.array([yv_lookup[(int(a), int(b))] for a, b in zip(sv, tv)])

    rk = np.load("features/val_rank_logits.npz")
    rk_key = {(int(a), int(b)): float(v)
              for a, b, v in zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank_score = np.array([rk_key[(int(a), int(b))] for a, b in zip(sv, tv)])

    grus = [np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"]
            for m in ("020", "021", "022")]
    gru_mean = np.mean(grus, axis=0)

    # attention sub-ensemble: average raw seed logits, then rescale to GRU scale
    attn_raw = []
    for f in attn_files:
        a = np.load(f)
        al = a["val_logit"]
        # align if order differs (defensive; train_attn saves in cached order)
        if not (np.array_equal(a["series_id"], sv) and np.array_equal(a["t_online"], tv)):
            key = {(int(s), int(tt)): float(v)
                   for s, tt, v in zip(a["series_id"], a["t_online"], al)}
            al = np.array([key[(int(s), int(tt))] for s, tt in zip(sv, tv)])
        attn_raw.append(al)
    attn_sub_raw = np.mean(attn_raw, axis=0)
    attn_sub = z(attn_sub_raw) * gru_mean.std() + gru_mean.mean()

    hA = half_mask(sv)

    # ---- fixed GBT+rank+logistic pillar (round-10) ----
    rag = z(rank_score) * mean4.std() + mean4.mean()
    gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * rag
    base_logit = (1 - W_LIN) * gbt5 + W_LIN * log_logit

    def score(sh):
        return (ts_auc_grouped(tv, yv, sh),
                ts_auc_grouped(tv[hA], yv[hA], sh[hA]),
                ts_auc_grouped(tv[~hA], yv[~hA], sh[~hA]))

    def blend_modeA(wa, wg=W_GRU):
        neural = (1 - wa) * gru_mean + wa * attn_sub
        return (1 - wg) * base_logit + wg * neural

    print(f"attn seeds: {len(attn_files)}  ({', '.join(attn_files)})")
    print(f"rank-corr(attn_sub, gru_mean) = {_spearman(attn_sub_raw, gru_mean):.3f}")
    print(f"attn_sub standalone VAL TS-AUC = {ts_auc_grouped(tv, yv, attn_sub):.4f}\n")

    # baseline = current shipped blend (no attention)
    bl = score(blend_modeA(0.0))
    print(f"  [shipped wa=0.00       ] full={bl[0]:.4f} halfA={bl[1]:.4f} halfB={bl[2]:.4f}")
    anchor = score(blend_modeA(0.25))
    print(f"  [probe   wa=0.25       ] full={anchor[0]:.4f} halfA={anchor[1]:.4f} "
          f"halfB={anchor[2]:.4f}  Δfull={anchor[0]-bl[0]:+.4f}\n")

    # ---- Mode A: search wa at WG=0.45 ----
    print("Mode A  (share neural pillar, WG=0.45):")
    best = (bl[0], 0.0, bl)
    for wa in np.round(np.arange(0.05, 0.71, 0.05), 2):
        s = score(blend_modeA(float(wa)))
        ok = "  *both halves up" if (s[1] > bl[1] and s[2] > bl[2]) else ""
        flag = " <-- best-full" if s[0] > best[0] else ""
        print(f"    wa={wa:.2f}  full={s[0]:.4f} ({s[0]-bl[0]:+.4f})  "
              f"halfA={s[1]:.4f} ({s[1]-bl[1]:+.4f})  halfB={s[2]:.4f} "
              f"({s[2]-bl[2]:+.4f}){flag}{ok}")
        if s[0] > best[0] and s[1] > bl[1] and s[2] > bl[2]:
            best = (s[0], float(wa), s)
    print(f"\n  Mode A best (honest): wa={best[1]:.2f}  full={best[2][0]:.4f} "
          f"halfA={best[2][1]:.4f} halfB={best[2][2]:.4f}  "
          f"Δfull={best[2][0]-bl[0]:+.4f}")

    # ---- Mode B: joint (wa, WG) grid ----
    print("\nMode B  (joint wa x WG grid; honest = both halves up):")
    bestB = (bl[0], 0.0, W_GRU, bl)
    for wg in np.round(np.arange(0.40, 0.61, 0.05), 2):
        for wa in np.round(np.arange(0.10, 0.61, 0.10), 2):
            s = score(blend_modeA(float(wa), float(wg)))
            if s[0] > bestB[0] and s[1] > bl[1] and s[2] > bl[2]:
                bestB = (s[0], float(wa), float(wg), s)
    print(f"  Mode B best (honest): wa={bestB[1]:.2f} WG={bestB[2]:.2f}  "
          f"full={bestB[3][0]:.4f} halfA={bestB[3][1]:.4f} halfB={bestB[3][2]:.4f}  "
          f"Δfull={bestB[3][0]-bl[0]:+.4f}")


if __name__ == "__main__":
    main()
