"""Round 11 — generalized neural-stack gate for multiple decorrelated members.

The shipped blend is  final = (1-WG)*base + WG*neural  with neural = mean(3 GRUs).
Round 11 adds candidate neural members (calibrated-feature attention, raw-z
attention). A new member helps the BLEND only if it is decorrelated from EVERY
existing member (round-6 lesson: the round-6 raw CNN was decorrelated from the
GBTs but redundant with the GRU). So this gate:

  1. loads the GRU mean + each named candidate (rescaled to the GRU logit scale),
  2. reports standalone VAL + a full rank-corr MATRIX among {gru, candidates},
  3. forms the neural pillar as a convex mix  neural = sum_i w_i * member_i
     (w summing to 1) and searches the candidate weights (GRU gets the remainder)
     at WG in {0.45..0.55}, REQUIRING a lift on full VAL AND both honest halves,
  4. prints the best honest config vs the shipped (GRU-only neural) baseline.

Everything is pure-numpy on CACHED logits (no retrain). Members are passed as
name=path pairs; the GRU mean (020/021/022_nolog) is always member 0.

Run:
  uv run python scripts/attn_stack_search.py \
      calib=features/val_attn_s1.npz raw=features/val_attn_raw_concat.npz
"""
from __future__ import annotations

import itertools
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
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt(ra @ ra) * np.sqrt(rb @ rb) + 1e-12))


def z(a):
    return (a - a.mean()) / (a.std() + 1e-12)


def main():
    pairs = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        pairs.append((name, path))
    if not pairs:
        pairs = [("calib", "features/val_attn_s1.npz")]

    base = np.load("features/val_base_logits.npz")
    sv, tv = base["series_id"], base["t_online"]
    mean4 = base["gbt_logits"].mean(axis=1)
    log_logit = base["log_logit"]

    d = np.load("features/train_features_v4.npz", allow_pickle=True)
    sid, t, y = d["series_id"], d["t_online"], d["y"]
    uniq = np.unique(sid); rng = np.random.default_rng(SPLIT_SEED); rng.shuffle(uniq)
    va = np.isin(sid, list(set(uniq[: int(0.2 * len(uniq))].tolist())))
    yv_lk = {(int(a), int(b)): int(c) for a, b, c in zip(sid[va], t[va], y[va])}
    yv = np.array([yv_lk[(int(a), int(b))] for a, b in zip(sv, tv)])

    rk = np.load("features/val_rank_logits.npz")
    rk_key = {(int(a), int(b)): float(v)
              for a, b, v in zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank_score = np.array([rk_key[(int(a), int(b))] for a, b in zip(sv, tv)])

    grus = [np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"]
            for m in ("020", "021", "022")]
    gru_mean = np.mean(grus, axis=0)
    g_std, g_mean = gru_mean.std(), gru_mean.mean()

    # candidate members, rescaled to the GRU logit scale
    members = [("gru", gru_mean)]
    for name, path in pairs:
        a = np.load(path)
        al = a["val_logit"]
        if not (np.array_equal(a["series_id"], sv) and np.array_equal(a["t_online"], tv)):
            key = {(int(s), int(tt)): float(v)
                   for s, tt, v in zip(a["series_id"], a["t_online"], al)}
            al = np.array([key[(int(s), int(tt))] for s, tt in zip(sv, tv)])
        members.append((name, z(al) * g_std + g_mean))

    hA = half_mask(sv)
    rag = z(rank_score) * mean4.std() + mean4.mean()
    gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * rag
    base_logit = (1 - W_LIN) * gbt5 + W_LIN * log_logit

    def score(sh):
        return (ts_auc_grouped(tv, yv, sh),
                ts_auc_grouped(tv[hA], yv[hA], sh[hA]),
                ts_auc_grouped(tv[~hA], yv[~hA], sh[~hA]))

    def blend(neural, wg=W_GRU):
        return (1 - wg) * base_logit + wg * neural

    # --- standalone + rank-corr matrix ---
    print("standalone VAL TS-AUC:")
    for name, m in members:
        print(f"    {name:6s} {ts_auc_grouped(tv, yv, m):.4f}")
    print("rank-corr matrix:")
    names = [n for n, _ in members]
    print("           " + "  ".join(f"{n:>6s}" for n in names))
    for i, (ni, mi) in enumerate(members):
        cells = "  ".join(f"{_spearman(mi, mj):6.3f}" for _, mj in members)
        print(f"    {ni:6s} {cells}")

    bl = score(blend(gru_mean))
    print(f"\n  [shipped GRU-only] full={bl[0]:.4f} halfA={bl[1]:.4f} halfB={bl[2]:.4f}")

    # --- search candidate weights inside the neural pillar (GRU = remainder) ---
    cand = members[1:]
    grid = [round(x, 2) for x in np.arange(0.0, 0.61, 0.05)]
    best = (bl[0], None, W_GRU, bl)
    for wg in (0.45, 0.50, 0.55):
        for combo in itertools.product(grid, repeat=len(cand)):
            wsum = sum(combo)
            if wsum > 0.85:
                continue
            neural = (1.0 - wsum) * gru_mean
            for w, (_, m) in zip(combo, cand):
                neural = neural + w * m
            s = score(blend(neural, wg))
            if s[0] > best[0] and s[1] > bl[1] and s[2] > bl[2]:
                best = (s[0], combo, wg, s)

    if best[1] is None:
        print("\n  no honest weight beat the shipped blend on full + both halves.")
        return
    wtxt = ", ".join(f"{n}={w:.2f}" for (n, _), w in zip(cand, best[1]))
    print(f"\n  BEST honest: WG={best[2]:.2f}  {wtxt}  (gru={1-sum(best[1]):.2f})")
    print(f"    full={best[3][0]:.4f} ({best[3][0]-bl[0]:+.4f})  "
          f"halfA={best[3][1]:.4f} ({best[3][1]-bl[1]:+.4f})  "
          f"halfB={best[3][2]:.4f} ({best[3][2]-bl[2]:+.4f})")


if __name__ == "__main__":
    main()
