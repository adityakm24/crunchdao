"""Identify WHICH cached attention member(s) give the banked +0.0022 blend lift,
so the retrain (for servable weights) targets the right config. Pure-numpy on
cached logits: reconstruct the shipped 3-GRU blend, then add each attn file (and
the s1+s2 mean) as a 4th neural member exactly like train_attn.py's gate, and
report full + both honest halves (seed 1) lift. Read-only.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.path.insert(0, "src")
from sb.metric import ts_auc_grouped  # noqa: E402

RANK_GW, W_LIN, W_GRU = 0.10, 0.10, 0.45


def z(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def main():
    base = np.load("features/val_base_logits.npz")
    sid, t, y = base["series_id"], base["t_online"], base["y"].astype(np.int64)
    mean4 = base["gbt_logits"].mean(axis=1)
    log_logit = base["log_logit"]
    rk = np.load("features/val_rank_logits.npz")
    rkey = {(int(a), int(b)): float(v)
            for a, b, v in zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank = np.array([rkey[(int(a), int(b))] for a, b in zip(sid, t)])
    grus = [np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"]
            for m in ("020", "021", "022")]
    gru = np.mean(grus, axis=0)

    uniq = np.unique(sid)
    rng = np.random.default_rng(1); rng.shuffle(uniq)
    hA = np.isin(sid, list(set(uniq[: len(uniq) // 2].tolist())))

    def blend(neural):
        rag = z(rank) * mean4.std() + mean4.mean()
        gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * rag
        b = (1 - W_LIN) * gbt5 + W_LIN * log_logit
        return (1 - W_GRU) * b + W_GRU * neural

    def trio(neural):
        s = blend(neural)
        return np.array([ts_auc_grouped(t, y, s),
                         ts_auc_grouped(t[hA], y[hA], s[hA]),
                         ts_auc_grouped(t[~hA], y[~hA], s[~hA])])

    r = trio(gru)
    print(f"shipped 3-GRU baseline: full={r[0]:.4f} A={r[1]:.4f} B={r[2]:.4f}\n")

    def rescale(a):
        return z(a) * gru.std() + gru.mean()

    def lut(f):
        d = np.load(f)
        if np.array_equal(d["series_id"], sid) and np.array_equal(d["t_online"], t):
            return d["val_logit"].astype(np.float64)
        m = {(int(a), int(b)): float(v)
             for a, b, v in zip(d["series_id"], d["t_online"], d["val_logit"])}
        return np.array([m[(int(a), int(b))] for a, b in zip(sid, t)])

    files = sorted(glob.glob("features/val_attn*.npz"))
    cache = {f: lut(f) for f in files}
    print(f"{'member (+attn as 4th neural)':42s} {'full':>8s} {'A':>8s} {'B':>8s}   robustΔ")
    rows = []
    for f in files:
        a4 = np.mean(grus + [rescale(cache[f])], axis=0)
        d = trio(a4) - r
        rows.append((min(d), f, d))
        print(f"  {f.split('/')[-1]:40s} {d[0]:+.4f} {d[1]:+.4f} {d[2]:+.4f}   {min(d):+.4f}")

    # 2-seed means (s1 with each other base seed)
    print("\n2-seed / multi-seed combos:")
    combos = {
        "s1+s2": ["features/val_attn_s1.npz", "features/val_attn_s2.npz"],
        "s1+s2+s3": ["features/val_attn_s1.npz", "features/val_attn_s2.npz",
                     "features/val_attn_s3.npz"],
    }
    for nm, fs in combos.items():
        if all(f in cache for f in fs):
            am = np.mean([cache[f] for f in fs], axis=0)
            a4 = np.mean(grus + [rescale(am)], axis=0)
            d = trio(a4) - r
            rows.append((min(d), nm, d))
            print(f"  {nm:40s} {d[0]:+.4f} {d[1]:+.4f} {d[2]:+.4f}   {min(d):+.4f}")

    best = max(rows, key=lambda x: x[0])
    print(f"\n>>> BEST robust-min member = {best[1]}  (robust {best[0]:+.4f}, "
          f"full {best[2][0]:+.4f})")


if __name__ == "__main__":
    main()
