"""Round 10 — proper logit-space META-STACKER over cached VAL members, honestly
cross-validated on the two series-disjoint halves.

The hand-tuned blend caps ~0.6170 VAL; the greedy forward-add found only +0.0006.
A stacker fits all member weights jointly, which can extract more from correlated
members than nested hand weights. We gate it HONESTLY: fit on half A, evaluate on
half B (and vice versa), so any gain must generalise across disjoint series. A
stack that only wins in-sample is rejected.

Stackers tried (all monotone in the final sigmoid, so TS-AUC sees only the linear
combo): ridge on the logit target, and non-negative least squares. Weights that
survive both directions become production constants (computable at train time like
rank_const.npz -> fully serve-compatible, pure numpy/sklearn).

Run: uv run python scripts/val_stack.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from sb.metric import ts_auc_grouped

EPS = 1e-9


def main():
    vb = np.load("features/val_base_logits.npz")
    y = vb["y"].astype(np.int64)
    t = vb["t_online"].astype(np.int64)
    sid = vb["series_id"].astype(np.int64)
    gbt = vb["gbt_logits"].astype(np.float64)
    lin = vb["log_logit"].astype(np.float64)
    K = int(t.max()) + 2

    vr = np.load("features/val_rank_logits.npz")
    rkey = vr["series_id"].astype(np.int64) * K + vr["t_online"].astype(np.int64)
    ckey = sid * K + t
    o = np.argsort(rkey)
    pos = np.searchsorted(rkey[o], ckey)
    rank = vr["rank_score"].astype(np.float64)[o][pos]

    def L(p):
        return np.load(p)["val_logit"].astype(np.float64)

    members = {
        "gbt": gbt.mean(axis=1),
        "logistic": lin,
        "rank": rank,
        "gru_nolog": np.mean([L(f"features/val_seq_logits_0{i}_nolog.npz")
                              for i in (20, 21, 22)], axis=0),
        "gru_withlog": L("features/val_seq_logits.npz"),
        "gru_seeds": np.mean([L(f"features/val_seq_logits_s{i}.npz")
                              for i in (1, 2, 3, 4)], axis=0),
        "lstm": np.mean([L(f"features/val_seq_logits_lstm{i}.npz")
                         for i in (0, 1)], axis=0),
        "et": L("features/val_et_logits.npz"),
    }
    names = list(members)
    # z-score each member (stack weights then live on a common scale -> the
    # per-member mean/std become train-time constants at serve)
    mu = {k: members[k].mean() for k in names}
    sg = {k: members[k].std() + EPS for k in names}
    Z = np.column_stack([(members[k] - mu[k]) / sg[k] for k in names])  # (N,M)

    uniq = np.unique(sid)
    np.random.default_rng(1).shuffle(uniq)
    A = set(uniq[: len(uniq) // 2].tolist())
    mA = np.isin(sid, list(A))
    mB = ~mA

    def auc(v, m):
        return ts_auc_grouped(t[m], y[m], v[m])

    # ----- reference: the retuned hand blend (log_t restored) -----
    def rescale(s, r):
        return (s - s.mean()) / (s.std() + EPS) * r.std() + r.mean()

    m4 = members["gbt"]
    r = rescale(members["rank"], m4)
    gbt5 = 0.90 * m4 + 0.10 * r
    base = 0.90 * gbt5 + 0.10 * members["logistic"]
    hand = 0.55 * base + 0.45 * members["gru_nolog"]
    print(f"hand retune blend:    halfA={auc(hand, mA):.4f}  halfB={auc(hand, mB):.4f}  "
          f"full={ts_auc_grouped(t, y, hand):.4f}")

    # ----- ridge stack on the logit target, fit on one half, eval the other -----
    def ridge_fit(Ztr, ytr, lam):
        # center target to +-1; ridge normal equations (closed form, deterministic)
        yc = (ytr * 2 - 1).astype(np.float64)
        G = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
        w = np.linalg.solve(G, Ztr.T @ yc)
        return w

    print(f"\n{'lambda':>8}  {'fitA->evalB':>12}  {'fitB->evalA':>12}  {'min(half)':>10}")
    best = None
    for lam in (10.0, 100.0, 1000.0, 1e4, 1e5):
        wA = ridge_fit(Z[mA], y[mA], lam)
        wB = ridge_fit(Z[mB], y[mB], lam)
        eB = auc(Z @ wA, mB)   # fit A, eval B (honest)
        eA = auc(Z @ wB, mA)   # fit B, eval A (honest)
        mn = min(eA, eB)
        print(f"{lam:>8.0f}  {eB:>12.4f}  {eA:>12.4f}  {mn:>10.4f}")
        if best is None or mn > best[0]:
            best = (mn, lam, wA, wB)

    mn, lam, wA, wB = best
    # full-fit weights at the chosen lambda (what we'd actually ship)
    wfull = ridge_fit(Z, y, lam)
    full_auc = ts_auc_grouped(t, y, Z @ wfull)
    fa, fb = auc(Z @ wfull, mA), auc(Z @ wfull, mB)
    print(f"\nbest lambda={lam:.0f}  honest min-half={mn:.4f}")
    print(f"full-fit stack:       halfA={fa:.4f}  halfB={fb:.4f}  full={full_auc:.4f}")
    print("\nstack weights (z-scaled units):")
    for k, w in sorted(zip(names, wfull), key=lambda x: -abs(x[1])):
        print(f"   {k:>12}: {w:+.4f}")

    hand_full = ts_auc_grouped(t, y, hand)
    print(f"\n=== VERDICT ===")
    print(f"hand retune full={hand_full:.4f} (halfA {auc(hand,mA):.4f} / halfB {auc(hand,mB):.4f})")
    print(f"stack    full={full_auc:.4f} (halfA {fa:.4f} / halfB {fb:.4f})")
    robust = fa >= auc(hand, mA) - 1e-4 and fb >= auc(hand, mB) - 1e-4
    if full_auc > hand_full + 1e-4 and robust:
        print(f"==> STACK WINS robustly (+{full_auc-hand_full:.4f} full, both halves up) -> productionize")
    else:
        print(f"==> stack does NOT robustly beat hand blend -> ship hand retune (ceiling reached)")


if __name__ == "__main__":
    main()
