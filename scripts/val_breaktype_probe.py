"""Round 11 — per-break-TYPE oracle probe (premise test for a Mixture-of-Experts).

User idea #2: per-break-type experts + a gate. Before building any MoE, test the
PREMISE cheaply and decisively with an ORACLE: give the router the TRUE break type
(mean-shift / variance / both / subtle) and ask whether ANY existing member beats
the shipped blend on some type-cohort. Routing can only help if (a) different
members win on different types with a real margin. The oracle removes the gate's
job entirely (perfect type knowledge) -> it is an UPPER BOUND on what a real
gate+expert system could achieve. If the shipped blend dominates every type even
against this cheating oracle, the MoE lever is dead and no gate can rescue it.

Type of a broken series (from the TRUE tau, measured like eda_dgp):
  dmean = (post.mean - pre.mean)/sd_hist ; lvr = log(post.var/pre.var) ; KS(pre,post)
  subtle   : KS < 0.15                       (the 68% cohort, documented floor)
  mean     : |dmean| > 0.25 and |lvr| < 0.3
  var      : |dmean| < 0.10 and |lvr| > 0.4
  both     : |dmean| > 0.25 and |lvr| > 0.4
  other    : the remainder
A positive row (t >= tau) inherits its series' type; every negative row is shared.
For each type T we score (type-T positives + ALL negatives) per step and report
TS-AUC of the shipped blend and each candidate member. Headroom exists only if
max_member(T) - shipped(T) is clearly positive for some T.

Read-only: cached VAL logits + measured types. Run:
  uv run python scripts/val_breaktype_probe.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from sb.data import iter_series        # noqa: E402
from sb.metric import ts_auc_grouped   # noqa: E402

SEED = 42
RANK_GW, W_LIN, W_GRU = 0.10, 0.10, 0.45


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def ks(a, b):
    grid = np.concatenate([a, b]); grid.sort()
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def classify(xh, xo, tau):
    pre, post = xo[:tau], xo[tau:]
    if len(pre) < 5 or len(post) < 5:
        return "other"
    sd = xh.std() + 1e-9
    dmean = abs((post.mean() - pre.mean()) / sd)
    lvr = abs(np.log((post.var() + 1e-9) / (pre.var() + 1e-9)))
    k = ks(pre, post)
    if k < 0.15:
        return "subtle"
    if dmean > 0.25 and lvr > 0.4:
        return "both"
    if dmean > 0.25 and lvr < 0.3:
        return "mean"
    if dmean < 0.10 and lvr > 0.4:
        return "var"
    return "other"


def keyed(path, key, sid, t):
    """Load val_logit-like file, align to (sid,t) order by lookup if needed."""
    d = np.load(path)
    v = d[key].astype(np.float64)
    if "series_id" in d.files and not (
        np.array_equal(d["series_id"], sid) and np.array_equal(d["t_online"], t)
    ):
        lut = {(int(s), int(tt)): float(x)
               for s, tt, x in zip(d["series_id"], d["t_online"], v)}
        v = np.array([lut[(int(s), int(tt))] for s, tt in zip(sid, t)])
    return v


def main():
    base = np.load("features/val_base_logits.npz")
    sid = base["series_id"].astype(np.int64)
    t = base["t_online"].astype(np.int64)
    y = base["y"].astype(np.int64)
    gbt = base["gbt_logits"].astype(np.float64)
    log_logit = base["log_logit"].astype(np.float64)
    mean4 = gbt.mean(axis=1)
    rank = keyed("features/val_rank_logits.npz", "rank_score", sid, t)
    rank_as_gbt = zscore(rank) * mean4.std() + mean4.mean()
    gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * rank_as_gbt
    base_logit = (1 - W_LIN) * gbt5 + W_LIN * log_logit
    grus = [keyed(f"features/val_seq_logits_{m}_nolog.npz", "val_logit", sid, t)
            for m in ("020", "021", "022")]
    gru = np.mean(grus, axis=0)
    shipped = (1 - W_GRU) * base_logit + W_GRU * gru

    members = {
        "gbt_mean": mean4, "logistic": log_logit, "rank": rank,
        "gru": gru, "shipped": shipped,
    }
    for nm, pth, ky in [
        ("attn", "features/val_attn_s1.npz", "val_logit"),
        ("tsfm_ft", "features/val_tsfm_ft_s0.npz", "val_logit"),
    ]:
        try:
            members[nm] = keyed(pth, ky, sid, t)
        except Exception as e:
            print(f"(skip {nm}: {e})")

    # ---- per-series break type from true tau ----
    val_ids = sorted(set(int(s) for s in sid))
    type_of = {}
    t0 = time.time()
    for s in iter_series("train", ids=val_ids):
        if s.has_break and s.tau_index is not None and 5 < s.tau_index < len(s.x_online) - 5:
            type_of[int(s.series_id)] = classify(
                s.x_hist.astype(np.float64), s.x_online.astype(np.float64), int(s.tau_index))
    print(f"typed {len(type_of)} broken series  ({time.time()-t0:.0f}s)")

    # per-row type: positives inherit series type; negatives = shared "neg"
    row_type = np.array(
        [type_of.get(int(s), "other") if yy == 1 else "neg"
         for s, yy in zip(sid, y)], dtype=object)
    types = ["mean", "var", "both", "subtle", "other"]
    counts = {T: int(np.sum((row_type == T))) for T in types}
    pos_series = {T: sum(1 for v in type_of.values() if v == T) for T in types}
    print("type cohort sizes (positive rows / broken series):")
    for T in types:
        print(f"  {T:7s} rows={counts[T]:7d}  series={pos_series[T]:5d}")

    neg = (y == 0)
    print(f"\n{'type':7s} " + " ".join(f"{m:>9s}" for m in members) + "   best-vs-shipped")
    overall_headroom = -9.9
    for T in types:
        if pos_series[T] < 30:
            print(f"  {T:7s}  (too few series, skip)")
            continue
        sub = neg | (row_type == T)        # type-T positives vs ALL negatives
        ts, ys = t[sub], y[sub]
        scores = {}
        for m, v in members.items():
            scores[m] = ts_auc_grouped(ts, ys, v[sub])
        ship = scores["shipped"]
        best_m = max((m for m in members if m != "shipped"), key=lambda m: scores[m])
        gap = scores[best_m] - ship
        overall_headroom = max(overall_headroom, gap)
        row = " ".join(f"{scores[m]:9.4f}" for m in members)
        print(f"  {T:7s} {row}   {best_m}{gap:+.4f}")

    # ---- ORACLE full-VAL lift: the number that actually matters (dilution + honest halves) ----
    # Route each POSITIVE row to its type's best member (z-scaled); negatives keep z(shipped).
    # This is an UPPER BOUND (knows the true type for free). Measure full + both honest halves.
    zmem = {m: zscore(v) for m, v in members.items()}
    best_for = {}
    for T in types:
        if pos_series[T] < 30:
            best_for[T] = "shipped"; continue
        sub = neg | (row_type == T)
        best_for[T] = max(members, key=lambda m: ts_auc_grouped(t[sub], y[sub], members[m][sub]))
    print("\noracle routing (type -> best member): " +
          ", ".join(f"{T}:{best_for[T]}" for T in types))

    oracle = zmem["shipped"].copy()
    for T in types:
        m = (row_type == T)
        if m.any():
            oracle[m] = zmem[best_for[T]][m]
    # also a NARROW oracle: only fix the mean cohort (the sole headroom), everything else = shipped
    narrow = zmem["shipped"].copy()
    mmask = (row_type == "mean")
    narrow[mmask] = zmem["gbt_mean"][mmask]

    u_ids = np.array(sorted(val_ids)); np.random.default_rng(1).shuffle(u_ids)
    hA = set(int(v) for v in u_ids[: len(u_ids) // 2])
    inA = np.array([int(x) in hA for x in sid])

    def trio(v):
        return (ts_auc_grouped(t, y, v),
                ts_auc_grouped(t[inA], y[inA], v[inA]),
                ts_auc_grouped(t[~inA], y[~inA], v[~inA]))
    sf, sA, sB = trio(zmem["shipped"])
    of, oA, oB = trio(oracle)
    nf, nA, nB = trio(narrow)
    print(f"\n{'':16s} {'full':>8s} {'halfA':>8s} {'halfB':>8s}")
    print(f"  shipped        {sf:8.4f} {sA:8.4f} {sB:8.4f}")
    print(f"  oracle(all)    {of:8.4f} {oA:8.4f} {oB:8.4f}   d {of-sf:+.4f}/{oA-sA:+.4f}/{oB-sB:+.4f}")
    print(f"  oracle(mean)   {nf:8.4f} {nA:8.4f} {nB:8.4f}   d {nf-sf:+.4f}/{nA-sA:+.4f}/{nB-sB:+.4f}")

    print("\n================ VERDICT ================")
    print(f"max single-member per-type advantage (per-cohort): {overall_headroom:+.4f}")
    print(f"ORACLE full-VAL lift (perfect type routing): {of-sf:+.4f} full / "
          f"{oA-sA:+.4f} A / {oB-sB:+.4f} B")
    oracle_robust = min(of - sf, oA - sA, oB - sB)
    if oracle_robust > 0.002:
        print(f"PREMISE HOLDS: even diluted + honest-halved, oracle routing lifts {oracle_robust:+.4f} "
              "robustly. The GRU is weak on mean-shift breaks; a serve-time gate that detects mean-"
              "shift conditions and downweights the GRU could recover part of this. -> build + gate a "
              "REAL (causal-feature) type gate; ship only if it lifts VAL full+both halves.")
    else:
        print(f"PREMISE FAILS AT THE BLEND: per-cohort gap exists but the ORACLE full-VAL lift is only "
              f"{oracle_robust:+.4f} (robust min) — diluted through the 68% subtle floor it vanishes. "
              "Even perfect routing can't move the metric; a real gate does worse. MoE DEAD.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
