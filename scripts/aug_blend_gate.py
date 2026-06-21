"""Round 11 — blend-level augmentation gate (local, no push).

The single-GBT A/B showed faithful synth lifts the GBT (full +0.0052, halfB +0.0118).
The decisive question (the stepnorm lesson): does it SURVIVE the full blend, or does
the GRU already capture it? This reconstructs the round-10 shipped logit on real VAL
and splices in augmented member logits, measuring TS-AUC on full + honest halves.

Members (round-10 weights): mean4 GBT + rank(0.10) + logistic(0.10) inside base,
then base(0.55) + 3-GRU-mean(0.45). VAL stays 100% real.

  --aug-gru / --ctrl-gru : single-GRU val_logit npz (replaces ONE slot in the
        3-GRU mean, the others stay shipped-real) — isolates GRU augmentation.
  --aug-gbt / --ctrl-gbt : single-GBT VAL logit npz (replaces mean4) — isolates
        GBT augmentation at the blend.

Run: uv run python scripts/aug_blend_gate.py --aug-gru features/val_gru_aug.npz \
        --ctrl-gru features/val_gru_ctrl.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sb.metric import ts_auc_grouped       # noqa: E402

SEED = 42
RANK_GW, W_LIN, W_GRU = 0.10, 0.10, 0.45


def half_mask(sid_val):
    uniq = np.unique(sid_val); rng = np.random.default_rng(1); rng.shuffle(uniq)
    return np.isin(sid_val, list(set(uniq[: len(uniq) // 2].tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-gru", default="")
    ap.add_argument("--ctrl-gru", default="")
    ap.add_argument("--aug-gbt", default="")
    ap.add_argument("--ctrl-gbt", default="")
    ap.add_argument("--gru-slot", type=int, default=2, help="which of 3 GRUs to swap (0/1/2)")
    args = ap.parse_args()

    base = np.load("features/val_base_logits.npz")
    sid = base["series_id"].astype(np.int64)
    t = base["t_online"].astype(np.int64)
    y = base["y"].astype(np.int64)
    gbt = base["gbt_logits"].astype(np.float64)
    log_logit = base["log_logit"].astype(np.float64)
    mean4 = gbt.mean(axis=1)

    rk = np.load("features/val_rank_logits.npz")
    rk_key = {(int(s), int(tt)): float(v) for s, tt, v in
              zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank_score = np.array([rk_key[(int(s), int(tt))] for s, tt in zip(sid, t)])

    grus = [np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"].astype(np.float64)
            for m in ("020", "021", "022")]

    def blend(mean4_v, gru_list):
        rag = (rank_score - rank_score.mean()) / (rank_score.std() + 1e-12)
        rag = rag * mean4_v.std() + mean4_v.mean()
        gbt5 = (1 - RANK_GW) * mean4_v + RANK_GW * rag
        base_logit = (1 - W_LIN) * gbt5 + W_LIN * log_logit
        gru = np.mean(gru_list, axis=0)
        return (1 - W_GRU) * base_logit + W_GRU * gru

    hA = half_mask(sid)

    def report(tag, sh):
        f = ts_auc_grouped(t, y, sh)
        a = ts_auc_grouped(t[hA], y[hA], sh[hA])
        b = ts_auc_grouped(t[~hA], y[~hA], sh[~hA])
        print(f"  [{tag:16s}] full={f:.4f}  halfA={a:.4f}  halfB={b:.4f}  min={min(a,b):.4f}")
        return np.array([f, a, b])

    print("shipped baseline (all real):")
    r_base = report("REAL shipped", blend(mean4, grus))

    def load_logit(path):
        d = np.load(path, allow_pickle=True)
        key = "val_logit" if "val_logit" in d.files else "logit"
        return d[key].astype(np.float64)

    if args.ctrl_gru and args.aug_gru:
        print(f"\nGRU augmentation (swap slot {args.gru_slot} of 3-GRU mean):")
        gctrl = grus[:]; gctrl[args.gru_slot] = load_logit(args.ctrl_gru)
        gaug = grus[:];  gaug[args.gru_slot] = load_logit(args.aug_gru)
        rc = report("ctrl-GRU blend", blend(mean4, gctrl))
        ra = report("aug-GRU blend", blend(mean4, gaug))
        d = ra - rc
        print(f"  Δ(aug-ctrl) full={d[0]:+.4f} halfA={d[1]:+.4f} halfB={d[2]:+.4f}"
              f"  -> {'LIFTS blend (full+both)' if (d>0).all() else 'no robust blend gain'}")

    if args.ctrl_gbt and args.aug_gbt:
        print("\nGBT augmentation (replace mean4 with single GBT):")
        rc = report("ctrl-GBT blend", blend(load_logit(args.ctrl_gbt), grus))
        ra = report("aug-GBT blend", blend(load_logit(args.aug_gbt), grus))
        d = ra - rc
        print(f"  Δ(aug-ctrl) full={d[0]:+.4f} halfA={d[1]:+.4f} halfB={d[2]:+.4f}"
              f"  -> {'LIFTS blend (full+both)' if (d>0).all() else 'no robust blend gain'}")


if __name__ == "__main__":
    main()
