"""Round 8 — OOS reduced-test check for the LightGBM RANK member (model_033_xendcg).

The rank_xendcg member PASSED the VAL honest-halves gate (both halves up, min
+0.0003 at small weight, on both integration paths). VAL halves overstate, so the
reduced local test (100 series, ids>=10000) is the TRUE OOS signal before shipping.

Deployable form: the rank booster outputs an unbounded relevance score, so we map
it into the blend with FROZEN constants computed on VAL (NOT on the test set):
  * TEST A (top-level z-blend):  (1-w)*zsh + w*zrk
        zsh = (shipped - SH_MU)/SH_SD,  zrk = (rank - RK_MU)/RK_SD
  * TEST B (rank as 5th base GBT): rank_as_gbt = (rank-RK_MU)/RK_SD*GBT_SD + GBT_MU
        gbt5 = (1-gw)*mean4 + gw*rank_as_gbt -> base5 -> 0.55*base5 + 0.45*GRU
All four constants are frozen from the VAL caches, so this is exactly what would
ship (one series at a time, no cross-series statistics).

Run: uv run python scripts/reduced_rank.py [model_033_xendcg]
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "submission")

import numpy as np

from sb.data import iter_series, load_test_targets
from sb.metric import ts_auc_grouped
import main as sub

MODEL_DIR = "submission/_model"


def _logit(p):
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def frozen_constants():
    """Compute the four z-constants on VAL (the deployable, test-blind form)."""
    base = np.load("features/val_base_logits.npz")
    gbt = base["gbt_logits"]                # (N,4)
    base_logit = base["base_logit"]
    gru_files = ("val_seq_logits.npz", "val_seq_logits_s1.npz", "val_seq_logits_s2.npz")
    gru = np.mean([np.load("features/" + f)["val_logit"] for f in gru_files], axis=0)
    shipped = 0.55 * base_logit + 0.45 * gru
    r = np.load("features/val_rank_logits.npz")["rank_score"].astype(np.float64)
    c = dict(
        SH_MU=float(shipped.mean()), SH_SD=float(shipped.std() + 1e-12),
        RK_MU=float(r.mean()), RK_SD=float(r.std() + 1e-12),
        GBT_MU=float(gbt.mean()), GBT_SD=float(gbt.mean(axis=1).std()),
    )
    return c


def main():
    import lightgbm as lgb

    model_id = sys.argv[1] if len(sys.argv) > 1 else "model_033_xendcg"
    rank_path = f"artifacts/models/{model_id}/lgbm_rank.txt"
    rank_b = lgb.Booster(model_file=rank_path)
    rank_keep = [sub._NAME_TO_COL[n] for n in rank_b.feature_name()]
    print(f"loaded rank member: {rank_path}  ({len(rank_keep)} feats)")

    C = frozen_constants()
    print("frozen VAL constants:", {k: round(v, 4) for k, v in C.items()})

    boosters, keeps = [], []
    for mi, (seed, rounds, version) in enumerate(sub.ENSEMBLE):
        boosters.append(lgb.Booster(
            model_file=os.path.join(MODEL_DIR, f"lgbm_{mi}.txt")))
        keeps.append(sub._keep_for(version))
    w_gbt = (1.0 - sub.W_LIN) / len(boosters)

    lg = np.load(os.path.join(MODEL_DIR, "logistic.npz"))
    l_keep, l_mean, l_scale = lg["keep"], lg["mean"], lg["scale"]
    l_coef, l_int = lg["coef"], float(lg["intercept"][0])

    grus = sub._load_grus(sub._NAME_TO_COL)
    calib_grus = [g for g in grus if getattr(g, "is_raw", False) is False]
    raw_grus = [g for g in grus if getattr(g, "is_raw", False)]
    w_seq = 1.0 / len(grus) if grus else 0.0

    det = sub.StreamingDetector()

    yt = load_test_targets().reset_index().sort_values(["id", "time"])
    yt["k"] = yt.groupby("id").cumcount()
    key = yt.set_index(["id", "k"])["target"]

    shipped_l, mean4_l, lin_l, seq_l, rank_l = [], [], [], [], []
    ys, ts = [], []

    for s in iter_series("test"):
        det.calibrate(np.asarray(s.x_hist, dtype=np.float64))
        for g in grus:
            g.reset()
        mu_h, sd_h = det.mu_h, det.sd_h
        feat_rows = []
        for i, x in enumerate(s.x_online):
            feats = det.update(float(x))
            acc = 0.0
            for b, keep in zip(boosters, keeps):
                p = float(b.predict(feats[keep].reshape(1, -1))[0])
                acc += _logit(p) * w_gbt
            mean4 = acc / (1.0 - sub.W_LIN)          # = mean of 4 GBT logits
            xs = np.clip((feats[l_keep] - l_mean) / l_scale, -8.0, 8.0)
            lin = float(np.dot(xs, l_coef)) + l_int
            base = acc + sub.W_LIN * lin
            if grus:
                seq = sum(g.step(feats) for g in calib_grus)
                if raw_grus:
                    rv = sub._raw_vec(float(x), mu_h, sd_h)
                    seq += sum(g.step(rv) for g in raw_grus)
                seq *= w_seq
                shipped = (1.0 - sub.W_GRU) * base + sub.W_GRU * seq
            else:
                seq = 0.0
                shipped = base
            shipped_l.append(shipped)
            mean4_l.append(mean4)
            lin_l.append(lin)
            seq_l.append(seq)
            feat_rows.append(feats[rank_keep])
            ys.append(int(key.loc[(s.series_id, i)]))
            ts.append(i)
        Fr = np.asarray(feat_rows, dtype=np.float64)
        rank_l.extend(rank_b.predict(Fr).tolist())

    shipped_l = np.asarray(shipped_l)
    mean4_l = np.asarray(mean4_l)
    lin_l = np.asarray(lin_l)
    seq_l = np.asarray(seq_l)
    rank_l = np.asarray(rank_l)
    ys = np.asarray(ys, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.int64)

    base_auc = ts_auc_grouped(ts, ys, shipped_l)
    rank_auc = ts_auc_grouped(ts, ys, rank_l)
    rc = np.corrcoef(shipped_l, rank_l)[0, 1]
    print(f"\nOOS rows={len(ys):,}  pos_rate={ys.mean():.4f}")
    print(f"shipped (base+GRU) OOS TS-AUC = {base_auc:.4f}")
    print(f"rank standalone    OOS TS-AUC = {rank_auc:.4f}   rankcorr={rc:.3f}\n")

    zsh = (shipped_l - C["SH_MU"]) / C["SH_SD"]
    zrk = (rank_l - C["RK_MU"]) / C["RK_SD"]
    print("TEST A — (1-w)*shipped + w*rank  (frozen z-space):")
    print(f"{'w':>6}{'OOS TS-AUC':>12}")
    for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        bl = (1.0 - w) * zsh + w * zrk
        a = ts_auc_grouped(ts, ys, bl)
        tag = "  <-- beats shipped" if a > base_auc + 1e-9 else ""
        print(f"{w:>6.2f}{a:>12.4f}{tag}")

    rank_as_gbt = zrk * C["GBT_SD"] + C["GBT_MU"]
    print("\nTEST B — rank as 5th base GBT (frozen scale):")
    print(f"{'gw':>6}{'OOS TS-AUC':>12}")
    for gw in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3):
        gbt5 = (1.0 - gw) * mean4_l + gw * rank_as_gbt
        base5 = (1.0 - sub.W_LIN) * gbt5 + sub.W_LIN * lin_l
        final5 = (1.0 - sub.W_GRU) * base5 + sub.W_GRU * seq_l
        a = ts_auc_grouped(ts, ys, final5)
        tag = "  <-- beats shipped" if a > base_auc + 1e-9 else ""
        print(f"{gw:>6.2f}{a:>12.4f}{tag}")


if __name__ == "__main__":
    main()
