"""Compare base-only vs base+GRU on the reduced local test WITHOUT retraining.

Reuses the already-trained submission/_model boosters + logistic and the GRUs
embedded in submission/main.py. Streams the reduced test once, captures both the
base logit and the base+GRU logit per point, and reports both TS-AUCs (+ a few
W_GRU settings). This isolates whether the neural member helps on the held-out
reduced test (100 series, ids>=10000 the GRU never saw), separate from VAL.

Run: uv run python scripts/reduced_ab.py
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


def main() -> None:
    import lightgbm as lgb

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
    w_seq = 1.0 / len(grus)

    det = sub.StreamingDetector()

    base_logits, seq_logits = [], []
    ys, ts, sids = [], [], []

    # ground truth aligned by (id, online step)
    yt = load_test_targets().reset_index().sort_values(["id", "time"])
    yt["k"] = yt.groupby("id").cumcount()
    key = yt.set_index(["id", "k"])["target"]

    for s in iter_series("test"):
        det.calibrate(np.asarray(s.x_hist, dtype=np.float64))
        for g in grus:
            g.reset()
        mu_h, sd_h = det.mu_h, det.sd_h
        for i, x in enumerate(s.x_online):
            feats = det.update(float(x))
            acc = 0.0
            for b, keep in zip(boosters, keeps):
                p = float(b.predict(feats[keep].reshape(1, -1))[0])
                p = min(max(p, 1e-7), 1.0 - 1e-7)
                acc += math.log(p / (1.0 - p)) * w_gbt
            xs = np.clip((feats[l_keep] - l_mean) / l_scale, -8.0, 8.0)
            lin = float(np.dot(xs, l_coef)) + l_int
            base = acc + sub.W_LIN * lin
            seq = sum(g.step(feats) for g in calib_grus)
            if raw_grus:
                rv = sub._raw_vec(float(x), mu_h, sd_h)
                seq += sum(g.step(rv) for g in raw_grus)
            seq *= w_seq
            base_logits.append(base)
            seq_logits.append(seq)
            ys.append(int(key.loc[(s.series_id, i)]))
            ts.append(i)
            sids.append(s.series_id)

    base_logits = np.array(base_logits)
    seq_logits = np.array(seq_logits)
    ys = np.array(ys, dtype=np.int64)
    ts = np.array(ts, dtype=np.int64)

    base_auc = ts_auc_grouped(ts, ys, base_logits)
    seq_auc = ts_auc_grouped(ts, ys, seq_logits)
    print(f"reduced test: {len(set(sids))} series, {len(ys)} points")
    print(f"  BASE only     TS-AUC = {base_auc:.4f}")
    print(f"  GRU member    TS-AUC = {seq_auc:.4f}  (standalone)")
    for w in (0.0, 0.15, 0.25, 0.40, 0.50):
        blend = (1.0 - w) * base_logits + w * seq_logits
        print(f"  W_GRU={w:.2f}     TS-AUC = {ts_auc_grouped(ts, ys, blend):.4f}")


if __name__ == "__main__":
    main()
