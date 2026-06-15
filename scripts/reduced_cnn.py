"""OOS reduced-test gate for the CNN member (pure numpy, no torch).

Streams the reduced local test (100 series, ids>=10000 the nets never saw),
reconstructs the SHIPPED logit (base + 3 GRU exactly as submission/main.py),
and in parallel runs the trained CNN (model_031_cnn/cnn.npz) over each series'
raw standardized z-stream (numpy_cnn_forward, float64). Reports shipped TS-AUC
vs (1-Wc)*shipped + Wc*cnn over a weight grid. OOS is the decisive gate:
ship the CNN only if it beats shipped here AND on VAL.

Run: uv run python scripts/reduced_cnn.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "submission")
sys.path.insert(0, "scripts")

import numpy as np

from sb.data import iter_series, load_test_targets
from sb.metric import ts_auc_grouped
from train_cnn import numpy_cnn_forward
import main as sub

MODEL_DIR = "submission/_model"
CLIP = 8.0


def main() -> None:
    import lightgbm as lgb

    boosters, keeps = [], []
    for mi, (seed, rounds, version) in enumerate(sub.ENSEMBLE):
        boosters.append(lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lgbm_{mi}.txt")))
        keeps.append(sub._keep_for(version))
    w_gbt = (1.0 - sub.W_LIN) / len(boosters)

    lg = np.load(os.path.join(MODEL_DIR, "logistic.npz"))
    l_keep, l_mean, l_scale = lg["keep"], lg["mean"], lg["scale"]
    l_coef, l_int = lg["coef"], float(lg["intercept"][0])

    grus = sub._load_grus(sub._NAME_TO_COL)
    calib_grus = [g for g in grus if getattr(g, "is_raw", False) is False]
    raw_grus = [g for g in grus if getattr(g, "is_raw", False)]
    w_seq = 1.0 / len(grus)
    W_GRU = sub.W_GRU

    cnn = np.load("artifacts/models/model_031_cnn/cnn.npz", allow_pickle=True)
    p = {k: cnn[k] for k in cnn.files}
    in_ch = int(p["in_channels"]) if "in_channels" in p else 1

    det = sub.StreamingDetector()
    shipped_logits, cnn_logits = [], []
    ys, ts, sids = [], [], []

    yt = load_test_targets().reset_index().sort_values(["id", "time"])
    yt["k"] = yt.groupby("id").cumcount()
    key = yt.set_index(["id", "k"])["target"]

    for s in iter_series("test"):
        x_hist = np.asarray(s.x_hist, dtype=np.float64)
        det.calibrate(x_hist)
        for g in grus:
            g.reset()
        mu_h, sd_h = det.mu_h, det.sd_h
        xo = np.asarray(s.x_online, dtype=np.float64)

        # CNN over the whole online z-stream for this series (causal -> per-step)
        z = (xo - mu_h) / (sd_h if sd_h > 0 else 1.0)
        if in_ch == 1:
            zc = z
        else:
            cols = [z, z * z, np.abs(z)][:in_ch]
            zc = np.stack(cols, axis=1)  # (T, in_ch)
        cnn_series = numpy_cnn_forward(zc, p)  # (T,)

        for i, x in enumerate(s.x_online):
            feats = det.update(float(x))
            acc = 0.0
            for b, keep in zip(boosters, keeps):
                pr = float(b.predict(feats[keep].reshape(1, -1))[0])
                pr = min(max(pr, 1e-7), 1.0 - 1e-7)
                acc += math.log(pr / (1.0 - pr)) * w_gbt
            xs = np.clip((feats[l_keep] - l_mean) / l_scale, -8.0, 8.0)
            lin = float(np.dot(xs, l_coef)) + l_int
            base = acc + sub.W_LIN * lin
            seq = sum(g.step(feats) for g in calib_grus)
            if raw_grus:
                rv = sub._raw_vec(float(x), mu_h, sd_h)
                seq += sum(g.step(rv) for g in raw_grus)
            seq *= w_seq
            shipped = (1.0 - W_GRU) * base + W_GRU * seq
            shipped_logits.append(shipped)
            cnn_logits.append(float(cnn_series[i]))
            ys.append(int(key.loc[(s.series_id, i)]))
            ts.append(i)
            sids.append(s.series_id)

    shipped_logits = np.array(shipped_logits)
    cnn_logits = np.array(cnn_logits)
    ys = np.array(ys, dtype=np.int64)
    ts = np.array(ts, dtype=np.int64)

    ship_auc = ts_auc_grouped(ts, ys, shipped_logits)
    cnn_auc = ts_auc_grouped(ts, ys, cnn_logits)
    rc = np.corrcoef(shipped_logits, cnn_logits)[0, 1]
    print(f"reduced test: {len(set(sids))} series, {len(ys)} points  (in_ch={in_ch})")
    print(f"  SHIPPED        TS-AUC = {ship_auc:.4f}")
    print(f"  CNN standalone TS-AUC = {cnn_auc:.4f}   rankcorr(shipped) = {rc:.3f}")
    for w in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        blend = (1.0 - w) * shipped_logits + w * cnn_logits
        a = ts_auc_grouped(ts, ys, blend)
        tag = "  <--" if a > ship_auc else ""
        print(f"  Wc={w:.2f}       TS-AUC = {a:.4f}{tag}")


if __name__ == "__main__":
    main()
