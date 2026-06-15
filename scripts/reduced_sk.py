"""Round 6 — OOS reduced-test check for a saved sklearn diversity member.

Loads a joblib-saved ET/RF member (trained by diverse_members.py on the VAL-split
TRAIN rows) and the shipped base (submission/_model boosters + logistic), streams
the reduced local test (100 series, ids>=10000), and reports base-only vs
base+member TS-AUC across a weight grid. The reduced test is the TRUE OOS signal
(VAL honest halves overstate). A member ships only if it lifts OOS here AND VAL.

The member must be fed the SAME 162-col v4 feature vector the matrix used, in the
same column order (FEATURE_NAMES). We rebuild that vector with the submission's
StreamingDetector (identical feature code), so train/serve parity holds.

Run: uv run python scripts/reduced_sk.py artifacts/models/model_030_extratrees/model.joblib
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "submission")

import joblib
import numpy as np

from sb.data import iter_series, load_test_targets
from sb.metric import ts_auc_grouped
import main as sub

MODEL_DIR = "submission/_model"


def _logit(p):
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def main():
    import lightgbm as lgb

    member_path = sys.argv[1] if len(sys.argv) > 1 else \
        "artifacts/models/model_030_extratrees/model.joblib"
    clf = joblib.load(member_path)
    print(f"loaded member: {member_path}")

    boosters, keeps = [], []
    for mi, (seed, rounds, version) in enumerate(sub.ENSEMBLE):
        boosters.append(lgb.Booster(
            model_file=os.path.join(MODEL_DIR, f"lgbm_{mi}.txt")))
        keeps.append(sub._keep_for(version))
    w_gbt = (1.0 - sub.W_LIN) / len(boosters)

    lg = np.load(os.path.join(MODEL_DIR, "logistic.npz"))
    l_keep, l_mean, l_scale = lg["keep"], lg["mean"], lg["scale"]
    l_coef, l_int = lg["coef"], float(lg["intercept"][0])

    # neural member (kept in the served blend); base here = GBT+logistic+GRU
    grus = sub._load_grus(sub._NAME_TO_COL)
    calib_grus = [g for g in grus if getattr(g, "is_raw", False) is False]
    raw_grus = [g for g in grus if getattr(g, "is_raw", False)]
    w_seq = 1.0 / len(grus) if grus else 0.0

    det = sub.StreamingDetector()

    yt = load_test_targets().reset_index().sort_values(["id", "time"])
    yt["k"] = yt.groupby("id").cumcount()
    key = yt.set_index(["id", "k"])["target"]

    shipped_logits, member_logits = [], []
    ys, ts = [], []
    feat_buf = []  # batch member prediction per series for speed

    for s in iter_series("test"):
        det.calibrate(np.asarray(s.x_hist, dtype=np.float64))
        for g in grus:
            g.reset()
        mu_h, sd_h = det.mu_h, det.sd_h
        rows_feats = []
        for i, x in enumerate(s.x_online):
            feats = det.update(float(x))
            acc = 0.0
            for b, keep in zip(boosters, keeps):
                p = float(b.predict(feats[keep].reshape(1, -1))[0])
                acc += _logit(p) * w_gbt
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
                shipped = base
            shipped_logits.append(shipped)
            rows_feats.append(feats.copy())
            ys.append(int(key.loc[(s.series_id, i)]))
            ts.append(i)
        # batch member proba for this series
        Fm = np.nan_to_num(np.array(rows_feats, dtype=np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        pm = clf.predict_proba(Fm)[:, 1]
        member_logits.extend(_logit(float(p)) for p in pm)

    shipped_logits = np.array(shipped_logits)
    member_logits = np.array(member_logits)
    ys = np.array(ys, dtype=np.int64)
    ts = np.array(ts, dtype=np.int64)

    base_auc = ts_auc_grouped(ts, ys, shipped_logits)
    mem_auc = ts_auc_grouped(ts, ys, member_logits)
    rc = np.corrcoef(shipped_logits, member_logits)[0, 1]
    print(f"\nshipped (base+GRU) OOS TS-AUC = {base_auc:.4f}")
    print(f"member standalone   OOS TS-AUC = {mem_auc:.4f}   rankcorr={rc:.3f}\n")
    print(f"{'w':>6}{'OOS TS-AUC':>12}")
    for w in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        bl = (1.0 - w) * shipped_logits + w * member_logits
        a = ts_auc_grouped(ts, ys, bl)
        tag = "  <-- beats shipped" if a > base_auc else ""
        print(f"{w:>6.2f}{a:>12.4f}{tag}")


if __name__ == "__main__":
    main()
