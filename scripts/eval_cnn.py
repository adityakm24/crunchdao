"""Evaluate the saved CNN member (model_031_cnn/cnn.npz) on VAL: standalone
TS-AUC, rank-corr vs the shipped stack, and blend lift onto the shipped
base+GRU (0.6160) over a weight grid with honest halves. Reuses the trainer's
numpy_cnn_forward so the served (float64) path is exactly what we measure.

Run: uv run python scripts/eval_cnn.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np

from sb.metric import ts_auc_grouped
from train_cnn import numpy_cnn_forward, val_mask

RAW = "features/train_raw_stream.npz"
GRU_FILES = ("val_seq_logits.npz", "val_seq_logits_s1.npz", "val_seq_logits_s2.npz")


def half_mask_A(sid_val):
    uniq = np.unique(sid_val)
    rng = np.random.default_rng(1)
    rng.shuffle(uniq)
    a = set(uniq[: len(uniq) // 2].tolist())
    return np.isin(sid_val, list(a))


def main():
    d = np.load(RAW, allow_pickle=True)
    cnn = np.load("artifacts/models/model_031_cnn/cnn.npz", allow_pickle=True)
    p = {k: cnn[k] for k in cnn.files}
    in_ch = int(p["in_channels"]) if "in_channels" in p else 1
    z_all = d["X"][:, :in_ch].astype(np.float64)  # (N, in_ch)
    y, sid, t = d["y"], d["series_id"], d["t_online"]
    va = val_mask(sid)
    uniq, start = np.unique(sid, return_index=True)
    bounds = list(start) + [len(sid)]
    series = [(int(start[i]), int(bounds[i + 1])) for i in range(len(uniq))]
    val_idx = [i for i in range(len(series)) if bool(va[series[i][0]])]

    val_pred = np.zeros(len(sid), dtype=np.float64)
    for i in val_idx:
        s0, s1 = series[i]
        val_pred[s0:s1] = numpy_cnn_forward(z_all[s0:s1], p)
    cnn_logit = val_pred[va]
    np.savez("features/val_cnn_logits.npz", val_logit=cnn_logit)

    yv, tv, sv = y[va].astype(np.int64), t[va], sid[va]
    base = np.load("features/val_base_logits.npz")
    assert np.array_equal(base["series_id"], sv) and np.array_equal(base["t_online"], tv)
    gru = np.mean([np.load("features/" + f)["val_logit"] for f in GRU_FILES], axis=0)
    shipped = 0.55 * base["base_logit"] + 0.45 * gru
    ship_full = ts_auc_grouped(tv, yv, shipped)

    cnn_sa = ts_auc_grouped(tv, yv, cnn_logit)
    rc = np.corrcoef(shipped, cnn_logit)[0, 1]
    print(f"CNN standalone VAL TS-AUC = {cnn_sa:.4f}   rankcorr(shipped) = {rc:.3f}")
    print(f"shipped VAL TS-AUC        = {ship_full:.4f}\n")

    halfA = half_mask_A(sv)
    print(f"{'Wc':>6}{'full':>9}{'halfA':>9}{'halfB':>9}{'min':>9}")
    sa = ts_auc_grouped(tv[halfA], yv[halfA], shipped[halfA])
    sb = ts_auc_grouped(tv[~halfA], yv[~halfA], shipped[~halfA])
    print(f"{'ship':>6}{ship_full:>9.4f}{sa:>9.4f}{sb:>9.4f}{min(sa, sb):>9.4f}")
    for wc in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
        bl = (1 - wc) * shipped + wc * cnn_logit
        full = ts_auc_grouped(tv, yv, bl)
        aA = ts_auc_grouped(tv[halfA], yv[halfA], bl[halfA])
        aB = ts_auc_grouped(tv[~halfA], yv[~halfA], bl[~halfA])
        tag = "  <--" if full > ship_full else ""
        print(f"{wc:>6.2f}{full:>9.4f}{aA:>9.4f}{aB:>9.4f}{min(aA, aB):>9.4f}{tag}")


if __name__ == "__main__":
    main()
