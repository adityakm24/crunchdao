"""Quantify the round-4 GRU serve bug: bugged StreamingGRU vs the correct forward.

The cached features/val_seq_logits.npz for model_020 was computed with the
CORRECT numpy_gru_forward. Here we recompute model_020's VAL logit with the OLD
(bugged) recurrence -- n-gate folding b_hn into the input side -- and compare
standalone + blended VAL TS-AUC, so we know the fix's direction/magnitude before
rebuilding round 5 on it.

Run: uv run python scripts/bug_impact.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from sb.metric import ts_auc_grouped

V4 = "features/train_features_v4.npz"
SPLIT_SEED = 42
DROP = {"t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
        "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2"}
GRU_DIR = "artifacts/models/model_020_gru"


def val_mask(sid):
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids))


def bugged_forward(X, p):
    """The OLD shipped recurrence: b_hn folded into the input side (wrong)."""
    Wih, Whh = p["Wih"], p["Whh"]
    bih = np.asarray(p["bih"]) + np.asarray(p["bhh"])   # combined (the bug)
    Wo, bo = p["Wo"], float(p["bo"])
    mean, scale = p["mean"], p["scale"]
    H = Whh.shape[1]
    Xs = np.clip((X - mean) / scale, -8.0, 8.0)
    np.nan_to_num(Xs, copy=False)
    h = np.zeros(H, dtype=np.float64)
    out = np.empty(len(Xs), dtype=np.float64)
    for i in range(len(Xs)):
        g = Wih @ Xs[i] + bih
        gh = Whh @ h
        r = 1.0 / (1.0 + np.exp(-(g[:H] + gh[:H])))
        z = 1.0 / (1.0 + np.exp(-(g[H:2 * H] + gh[H:2 * H])))
        n = np.tanh(g[2 * H:] + r * gh[2 * H:])   # bug: bhn added unconditionally
        h = (1.0 - z) * n + z * h
        out[i] = Wo @ h + bo
    return out


def main() -> None:
    d = np.load(V4, allow_pickle=True)
    X, sid = d["X"], d["series_id"]
    names = [str(n) for n in d["feature_names"]]
    keep = [i for i, n in enumerate(names) if n not in DROP]
    va = val_mask(sid)

    g = np.load(f"{GRU_DIR}/gru.npz", allow_pickle=True)
    p = {k: g[k] for k in g.files}

    base = np.load("features/val_base_logits.npz")
    yv, tv = base["y"], base["t_online"]
    base_logit = base["base_logit"]
    correct = np.load("features/val_seq_logits.npz")["val_logit"]

    uniq, start = np.unique(sid, return_index=True)
    bounds = list(start) + [len(sid)]
    bug = np.zeros(len(sid), dtype=np.float64)
    for i in range(len(uniq)):
        s0, s1 = int(start[i]), int(bounds[i + 1])
        if va[s0]:
            bug[s0:s1] = bugged_forward(X[s0:s1][:, keep].astype(np.float64), p)
    bug_v = bug[va]

    print(f"model_020 GRU standalone VAL TS-AUC:")
    print(f"  bugged  = {ts_auc_grouped(tv, yv, bug_v):.5f}")
    print(f"  correct = {ts_auc_grouped(tv, yv, correct):.5f}")
    print(f"max|bug-correct| logit = {np.abs(bug_v - correct).max():.4e}")
    for w in (0.25, 0.40):
        b = (1 - w) * base_logit + w * bug_v
        c = (1 - w) * base_logit + w * correct
        print(f"  blend W={w}: bugged={ts_auc_grouped(tv, yv, b):.5f}  "
              f"correct={ts_auc_grouped(tv, yv, c):.5f}")


if __name__ == "__main__":
    main()
