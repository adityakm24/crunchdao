"""Failure-mode diagnostics for a trained model on the VAL split.

Answers:
 1. Per-step AUC curve (where over time are we weak?).
 2. AUC by break type / elapsed age / online position.
 3. Does the per-series score trajectory decay after detection?
    -> running-max / smoothing postprocessing tests.
 4. Score distribution of false-positive-prone no-break series.

Run: uv run python scripts/diagnose.py --model-id model_002
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import pandas as pd

from sb.features import FEATURE_NAMES
from sb.metric import ts_auc_grouped

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="model_002")
    ap.add_argument("--features", default="features/train_features.npz")
    args = ap.parse_args()

    booster = lgb.Booster(model_file=os.path.join("artifacts", "models", args.model_id, "lgbm_valsplit.txt"))
    used = booster.feature_name()

    d = np.load(args.features, allow_pickle=True)
    X, y, sid, t_online = d["X"], d["y"], d["series_id"], d["t_online"]
    names = [str(n) for n in d["feature_names"]]
    keep_idx = [names.index(n) for n in used]

    uniq = np.unique(sid)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    n_val = int(0.2 * len(uniq))
    val_ids = set(uniq[:n_val].tolist())
    va = np.isin(sid, list(val_ids))

    Xv, yv, sv, tv = X[va][:, keep_idx], y[va].astype(np.int64), sid[va], t_online[va]
    pred = booster.predict(Xv)
    base = ts_auc_grouped(tv, yv, pred)
    print(f"VAL TS-AUC ({args.model_id}): {base:.4f}   rows={len(yv):,}  series={n_val}")

    # ---- per-series trajectories ----
    idx = pd.read_parquet("Dataset/y_train_index.parquet")
    tau_map = idx["tau_index"].to_dict()

    order = np.lexsort((tv, sv))
    sv_o, tv_o, yv_o, p_o = sv[order], tv[order], yv[order], pred[order]

    # postprocessing variants computed per series
    p_max = np.empty_like(p_o)      # running max
    p_ewm = np.empty_like(p_o)      # EWMA(0.3) smoothing
    p_mix = np.empty_like(p_o)      # 0.5*current + 0.5*running max
    i, N = 0, len(sv_o)
    starts = []
    while i < N:
        j = i
        while j < N and sv_o[j] == sv_o[i]:
            j += 1
        starts.append((i, j))
        seg = p_o[i:j]
        p_max[i:j] = np.maximum.accumulate(seg)
        e = np.empty_like(seg)
        acc = seg[0]
        for k2 in range(len(seg)):
            acc = 0.7 * acc + 0.3 * seg[k2]
            e[k2] = acc
        p_ewm[i:j] = e
        p_mix[i:j] = 0.5 * seg + 0.5 * p_max[i:j]
        i = j

    for nm, pp in [("raw", p_o), ("running_max", p_max), ("ewma_0.3", p_ewm),
                   ("0.5raw+0.5max", p_mix)]:
        print(f"  postproc {nm:14s}: TS-AUC = {ts_auc_grouped(tv_o, yv_o, pp):.4f}")

    # ---- per-step AUC curve (coarse buckets) ----
    print("\nPer-step AUC (buckets of t):")
    buckets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 700), (700, 1000)]
    for lo, hi in buckets:
        m = (tv >= lo) & (tv < hi)
        if m.sum() == 0:
            continue
        auc = ts_auc_grouped(tv[m], yv[m], pred[m])
        # metric weight share of this bucket
        wsh = 0.0
        for t in range(lo, hi):
            mm = tv == t
            npos = int(yv[mm].sum()); nneg = int(mm.sum()) - npos
            wsh += npos * nneg
        print(f"  t in [{lo:4d},{hi:4d}) : AUC={auc:.4f}  weight_share~{wsh:.2e}")

    # ---- AUC by elapsed age of break (positives vs all negatives at same t) ----
    print("\nPositive detectability by elapsed steps since break (pos-only vs all negs):")
    tau_v = np.array([tau_map.get(int(s), -1) for s in sv])
    elapsed = np.where((tau_v >= 0) & (yv == 1), tv - tau_v, -1)
    for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 1000)]:
        m_pos = (elapsed >= lo) & (elapsed < hi)
        m = m_pos | (yv == 0)
        if m_pos.sum() == 0:
            continue
        auc = ts_auc_grouped(tv[m], yv[m], pred[m])
        print(f"  elapsed [{lo:4d},{hi:4d}) : AUC={auc:.4f}  n_pos_rows={m_pos.sum():,}")

    # ---- break-type taxonomy AUC ----
    # classify each broken val series by its dominant post-break deviation
    sys.path.insert(0, "scripts")
    from sb.data import iter_series  # noqa

    print("\nLoading raw series for break-type classification (val ids only)...")
    types = {}
    for s in iter_series("train", ids=[int(v) for v in val_ids]):
        if not s.has_break:
            continue
        x_h, x_o, tau = s.x_hist, s.x_online, s.tau_index
        post = x_o[tau:]
        if len(post) < 20:
            types[s.series_id] = "short_post"
            continue
        mu, sd = x_h.mean(), x_h.std() + 1e-12
        z = (post - mu) / sd

        def acf1(v):
            v = v - v.mean()
            d = np.dot(v, v)
            return float(np.dot(v[:-1], v[1:]) / d) if d > 0 else 0.0

        dev = dict(
            mean=abs(float(z.mean())) / 0.25,
            var=abs(float(np.log(post.var() / (sd * sd) + 1e-12))) / 0.30,
            acf=abs(acf1(post) - acf1(x_h)) / 0.12,
        )
        best = max(dev, key=dev.get)
        types[s.series_id] = best if dev[best] >= 1.0 else "subtle"

    tlab = np.array([types.get(int(s), "none") for s in sv])
    for bt in ["mean", "var", "acf", "subtle", "short_post"]:
        m = (tlab == bt) & (yv == 1)
        mm = m | (yv == 0)
        if m.sum() == 0:
            continue
        auc = ts_auc_grouped(tv[mm], yv[mm], pred[mm])
        n_series = len({int(s) for s in sv[m]})
        print(f"  break-type {bt:10s}: AUC={auc:.4f}  ({n_series} series, {m.sum():,} pos rows)")

    # ---- no-break false-alarm tail ----
    neg_sids = [int(s) for s in val_ids if tau_map.get(int(s), -1) < 0]
    neg_mask = np.isin(sv, neg_sids)
    df = pd.DataFrame(dict(sid=sv[neg_mask], p=pred[neg_mask]))
    worst = df.groupby("sid")["p"].max().sort_values(ascending=False)
    print(f"\nTop-10 false-alarm no-break series (max score): \n{worst.head(10).round(3).to_string()}")
    print(f"\nno-break series mean(max score) = {worst.mean():.3f}")


if __name__ == "__main__":
    main()
