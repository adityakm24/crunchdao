"""Probe: is a TabPFN member a decorrelated-AND-strong signal, or does it hit the
same per-series saturation wall as every other model class?

We fit TabPFN's in-context "training" on a balanced subsample of TRAIN-split rows
(seed-42 split, same as eval_base.py) over the v4 calibrated features, then predict
on all rows of a sample of VAL-split series (whole series so the per-online-step
cross-sectional groups used by TS-AUC stay intact). We report:
  - TabPFN standalone TS-AUC on the probe subset
  - base ensemble TS-AUC on the SAME subset (apples-to-apples)
  - Spearman rank-corr(TabPFN, base) -> decorrelation
  - blend TS-AUC at a few weights (does it lift the base on this subset?)

This only measures SIGNAL. Deployment/determinism is a separate problem we only
solve if the signal clears the bar.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sb.metric import ts_auc_grouped

SPLIT_SEED = 42
FEATS = "features/train_features_v4.npz"
DROP = {"t", "log_n_hist"}  # documented time-trap features


def val_mask(sid: np.ndarray) -> np.ndarray:
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids)), set(uniq[int(0.2 * len(uniq)):].tolist())


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=16000, help="TabPFN context rows")
    ap.add_argument("--val-series", type=int, default=300, help="# VAL series to score")
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    print(f"[load] {FEATS} (mmap)")
    d = np.load(FEATS, mmap_mode="r")
    names = [str(n) for n in d["feature_names"]]
    keep_cols = [i for i, n in enumerate(names) if n not in DROP]
    kept = [names[i] for i in keep_cols]
    print(f"[feats] {len(names)} -> {len(keep_cols)} (dropped {sorted(DROP)})")

    sid = np.asarray(d["series_id"])
    y = np.asarray(d["y"]).astype(np.int64)
    t = np.asarray(d["t_online"])

    vmask, train_ids = val_mask(sid)
    tmask = ~vmask
    print(f"[split] train rows {tmask.sum():,} | val rows {vmask.sum():,}")

    # --- balanced TRAIN context subsample ---
    rng = np.random.default_rng(0)
    tr_idx = np.where(tmask)[0]
    tr_y = y[tr_idx]
    pos = tr_idx[tr_y == 1]
    neg = tr_idx[tr_y == 0]
    half = args.context // 2
    ctx_idx = np.concatenate([
        rng.choice(pos, size=min(half, len(pos)), replace=False),
        rng.choice(neg, size=min(half, len(neg)), replace=False),
    ])
    ctx_idx.sort()
    Xc = np.asarray(d["X"][ctx_idx][:, keep_cols], dtype=np.float32)
    yc = y[ctx_idx]
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[context] {Xc.shape} pos_rate={yc.mean():.3f}")

    # --- VAL probe rows: whole series so per-step groups stay intact ---
    val_series = np.unique(sid[vmask])
    rng2 = np.random.default_rng(1)
    rng2.shuffle(val_series)
    pick = set(val_series[: args.val_series].tolist())
    pmask = vmask & np.isin(sid, list(pick))
    p_idx = np.where(pmask)[0]
    Xv = np.asarray(d["X"][p_idx][:, keep_cols], dtype=np.float32)
    Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
    yv = y[p_idx]
    tv = t[p_idx]
    print(f"[probe] {len(p_idx):,} rows over {len(pick)} val series")

    # --- base ensemble logits on the SAME rows (align by (sid,t)) ---
    base = np.load("features/val_base_logits.npz")
    key = base["series_id"].astype(np.int64) * 100000 + base["t_online"].astype(np.int64)
    pk = sid[p_idx].astype(np.int64) * 100000 + tv.astype(np.int64)
    order = np.argsort(key)
    pos_in = np.searchsorted(key[order], pk)
    base_logit_sub = base["base_logit"][order][pos_in]
    assert np.array_equal(base["y"][order][pos_in], yv), "alignment mismatch"

    # --- TabPFN ---
    from tabpfn import TabPFNClassifier
    print(f"[tabpfn] fit context, predict {len(p_idx):,} rows ...")
    t0 = time.time()
    clf = TabPFNClassifier(n_estimators=args.n_estimators, device=args.device,
                           ignore_pretraining_limits=True, random_state=0)
    clf.fit(Xc, yc)
    proba = clf.predict_proba(Xv)[:, 1]
    print(f"[tabpfn] done in {time.time()-t0:.0f}s")
    tab_logit = _logit(proba)

    # --- metrics ---
    auc_tab = ts_auc_grouped(tv, yv, tab_logit)
    auc_base = ts_auc_grouped(tv, yv, base_logit_sub)
    rc = spearmanr(tab_logit, base_logit_sub).correlation
    print("\n=== RESULTS (probe subset) ===")
    print(f"  base   TS-AUC = {auc_base:.4f}")
    print(f"  TabPFN TS-AUC = {auc_tab:.4f}")
    print(f"  rank-corr(TabPFN, base) = {rc:.3f}")
    bl = _logit(np.clip((proba), 1e-7, 1 - 1e-7))
    bs = base_logit_sub
    # standardize to comparable scale then blend
    blz = (bl - bl.mean()) / (bl.std() + 1e-9)
    bsz = (bs - bs.mean()) / (bs.std() + 1e-9)
    print("  blend (z-scaled logits):")
    for w in (0.1, 0.2, 0.3, 0.4, 0.5):
        mix = (1 - w) * bsz + w * blz
        print(f"    w_tab={w:.2f}  TS-AUC={ts_auc_grouped(tv, yv, mix):.4f}")


if __name__ == "__main__":
    main()
