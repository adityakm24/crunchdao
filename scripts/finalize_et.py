"""Round 6 — finalize the ExtraTrees diversity member and test the FULL stack.

diverse_members.py proved ExtraTrees (bagging) decorrelates from the boosting
ensemble (rankcorr 0.952) and lifts the BASE (0.6041 -> 0.6050 @ w0.2, both honest
halves up). RandomForest was weaker -> rejected. This script:

  1. trains the chosen ET (n_est=300, leaf=200, sqrt, bootstrap=False, seed=42)
     on the v4 VAL-split TRAIN rows;
  2. saves it to artifacts/models/model_030_extratrees/model.joblib (+ meta);
  3. caches its VAL logits to features/val_et_logits.npz (base-cache row order);
  4. reconstructs the SHIPPED stack (0.55*base + 0.45*mean(3 GRU) = 0.6160) and
     measures shipped + We*ET across a weight grid, full VAL + honest halves.

A positive, both-halves lift here (plus an OOS reduced win via reduced_sk.py) is
the ship gate. Run: uv run python scripts/finalize_et.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "src")

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from sb.metric import ts_auc_grouped

V4 = "features/train_features_v4.npz"
SPLIT_SEED = 42
W_LIN = 0.2
W_GRU = 0.45
OUT_DIR = "artifacts/models/model_030_extratrees"
GRU_FILES = ["val_seq_logits.npz", "val_seq_logits_s1.npz", "val_seq_logits_s2.npz"]
ET_PARAMS = dict(n_estimators=300, max_features="sqrt", min_samples_leaf=200,
                 bootstrap=False, n_jobs=-1, random_state=42)


def val_mask(sid):
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids))


def half_mask_A(sid_val):
    uniq = np.unique(sid_val)
    rng = np.random.default_rng(1)
    rng.shuffle(uniq)
    a = set(uniq[: len(uniq) // 2].tolist())
    return np.isin(sid_val, list(a))


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main():
    print("loading v4 matrix ...")
    d = np.load(V4, allow_pickle=True)
    X, y, sid, t = d["X"], d["y"], d["series_id"], d["t_online"]
    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    va = val_mask(sid)
    tr = ~va
    Xtr, ytr = X[tr], y[tr].astype(np.int64)
    Xv, yv, tv, sv = X[va], y[va].astype(np.int64), t[va], sid[va]
    print(f"  train rows={tr.sum():,}  val rows={va.sum():,}")

    cache = np.load("features/val_base_logits.npz")
    assert np.array_equal(cache["series_id"], sv) and np.array_equal(cache["t_online"], tv)
    base_logit = cache["base_logit"]
    gru = np.mean([np.load("features/" + f)["val_logit"] for f in GRU_FILES], axis=0)
    shipped = (1 - W_GRU) * base_logit + W_GRU * gru
    ship_full = ts_auc_grouped(tv, yv, shipped)
    print(f"  reconstructed SHIPPED VAL TS-AUC = {ship_full:.4f} (expect ~0.6160)")

    t0 = time.time()
    et = ExtraTreesClassifier(**ET_PARAMS)
    et.fit(Xtr, ytr)
    print(f"  ET trained in {time.time()-t0:.0f}s")
    pet = et.predict_proba(Xv)[:, 1]
    et_logit = _logit(pet)
    et_sa = ts_auc_grouped(tv, yv, pet)
    rc_base = np.corrcoef(base_logit, et_logit)[0, 1]
    rc_ship = np.corrcoef(shipped, et_logit)[0, 1]
    print(f"  ET standalone VAL={et_sa:.4f}  rankcorr(base)={rc_base:.3f}  rankcorr(shipped)={rc_ship:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(et, os.path.join(OUT_DIR, "model.joblib"), compress=3)
    np.savez("features/val_et_logits.npz", val_logit=et_logit)
    print(f"  saved -> {OUT_DIR}/model.joblib + features/val_et_logits.npz")

    halfA = half_mask_A(sv)
    print(f"\n{'We':>6}{'full':>9}{'halfA':>9}{'halfB':>9}{'min':>9}")
    print(f"{'(ship)':>6}{ship_full:>9.4f}"
          f"{ts_auc_grouped(tv[halfA], yv[halfA], shipped[halfA]):>9.4f}"
          f"{ts_auc_grouped(tv[~halfA], yv[~halfA], shipped[~halfA]):>9.4f}"
          f"{min(ts_auc_grouped(tv[halfA], yv[halfA], shipped[halfA]), ts_auc_grouped(tv[~halfA], yv[~halfA], shipped[~halfA])):>9.4f}")
    best = None
    for we in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        bl = (1 - we) * shipped + we * et_logit
        full = ts_auc_grouped(tv, yv, bl)
        aA = ts_auc_grouped(tv[halfA], yv[halfA], bl[halfA])
        aB = ts_auc_grouped(tv[~halfA], yv[~halfA], bl[~halfA])
        tag = "  <--" if full > ship_full else ""
        print(f"{we:>6.2f}{full:>9.4f}{aA:>9.4f}{aB:>9.4f}{min(aA, aB):>9.4f}{tag}")
        if best is None or full > best[1]:
            best = (we, full, min(aA, aB))

    meta = dict(kind="extratrees", params=ET_PARAMS, split_seed=SPLIT_SEED,
                standalone_val=float(et_sa), rankcorr_base=float(rc_base),
                rankcorr_shipped=float(rc_ship), shipped_val=float(ship_full),
                best_we=float(best[0]), best_full=float(best[1]),
                best_minhalf=float(best[2]))
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nbest We={best[0]:.2f} full={best[1]:.4f} min-half={best[2]:.4f}")
    print(f"meta -> {OUT_DIR}/meta.json")


if __name__ == "__main__":
    main()
