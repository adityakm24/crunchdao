"""Round 6 — "Epistemological Diversity": add genuinely different BASE LEARNERS.

Three exhausted hand-crafted avenues this round (subtle detectors, incremental
distributional, Shiryaev-Roberts) prove the base GBT has mined all the linearly /
tree-separable signal in the calibrated features. The documented 2025-winning
lever we have NOT tried is ENSEMBLE DIVERSITY via different base-learner FAMILIES
(bagging: RandomForest / ExtraTrees) rather than more boosting seeds (corr 0.97-
0.99). Bagging has a fundamentally different bias -> lower error-correlation ->
blend lift even when individually weaker (cf. the logistic member, corr 0.93,
lifted the blend). sklearn IS allowed in the cloud, so an RF/ET member trains in
train() and serves via joblib -- no neural export / parity risk.

We train ET and RF on the same VAL-split TRAIN rows of the v4 matrix, then measure
on VAL: standalone TS-AUC, rank-correlation vs the GBT mean, and the blend lift
when added to the shipped base (and to the GBT mean). Decision: integrate only if
it decorrelates (corr < ~0.95) AND lifts the blend on honest halves.

Run: uv run python scripts/diverse_members.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sb.metric import ts_auc_grouped

V4 = "features/train_features_v4.npz"
SPLIT_SEED = 42


def val_mask(sid):
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids))


def half_masks(sid_val):
    """Honest 2-fold on VAL series (rng seed 1, first half = A)."""
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
    print(f"  train rows={tr.sum():,}  val rows={va.sum():,}  feats={X.shape[1]}")

    # cached base logits (same split/order) for blend comparison
    cache = np.load("features/val_base_logits.npz")
    assert np.array_equal(cache["series_id"], sv) and np.array_equal(cache["t_online"], tv), \
        "cache VAL order mismatch"
    gbt_mean = cache["gbt_logits"].mean(axis=1)
    base_logit = cache["base_logit"]
    print(f"  GBT-mean VAL TS-AUC = {ts_auc_grouped(tv, yv, gbt_mean):.4f}")
    print(f"  BASE     VAL TS-AUC = {ts_auc_grouped(tv, yv, base_logit):.4f}")

    halfA = half_masks(sv)

    def report(name, p):
        pl = _logit(p)
        sa = ts_auc_grouped(tv, yv, p)
        rc = np.corrcoef(gbt_mean, pl)[0, 1]
        print(f"\n[{name}] standalone TS-AUC={sa:.4f}  rankcorr(vs GBTmean)={rc:.3f}")
        # blend onto base over weight grid, full + honest halves
        best = None
        for w in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
            bl = (1 - w) * base_logit + w * pl
            full = ts_auc_grouped(tv, yv, bl)
            aA = ts_auc_grouped(tv[halfA], yv[halfA], bl[halfA])
            aB = ts_auc_grouped(tv[~halfA], yv[~halfA], bl[~halfA])
            tag = "  <-- " if (full > ts_auc_grouped(tv, yv, base_logit)) else ""
            print(f"   w={w:.2f}  full={full:.4f}  halfA={aA:.4f}  halfB={aB:.4f}  min={min(aA,aB):.4f}{tag}")
            if best is None or full > best[1]:
                best = (w, full)
        return best

    # ---- ExtraTrees (fast, random splits = max diversity) ----
    t0 = time.time()
    et = ExtraTreesClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=200,
        bootstrap=False, n_jobs=-1, random_state=42,
    )
    et.fit(Xtr, ytr)
    pet = et.predict_proba(Xv)[:, 1]
    print(f"\nExtraTrees trained in {time.time()-t0:.0f}s")
    report("ExtraTrees", pet)

    # ---- RandomForest (bagging) ----
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", min_samples_leaf=200,
        bootstrap=True, n_jobs=-1, random_state=42,
    )
    rf.fit(Xtr, ytr)
    prf = rf.predict_proba(Xv)[:, 1]
    print(f"\nRandomForest trained in {time.time()-t0:.0f}s")
    report("RandomForest", prf)

    # ---- ET+RF averaged as one "bagging member", then blended ----
    pbag = _logit(np.clip(0.5 * (pet + prf), 1e-7, 1 - 1e-7))
    sa = ts_auc_grouped(tv, yv, 0.5 * (pet + prf))
    rc = np.corrcoef(gbt_mean, pbag)[0, 1]
    print(f"\n[ET+RF avg] standalone={sa:.4f}  rankcorr={rc:.3f}")
    for w in (0.15, 0.2, 0.25, 0.3):
        bl = (1 - w) * base_logit + w * pbag
        full = ts_auc_grouped(tv, yv, bl)
        aA = ts_auc_grouped(tv[halfA], yv[halfA], bl[halfA])
        aB = ts_auc_grouped(tv[~halfA], yv[~halfA], bl[~halfA])
        print(f"   w={w:.2f}  full={full:.4f}  halfA={aA:.4f}  halfB={aB:.4f}  min={min(aA,aB):.4f}")


if __name__ == "__main__":
    main()
