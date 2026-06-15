"""Round 7 probe #2 (STRONG form) — supervised head on Chronos embeddings.

The naive distance probe (eda_tsfm.py) showed ~0.50 standalone AUC: collapsing
the 384-dim embedding to a single cosine/L2 scalar throws away any signal that
lives in a *direction* of embedding space. This probe trains a lightweight
logistic head on the embedding-difference vectors (emb_cum - emb_hist, etc.) with
GROUPED cross-validation by series, then measures the OOF cross-sectional AUC and
the incremental lift when blended onto the shipped logit. This is the decisive,
strong-form test: if even a supervised projection of the embedding cannot beat
~0.50 / cannot lift the blend, the TSFM avenue is genuinely a negative.

Uses the embeddings already cached in features/val_tsfm_probe.npz (no Chronos).
Run: uv run python scripts/eda_tsfm_head.py
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
SNAP_STEPS = [60, 100, 150, 200, 300, 400]
GRU_FILES = ("val_seq_logits.npz", "val_seq_logits_s1.npz", "val_seq_logits_s2.npz")


def cross_auc(vals, labels):
    vals = np.asarray(vals, dtype=np.float64)
    labels = np.asarray(labels)
    pos = vals[labels == 1]
    neg = vals[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    rsum = ranks[:len(pos)].sum()
    return float((rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def main():
    d = np.load("features/val_tsfm_probe.npz", allow_pickle=True)
    meta = d["meta"]
    embs = d["embs"].astype(np.float64)
    E = {}
    for m, e in zip(meta, embs):
        E[(int(m[0]), str(m[1]), int(m[2]))] = e
    print(f"embeddings: {embs.shape}")

    # ---- shipped logit + label per (sid, online-step) ----
    base = np.load("features/val_base_logits.npz")
    gru = np.mean([np.load("features/" + f)["val_logit"] for f in GRU_FILES], axis=0)
    shipped = 0.55 * base["base_logit"] + 0.45 * gru
    sid_c, t_c, y_c = base["series_id"], base["t_online"], base["y"].astype(np.int64)
    ship_lookup = {}
    for k in range(len(sid_c)):
        ship_lookup[(int(sid_c[k]), int(t_c[k]))] = (float(shipped[k]), int(y_c[k]))

    # ---- assemble supervised dataset over all (sid, step) ----
    sids = sorted({int(m[0]) for m in meta})
    rows_X, rows_y, rows_ship, rows_grp, rows_step = [], [], [], [], []
    for sid in sids:
        if (sid, "hist", -1) not in E or (sid, "early", -1) not in E:
            continue
        eh = E[(sid, "hist", -1)]
        ee = E[(sid, "early", -1)]
        for t in SNAP_STEPS:
            kc = (sid, "cum", t)
            kt = (sid, "trail", t)
            if kc not in E or kt not in E:
                continue
            look = ship_lookup.get((sid, t))
            if look is None:
                continue
            ec, et = E[kc], E[kt]
            feat = np.concatenate([ec - eh, et - eh, et - ee])  # 3*384 = 1152 dims
            rows_X.append(feat)
            rows_y.append(look[1])
            rows_ship.append(look[0])
            rows_grp.append(sid)
            rows_step.append(t)
    X = np.array(rows_X)
    y = np.array(rows_y)
    sh = np.array(rows_ship)
    grp = np.array(rows_grp)
    step = np.array(rows_step)
    print(f"samples: {X.shape}, pos rate {y.mean():.3f}")

    # ---- grouped OOF logistic head ----
    for C in (0.01, 0.03, 0.1):
        oof = np.zeros(len(y))
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(X, y, groups=grp):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.decision_function(sc.transform(X[te]))

        # per-step AUC + blend
        head_aucs, ship_aucs, blends = [], [], []
        for t in SNAP_STEPS:
            mask = step == t
            if mask.sum() < 50 or y[mask].sum() == 0:
                continue
            a_head = cross_auc(oof[mask], y[mask])
            a_ship = cross_auc(sh[mask], y[mask])
            shz, hz = zscore(sh[mask]), zscore(oof[mask])
            best = a_ship
            for w in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
                best = max(best, cross_auc(shz + w * hz, y[mask]))
            head_aucs.append(a_head)
            ship_aucs.append(a_ship)
            blends.append(best)
        rc = np.corrcoef(sh, oof)[0, 1]
        print(f"\nC={C}: head_auc={np.mean(head_aucs):.4f}  ship={np.mean(ship_aucs):.4f}  "
              f"blend={np.mean(blends):.4f}  lift={np.mean(blends)-np.mean(ship_aucs):+.4f}  "
              f"rankcorr={rc:.3f}")
        for t, ha, sa, bl in zip([s for s in SNAP_STEPS if (step == s).sum() >= 50],
                                 head_aucs, ship_aucs, blends):
            print(f"   t={t:3d}  head={ha:.4f}  ship={sa:.4f}  blend={bl:.4f}")


if __name__ == "__main__":
    main()
