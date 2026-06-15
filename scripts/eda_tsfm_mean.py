"""Round 7 probe #3 (FINAL TSFM variant) — MEAN-pooled embeddings.

Probes #1 (distance) and #2 (supervised head) both used the last (EOS) token and
came back dead flat (~0.50 / ~0.515 AUC, +0.001 blend lift). The remaining design
gap is pooling: mean-pooling over all encoder positions is the standard embedding
choice and can capture global distributional structure the single EOS token misses.

To mean-pool EXACTLY (no left-pad contamination) we bucket windows by length so
every batch is padding-free, then average all L+1 token vectors. Then we re-run
BOTH the distance summary and the supervised head. If this is also flat, the TSFM
avenue is a bulletproof negative across {distance, supervised} x {EOS, mean}.

Run: HF_HUB_OFFLINE=1 uv run python scripts/eda_tsfm_mean.py
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
from sb.data import iter_series  # noqa: E402

SEED = 42
SNAP_STEPS = [60, 100, 150, 200, 300, 400]
TRAIL = 128
CTX = 512
EARLY = 64
CLIP = 8.0
MODEL = "models/chronos-t5-mini"
GRU_FILES = ("val_seq_logits.npz", "val_seq_logits_s1.npz", "val_seq_logits_s2.npz")


def cross_auc(vals, labels):
    vals = np.asarray(vals, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = vals[labels == 1], vals[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def main():
    import torch
    from chronos import BaseChronosPipeline
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    idx = pd.read_parquet("Dataset/y_train_index.parquet")
    uniq = np.array(sorted(idx.index.tolist()))
    np.random.default_rng(SEED).shuffle(uniq)
    val_ids = sorted(int(v) for v in uniq[: int(0.2 * len(uniq))])

    base = np.load("features/val_base_logits.npz")
    gru = np.mean([np.load("features/" + f)["val_logit"] for f in GRU_FILES], axis=0)
    shipped = 0.55 * base["base_logit"] + 0.45 * gru
    sid_c, t_c, y_c = base["series_id"], base["t_online"], base["y"].astype(np.int64)
    ship_lookup = {(int(sid_c[k]), int(t_c[k])): (float(shipped[k]), int(y_c[k])) for k in range(len(sid_c))}

    # ---- build windows ----
    windows, meta = [], []
    for s in iter_series("train", ids=val_ids):
        xh, xo = s.x_hist.astype(np.float64), s.x_online.astype(np.float64)
        if len(xo) <= 20:
            continue
        mu, sd = xh.mean(), xh.std() + 1e-9
        zh = np.clip((xh - mu) / sd, -CLIP, CLIP)
        zo = np.clip((xo - mu) / sd, -CLIP, CLIP)
        windows.append(zh[-CTX:].astype(np.float32)); meta.append((s.series_id, "hist", -1))
        windows.append(zo[:EARLY].astype(np.float32)); meta.append((s.series_id, "early", -1))
        for t in SNAP_STEPS:
            if len(xo) <= t:
                continue
            windows.append(zo[max(0, t - CTX):t].astype(np.float32)); meta.append((s.series_id, "cum", t))
            windows.append(zo[max(0, t - TRAIL):t].astype(np.float32)); meta.append((s.series_id, "trail", t))
    print(f"windows: {len(windows)}")

    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="mps", torch_dtype=torch.float32)

    # ---- MEAN-pool, length-bucketed (no padding) ----
    buckets = defaultdict(list)
    for i, w in enumerate(windows):
        buckets[len(w)].append(i)
    embs = np.zeros((len(windows), 384), dtype=np.float32)
    done = 0
    with torch.no_grad():
        for L, idxs in buckets.items():
            for j in range(0, len(idxs), 256):
                grp = idxs[j:j + 256]
                chunk = [torch.tensor(windows[k], dtype=torch.float32) for k in grp]
                e, _ = pipe.embed(chunk)              # (B, L+1, 384), uniform length -> no pad
                pooled = e.mean(dim=1).cpu().numpy().astype(np.float32)
                for k, p in zip(grp, pooled):
                    embs[k] = p
                done += len(grp)
            print(f"   L={L:4d} done, total {done}/{len(windows)}")

    E = {(int(m[0]), str(m[1]), int(m[2])): embs[i] for i, m in enumerate(meta)}

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # ---- distance summary ----
    feats = ["cos_hist_cum", "l2_hist_cum", "cos_hist_trail", "cos_early_trail", "l2_early_trail"]
    per_step = {f: [] for f in feats}
    ship_aucs, blend_best = [], {f: [] for f in feats}
    for t in SNAP_STEPS:
        rf = {f: [] for f in feats}
        rs, ry = [], []
        for sid in val_ids:
            if (sid, "cum", t) not in E:
                continue
            look = ship_lookup.get((sid, t))
            if look is None:
                continue
            eh, ee = E[(sid, "hist", -1)], E[(sid, "early", -1)]
            ec, et = E[(sid, "cum", t)], E[(sid, "trail", t)]
            rf["cos_hist_cum"].append(1 - cos(eh, ec))
            rf["l2_hist_cum"].append(float(np.linalg.norm(eh - ec)))
            rf["cos_hist_trail"].append(1 - cos(eh, et))
            rf["cos_early_trail"].append(1 - cos(ee, et))
            rf["l2_early_trail"].append(float(np.linalg.norm(ee - et)))
            rs.append(look[0]); ry.append(look[1])
        y = np.array(ry); shp = np.array(rs)
        if y.sum() == 0:
            continue
        a_ship = cross_auc(shp, y); ship_aucs.append(a_ship)
        for f in feats:
            fv = np.array(rf[f])
            per_step[f].append(cross_auc(fv, y))
            shz, fvz = zscore(shp), zscore(fv)
            best = a_ship
            for w in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
                best = max(best, cross_auc(shz + w * fvz, y))
            blend_best[f].append(best)
    print("\n--- MEAN-pool DISTANCE summary ---")
    print(f"shipped: {np.nanmean(ship_aucs):.4f}")
    for f in feats:
        print(f"{f:18s} alone={np.nanmean(per_step[f]):.4f} blend={np.nanmean(blend_best[f]):.4f} "
              f"lift={np.nanmean(blend_best[f])-np.nanmean(ship_aucs):+.4f}")

    # ---- supervised head on mean-pooled diffs ----
    rows_X, rows_y, rows_ship, rows_grp, rows_step = [], [], [], [], []
    for sid in val_ids:
        if (sid, "hist", -1) not in E or (sid, "early", -1) not in E:
            continue
        eh, ee = E[(sid, "hist", -1)], E[(sid, "early", -1)]
        for t in SNAP_STEPS:
            if (sid, "cum", t) not in E or (sid, "trail", t) not in E:
                continue
            look = ship_lookup.get((sid, t))
            if look is None:
                continue
            ec, et = E[(sid, "cum", t)], E[(sid, "trail", t)]
            rows_X.append(np.concatenate([ec - eh, et - eh, et - ee]))
            rows_y.append(look[1]); rows_ship.append(look[0])
            rows_grp.append(sid); rows_step.append(t)
    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y); sh = np.array(rows_ship)
    grp = np.array(rows_grp); step = np.array(rows_step)
    print(f"\n--- MEAN-pool SUPERVISED head ({X.shape}) ---")
    for C in (0.01, 0.03):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=grp):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.decision_function(sc.transform(X[te]))
        ha, sa, bl = [], [], []
        for t in SNAP_STEPS:
            m = step == t
            if m.sum() < 50 or y[m].sum() == 0:
                continue
            a_s = cross_auc(sh[m], y[m])
            shz, hz = zscore(sh[m]), zscore(oof[m])
            best = a_s
            for w in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
                best = max(best, cross_auc(shz + w * hz, y[m]))
            ha.append(cross_auc(oof[m], y[m])); sa.append(a_s); bl.append(best)
        print(f"C={C}: head={np.mean(ha):.4f} ship={np.mean(sa):.4f} blend={np.mean(bl):.4f} "
              f"lift={np.mean(bl)-np.mean(sa):+.4f} rankcorr={np.corrcoef(sh, oof)[0,1]:.3f}")


if __name__ == "__main__":
    main()
