"""Round 7 probe — does a TSFM (Chronos-T5-mini) embedding-shift feature carry
INCREMENTAL cross-sectional signal over the shipped 0.6160 stack?

The competition blueprint (info.md, "Blueprint 2") proposes a TSFM as a universal
embedding engine: embed the historical (break-free) segment and a trailing/online
window, and use their latent distance as a break score. This probe measures, at
several metric-heavy snapshot online steps, the cross-sectional AUC of such
distances ALONE and the incremental lift when blended onto the shipped logit
(base + 3 GRU, reconstructed from cached val logits). DECISIVE gate before any
deployment work: if the optimal blend weight is ~0 / no lift (like the 6 round-6
negatives), stop. If there is real orthogonal signal, proceed to dense compute +
full TS-AUC + honest halves + OOS.

Chronos pooling: last (EOS) token of the encoder output — verified batch-invariant
(left-padding). Run: HF_HUB_OFFLINE=1 uv run python scripts/eda_tsfm.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from sb.data import iter_series  # noqa: E402

SEED = 42
SNAP_STEPS = [60, 100, 150, 200, 300, 400]
TRAIL = 128
CTX = 512            # cap window length fed to Chronos (its native context)
EARLY = 64           # early-online reference window length
CLIP = 8.0
MODEL = "models/chronos-t5-mini"
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
    m, s = a.mean(), a.std() + 1e-12
    return (a - m) / s


def main():
    import torch
    from chronos import BaseChronosPipeline

    # ---- VAL ids: identical split to everything else (rng(42), first 20%) ----
    import pandas as pd
    idx = pd.read_parquet("Dataset/y_train_index.parquet")
    uniq = np.array(sorted(idx.index.tolist()))
    np.random.default_rng(SEED).shuffle(uniq)
    val_ids = set(int(v) for v in uniq[: int(0.2 * len(uniq))])
    print(f"VAL series: {len(val_ids)}")

    # ---- shipped logit + label per (sid, online-step) from caches ----
    base = np.load("features/val_base_logits.npz")
    gru = np.mean([np.load("features/" + f)["val_logit"] for f in GRU_FILES], axis=0)
    shipped = 0.55 * base["base_logit"] + 0.45 * gru
    sid_c, t_c, y_c = base["series_id"], base["t_online"], base["y"].astype(np.int64)
    ship_lookup = {}
    for k in range(len(sid_c)):
        ship_lookup[(int(sid_c[k]), int(t_c[k]))] = (float(shipped[k]), int(y_c[k]))

    # ---- gather windows to embed ----
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="mps", torch_dtype=torch.float32)

    def emb_batch(arrs, bs=256):
        outs = []
        with torch.no_grad():
            for i in range(0, len(arrs), bs):
                chunk = [torch.tensor(a, dtype=torch.float32) for a in arrs[i:i + bs]]
                e, _ = pipe.embed(chunk)        # (B, Lmax, 384), left-padded
                outs.append(e[:, -1, :].cpu().numpy().astype(np.float32))  # EOS token
                if (i // bs) % 10 == 0:
                    print(f"   embed {i+len(chunk)}/{len(arrs)}")
        return np.concatenate(outs, axis=0)

    windows = []          # list of float32 arrays
    meta = []             # (sid, kind, step)  kind in {hist, early, cum, trail}
    print("building windows ...")
    nseries = 0
    for s in iter_series("train", ids=sorted(val_ids)):
        xh = s.x_hist.astype(np.float64)
        xo = s.x_online.astype(np.float64)
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
        nseries += 1
    print(f"series used: {nseries}, windows: {len(windows)}")

    embs = emb_batch(windows)
    # index embeddings
    E = {}
    for m, e in zip(meta, embs):
        E[m] = e

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # ---- per-step features + decisive AUC ----
    feats = ["cos_hist_cum", "l2_hist_cum", "cos_hist_trail", "cos_early_trail", "l2_early_trail"]
    per_step = {f: [] for f in feats}
    ship_aucs, blend_best = [], {f: [] for f in feats}
    rankcorr = {f: [] for f in feats}

    for t in SNAP_STEPS:
        rows_feat = {f: [] for f in feats}
        rows_ship, rows_y = [], []
        for sid in sorted(val_ids):
            key = (sid, "cum", t)
            if key not in E:
                continue
            look = ship_lookup.get((sid, t))
            if look is None:
                continue
            eh = E[(sid, "hist", -1)]
            ee = E[(sid, "early", -1)]
            ec = E[(sid, "cum", t)]
            et = E[(sid, "trail", t)]
            rows_feat["cos_hist_cum"].append(1 - cos(eh, ec))
            rows_feat["l2_hist_cum"].append(float(np.linalg.norm(eh - ec)))
            rows_feat["cos_hist_trail"].append(1 - cos(eh, et))
            rows_feat["cos_early_trail"].append(1 - cos(ee, et))
            rows_feat["l2_early_trail"].append(float(np.linalg.norm(ee - et)))
            rows_ship.append(look[0]); rows_y.append(look[1])
        y = np.array(rows_y)
        sh = np.array(rows_ship)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        a_ship = cross_auc(sh, y)
        ship_aucs.append(a_ship)
        line = [f"t={t:3d} n={len(y):4d} pos={int(y.sum()):4d}  ship={a_ship:.4f}"]
        for f in feats:
            fv = np.array(rows_feat[f])
            a_f = cross_auc(fv, y)
            per_step[f].append(a_f)
            rankcorr[f].append(np.corrcoef(sh, fv)[0, 1])
            # best blend weight (z-scored)
            shz, fvz = zscore(sh), zscore(fv)
            best = a_ship
            for w in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
                best = max(best, cross_auc(shz + w * fvz, y))
            blend_best[f].append(best)
            line.append(f"{f}={a_f:.3f}(bl {best:.4f})")
        print("  " + "  ".join(line))

    print("\n================= SUMMARY (mean over snapshot steps) =================")
    print(f"shipped alone           : {np.nanmean(ship_aucs):.4f}")
    print(f"{'feature':18s}{'AUC_alone':>11s}{'blend':>9s}{'lift':>9s}{'rankcorr':>10s}")
    for f in feats:
        al = np.nanmean(per_step[f])
        bl = np.nanmean(blend_best[f])
        rc = np.nanmean(rankcorr[f])
        tag = "  <--" if bl > np.nanmean(ship_aucs) + 1e-4 else ""
        print(f"{f:18s}{al:>11.4f}{bl:>9.4f}{bl-np.nanmean(ship_aucs):>+9.4f}{rc:>10.3f}{tag}")

    np.savez("features/val_tsfm_probe.npz",
             meta=np.array(meta, dtype=object), embs=embs)
    print("\nsaved embeddings -> features/val_tsfm_probe.npz")


if __name__ == "__main__":
    main()
