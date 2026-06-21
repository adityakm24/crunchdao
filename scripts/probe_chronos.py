"""(B) signal probe — does a Chronos-Bolt embedding-distance carry break signal?

Cheap offline filter BEFORE any streaming/deployment plumbing. For a sample of
train series, at a grid of absolute online steps t, compute a FIXED trailing
online window (W_on) and a fixed historical window (W_hist), embed both with
chronos-bolt-small, and measure the pooled-embedding distance (cos / L2). Score
with the official TS-AUC and compare to simple mean/var-shift baselines on the
SAME rows + check rank-decorrelation (is it NEW signal?).

Note: Chronos instance-normalizes each context, so the distance reflects
DYNAMICS/SHAPE shifts (not pure level shifts our v2/v4 already capture) — the
hoped-for complementary signal. If TS-AUC ≈ 0.5 or it's redundant with mean-shift,
the TSFM avenue is dead and we save the heavy plumbing.

Run: HF_HUB_OFFLINE=1 uv run python scripts/probe_chronos.py --n 2500
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import torch
from scipy.stats import rankdata

from sb.data import iter_series
from sb.metric import ts_auc_grouped

W_ON = 96
W_HIST = 192
STRIDE = 16
MODEL = "models/chronos-bolt-small"


def pooled_embed(pipe, windows, bs=512):
    """Mean-pooled encoder embedding for a list of equal-length 1-D windows."""
    out = []
    for i in range(0, len(windows), bs):
        batch = [torch.tensor(w, dtype=torch.float32) for w in windows[i:i + bs]]
        emb, _ = pipe.embed(batch)              # (B, n_tok, 512), equal length => no pad
        out.append(emb.mean(dim=1).detach().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    args = ap.parse_args()

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cpu")

    # collect fixed-length windows + row metadata. reference = embed(hist[:W_HIST]);
    # null windows from disjoint LATER hist (break-free) give the per-series null
    # distance distribution (the v2 empirical-null calibration that made raw feats
    # jump 0.48->0.60). online windows are calibrated by that per-series null.
    N_NULL = 8
    ref_w, null_w, on_w = [], [], []
    null_owner, on_owner = [], []   # series row-index each null/online window maps to
    sid_r, t_r, y_r = [], [], []
    on_mshift, on_vshift = [], []
    null_mshift, null_vshift = [], []
    n_series = 0
    t0 = time.time()
    for s in iter_series("train"):
        if n_series >= args.n:
            break
        if s.n_hist < W_HIST + W_ON or s.n_online <= W_ON:
            continue
        xh = np.asarray(s.x_hist, dtype=np.float64)
        xo = np.asarray(s.x_online, dtype=np.float64)
        mu_h, sd_h = xh.mean(), xh.std() + 1e-9
        si = len(ref_w)
        ref_w.append(xh[:W_HIST])
        # null windows over later hist
        late = xh[W_HIST:]
        npos = max(1, (len(late) - W_ON) // max(1, ((len(late) - W_ON) // N_NULL + 1)))
        added_null = 0
        for p in range(0, len(late) - W_ON + 1, max(1, (len(late) - W_ON) // N_NULL + 1)):
            w = late[p:p + W_ON]
            null_w.append(w); null_owner.append(si)
            null_mshift.append(abs(w.mean() - mu_h) / sd_h)
            null_vshift.append(abs(np.log((w.std() + 1e-9) / sd_h)))
            added_null += 1
            if added_null >= N_NULL:
                break
        tau = s.tau_index if s.tau_index is not None else -1
        for t in range(W_ON - 1, s.n_online, STRIDE):
            win = xo[t - W_ON + 1: t + 1]
            on_w.append(win); on_owner.append(si)
            sid_r.append(s.series_id); t_r.append(t)
            y_r.append(1 if (tau >= 0 and t >= tau) else 0)
            on_mshift.append(abs(win.mean() - mu_h) / sd_h)
            on_vshift.append(abs(np.log((win.std() + 1e-9) / sd_h)))
        n_series += 1

    y = np.asarray(y_r); t = np.asarray(t_r)
    on_owner = np.asarray(on_owner); null_owner = np.asarray(null_owner)
    print(f"[collect] {n_series} series, {len(on_w)} online rows, "
          f"{len(null_w)} null rows, pos_rate={y.mean():.3f}, {time.time()-t0:.0f}s")

    t0 = time.time()
    Eref = pooled_embed(pipe, ref_w)
    Enull = pooled_embed(pipe, null_w)
    Eon = pooled_embed(pipe, on_w)
    print(f"[embed] ref {Eref.shape} null {Enull.shape} online {Eon.shape}  "
          f"{time.time()-t0:.0f}s")

    def dists(E, owner):
        R = Eref[owner]
        dot = (E * R).sum(1)
        cos = 1.0 - dot / ((np.linalg.norm(E, 1) if False else np.linalg.norm(E, axis=1) + 1e-9)
                           * (np.linalg.norm(R, axis=1) + 1e-9))
        l2 = np.linalg.norm(E - R, axis=1)
        return cos, l2

    on_cos, on_l2 = dists(Eon, on_owner)
    nl_cos, nl_l2 = dists(Enull, null_owner)

    def calibrate(on_feat, on_own, nl_feat, nl_own, n_series):
        """z-score each online feature by its series' null mean/std."""
        mu = np.zeros(n_series); sd = np.ones(n_series)
        for si in range(n_series):
            v = nl_feat[nl_own == si]
            if len(v) >= 2:
                mu[si] = v.mean(); sd[si] = v.std() + 1e-9
            elif len(v) == 1:
                mu[si] = v[0]
        return (on_feat - mu[on_own]) / sd[on_own]

    cos_cal = calibrate(on_cos, on_owner, nl_cos, null_owner, n_series)
    l2_cal = calibrate(on_l2, on_owner, nl_l2, null_owner, n_series)
    ms = np.asarray(on_mshift); vs = np.asarray(on_vshift)
    ms_cal = calibrate(ms, on_owner, np.asarray(null_mshift), null_owner, n_series)
    vs_cal = calibrate(vs, on_owner, np.asarray(null_vshift), null_owner, n_series)

    def auc(name, feat):
        a = ts_auc_grouped(t, y, feat)
        print(f"  TS-AUC {name:22s} = {a:.4f}")
        return a

    print("\n[TS-AUC RAW]")
    auc("chronos_cos raw", on_cos); auc("chronos_l2 raw", on_l2)
    auc("mean_shift raw", ms); auc("var_shift raw", vs)
    print("\n[TS-AUC PER-SERIES NULL-CALIBRATED]")
    a_cos = auc("chronos_cos CAL", cos_cal)
    a_l2 = auc("chronos_l2 CAL", l2_cal)
    a_ms = auc("mean_shift CAL", ms_cal)
    a_vs = auc("var_shift CAL", vs_cal)
    a_both = auc("mean+var CAL", ms_cal + 0.5 * vs_cal)

    def sp(a, b):
        return np.corrcoef(rankdata(a), rankdata(b))[0, 1]
    print("\n[decorrelation: spearman of CAL chronos vs CAL simple]")
    print(f"  cos vs mean = {sp(cos_cal, ms_cal):.3f}  cos vs var = {sp(cos_cal, vs_cal):.3f}")
    print(f"\n>>> chronos CAL best {max(a_cos,a_l2):.4f} vs simple CAL best "
          f"{max(a_ms,a_vs,a_both):.4f}  (NEW signal if chronos>0.55 AND spearman<0.5)")


if __name__ == "__main__":
    main()
