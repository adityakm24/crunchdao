"""Train a deterministic causal 1D-CNN sequence member over the RAW per-series-
standardized online stream z_t = (x_t - mu_h)/sd_h. This is the ONE untested
lever after five experiments showed the model is saturated on the 162 calibrated
features: a DIFFERENT INPUT (raw windowed shape) seen through a WINDOWED receptive
field — unlike the GBT/GRU summary-stat input and unlike the failed single-POINT
raw GRU. A stack of causal dilated convs JOINTLY learns multi-scale shape filters
(level-shift edges, variance envelopes, local autocorrelation) with nonlinear
composition the linear matched-filter probe (eda_window.py) cannot capture.

  * Input  : 1 channel, the raw standardized z stream (train_raw_stream.npz col 0).
  * Model  : K causal dilated Conv1d blocks (dilations 1,2,4,8...), C channels,
             kernel 3, tanh activations + residual; 1x1 conv head -> per-step logit.
  * Causal : left-pad only; step t depends only on z[:t+1] (receptive field R).
  * Loss   : per-step BCEWithLogits with the same elapsed-ramp positive weight.
  * Select : best VAL TS-AUC epoch (CNN can overfit the ranking metric too).

Inference must be deterministic -> exported to numpy float64 and re-implemented
as an explicit causal conv over a ring buffer (parity-checked vs torch here).
Saves artifacts/models/model_031_cnn/cnn.npz + features/val_cnn_logits.npz.

Run: uv run python scripts/train_cnn.py --channels 24 --blocks 4 --epochs 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np

from sb.metric import ts_auc_grouped

RAW = "features/train_raw_stream.npz"
SPLIT_SEED = 42
RAMP = 50.0
OUTDIR = "artifacts/models/model_031_cnn"


def val_mask(sid):
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids))


def ramp_weight(y_seg):
    w = np.ones(len(y_seg), dtype=np.float32)
    if y_seg.any():
        tau = int(y_seg.argmax())
        idx = np.arange(len(y_seg))
        w[tau:] = np.clip((idx[tau:] - tau) / RAMP, 0.2, 1.0)
    return w


def numpy_cnn_forward(z, p):
    """Pure-numpy causal dilated 1D-CNN over a standardized stream -> (T,).

    `z` is either (T,) single-channel or (T, Cin) multi-channel.
    Matches the torch model exactly (float64). Architecture per block b:
        y = tanh( conv1d_causal(x, W[b], dil[b]) + bconv[b] )   # Cin->C (block0)
        x = y + x   if residual and shapes match   else y
    head: logit_t = Wo . x_t + bo                                 # C->1 (1x1 conv)
    Standardization: clip to +-CLIP first (already standardized upstream).
    W[b] shape (Cout, Cin, k). Causal: output t uses inputs t-(k-1)*dil ... t.
    """
    CLIP = float(p["clip"])
    z = np.clip(np.asarray(z, dtype=np.float64), -CLIP, CLIP)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    T = z.shape[0]
    x = z.T.copy()  # (Cin, T)
    dils = p["dilations"]
    k = int(p["ksize"])
    nb = int(p["nblocks"])
    Ws = [p[f"W{b}"] for b in range(nb)]      # each (Cout,Cin,k)
    bs = [p[f"bconv{b}"] for b in range(nb)]  # each (Cout,)
    for b in range(nb):
        W = Ws[b]; bb = bs[b]; dil = int(dils[b])
        Cout, Cin, _ = W.shape
        y = np.zeros((Cout, T), dtype=np.float64)
        for j in range(k):
            shift = (k - 1 - j) * dil  # tap j is `shift` steps in the past
            if shift == 0:
                xs = x
            elif shift >= T:
                continue  # tap entirely before the series start -> zeros
            else:
                xs = np.zeros_like(x)
                xs[:, shift:] = x[:, :T - shift]
            # y[co] += sum_ci W[co,ci,j] * xs[ci]
            y += W[:, :, j] @ xs
        y += bb.reshape(Cout, 1)
        y = np.tanh(y)
        if y.shape[0] == x.shape[0]:
            x = y + x
        else:
            x = y
    Wo = p["Wo"]  # (C,)
    bo = float(p["bo"])
    return (Wo @ x + bo).astype(np.float64)  # (T,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=24)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--ksize", type=int, default=3)
    ap.add_argument("--in-channels", type=int, default=1,
                    help="1=z only; 2=z,z2; 3=z,z2,absz (raw-stream cols)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", type=float, default=8.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--val-out", default="features/val_cnn_logits.npz")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = ("mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device={dev} channels={args.channels} blocks={args.blocks} k={args.ksize}")

    d = np.load(RAW, allow_pickle=True)
    Xall, y, sid, t = d["X"], d["y"], d["series_id"], d["t_online"]
    Cin = int(args.in_channels)
    z_all = Xall[:, :Cin].astype(np.float32)  # (N, Cin) raw standardized channels
    print(f"raw stream rows={len(z_all):,}  in_channels={Cin}")

    va = val_mask(sid)
    uniq, start = np.unique(sid, return_index=True)
    bounds = list(start) + [len(sid)]
    series = [(int(start[i]), int(bounds[i + 1])) for i in range(len(uniq))]
    is_val = np.array([bool(va[s0]) for s0, _ in series])
    train_idx = [i for i in range(len(series)) if not is_val[i]]
    val_idx = [i for i in range(len(series)) if is_val[i]]
    train_idx.sort(key=lambda i: series[i][1] - series[i][0])

    dilations = [2 ** b for b in range(args.blocks)]
    R = 1 + (args.ksize - 1) * sum(dilations)
    print(f"dilations={dilations}  receptive field R={R}")

    class CausalConv(nn.Module):
        def __init__(self, ci, co, k, dil):
            super().__init__()
            self.pad = (k - 1) * dil
            self.conv = nn.Conv1d(ci, co, k, dilation=dil)

        def forward(self, x):
            x = nn.functional.pad(x, (self.pad, 0))
            return self.conv(x)

    class CNN(nn.Module):
        def __init__(self, C, nb, k, dils, dropout, in_ch):
            super().__init__()
            self.blocks = nn.ModuleList()
            ci = in_ch
            for b in range(nb):
                self.blocks.append(CausalConv(ci, C, k, dils[b]))
                ci = C
            self.drop = nn.Dropout(dropout)
            self.head = nn.Conv1d(C, 1, 1)

        def forward(self, x):  # x: (B,Cin,T)
            for b, blk in enumerate(self.blocks):
                y = torch.tanh(blk(x))
                x = y + x if y.shape[1] == x.shape[1] else y
            return self.head(self.drop(x)).squeeze(1)  # (B,T)

    net = CNN(args.channels, args.blocks, args.ksize, dilations, args.dropout, Cin).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    yv = y[va].astype(np.int64)
    tv = t[va]
    CLIP = args.clip

    def seq_t(s0, s1):
        z = np.clip(z_all[s0:s1].astype(np.float32), -CLIP, CLIP)  # (L, Cin)
        return torch.from_numpy(z.T.copy())  # (Cin, L)

    def eval_val():
        net.eval()
        val_pred = np.zeros(len(sid), dtype=np.float64)
        with torch.no_grad():
            B = 64
            for k in range(0, len(val_idx), B):
                chunk = val_idx[k:k + B]
                lens = [series[i][1] - series[i][0] for i in chunk]
                mx = max(lens)
                xb = torch.zeros(len(chunk), Cin, mx)
                for j, i in enumerate(chunk):
                    s0, s1 = series[i]
                    xb[j, :, : s1 - s0] = seq_t(s0, s1)
                logits = net(xb.to(dev)).cpu().numpy()
                for j, i in enumerate(chunk):
                    s0, s1 = series[i]
                    val_pred[s0:s1] = logits[j, : s1 - s0]
        return val_pred[va]

    best = (-1.0, None, -1)
    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        order = train_idx[:]
        rng = np.random.default_rng(1000 + ep)
        groups = [order[k:k + args.batch] for k in range(0, len(order), args.batch)]
        rng.shuffle(groups)
        tot = 0.0
        for g in groups:
            lens = [series[i][1] - series[i][0] for i in g]
            mx = max(lens)
            xb = torch.zeros(len(g), Cin, mx)
            yb = torch.zeros(len(g), mx)
            wb = torch.zeros(len(g), mx)
            mask = torch.zeros(len(g), mx)
            for j, i in enumerate(g):
                s0, s1 = series[i]
                L = s1 - s0
                xb[j, :, :L] = seq_t(s0, s1)
                yy = y[s0:s1].astype(np.float32)
                yb[j, :L] = torch.from_numpy(yy)
                wb[j, :L] = torch.from_numpy(ramp_weight(yy))
                mask[j, :L] = 1.0
            xb, yb, wb, mask = xb.to(dev), yb.to(dev), wb.to(dev), mask.to(dev)
            logits = net(xb)
            loss = (bce(logits, yb) * wb * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(g)
        sched.step()
        vauc = ts_auc_grouped(tv, yv, eval_val())
        print(f"  epoch {ep:2d}  loss {tot/len(train_idx):.4f}  VAL TS-AUC {vauc:.4f}  ({time.time()-t0:.1f}s)")
        if vauc > best[0]:
            sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best = (vauc, sd, ep)

    print(f"\nBest VAL TS-AUC {best[0]:.4f} @ epoch {best[2]}")
    net.load_state_dict(best[1])

    # export numpy weights (float64)
    sd = net.state_dict()
    W = [sd[f"blocks.{b}.conv.weight"].cpu().numpy().astype(np.float64) for b in range(args.blocks)]
    bconv = [sd[f"blocks.{b}.conv.bias"].cpu().numpy().astype(np.float64) for b in range(args.blocks)]
    Wo = sd["head.weight"].cpu().numpy()[:, :, 0][0].astype(np.float64)  # (C,)
    bo = float(sd["head.bias"].cpu().numpy()[0])
    p = dict(dilations=np.array(dilations), ksize=args.ksize, clip=float(CLIP),
             nblocks=args.blocks, Wo=Wo, bo=bo)
    for b in range(args.blocks):
        p[f"W{b}"] = W[b]
        p[f"bconv{b}"] = bconv[b]

    # parity check numpy vs torch on a few val series
    net.eval()
    maxdiff = 0.0
    import torch as _t
    with _t.no_grad():
        for i in val_idx[:5]:
            s0, s1 = series[i]
            zz = z_all[s0:s1].astype(np.float64)
            npy = numpy_cnn_forward(zz, p)
            xb = _t.zeros(1, Cin, s1 - s0)
            xb[0, :, :] = seq_t(s0, s1)
            tch = net(xb.to(dev)).cpu().numpy()[0, : s1 - s0]
            maxdiff = max(maxdiff, float(np.abs(npy - tch).max()))
    print(f"numpy<->torch parity max|diff| = {maxdiff:.2e}")

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "cnn.npz"), kind="cnn", in_channels=Cin,
             channels=args.channels, blocks=args.blocks, **p)
    val_pred = np.zeros(len(sid), dtype=np.float64)
    for i in val_idx:
        s0, s1 = series[i]
        val_pred[s0:s1] = numpy_cnn_forward(z_all[s0:s1].astype(np.float64), p)
    np.savez(args.val_out, val_logit=val_pred[va])
    final = ts_auc_grouped(tv, yv, val_pred[va])
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(dict(val_ts_auc=round(float(final), 5), kind="cnn",
                       channels=args.channels, blocks=args.blocks,
                       ksize=args.ksize, dilations=dilations, receptive_field=R,
                       best_epoch=best[2], parity_maxdiff=maxdiff), fh, indent=2)
    print(f"numpy VAL TS-AUC {final:.4f}  saved -> {args.out}/cnn.npz")


if __name__ == "__main__":
    main()
