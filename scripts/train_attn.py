"""Round 11 pivot — causal Transformer (self-attention) sequence-member PROBE.

Why: synthetic augmentation lifted single models but WASHED OUT at the blend
(round 11). The neural sub-ensemble is GRU/LSTM/CNN on the same v4 stream and is
documented-saturated (context.md §7: "do NOT keep adding nets on the same
calibrated input"). The ONE evidence-integration mechanism never tried is
SELF-ATTENTION: at each online step, attend (causally) over every prior online
step instead of gated recurrence. Round 4 proved a member only needs to be
DECORRELATED to help (the GRU was standalone ≈ GBT yet rank-corr 0.83 → blend
+0.012). So the gate is BOTH standalone VAL TS-AUC AND rank-corr to the shipped
GRU mean + whether it lifts the reconstructed round-10 blend (full + both honest
halves), all from CACHED logits (no GBT/GRU retrain).

Local probe only (torch MPS/CPU). VAL = the exact 2000 held-out real series
(split seed 42), matching the GRU rows so the logit is directly blendable. A
pure-numpy serve path is built ONLY if it clears the bar.

Run: KMP_DUPLICATE_LIB_OK=TRUE uv run python -u scripts/train_attn.py \
        --layers 3 --dmodel 128 --heads 8 --epochs 16 --no-logt
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

import numpy as np
from sb.metric import ts_auc_grouped

V4 = "features/train_features_v4.npz"
RAW_NPZ = "features/train_raw_stream.npz"
# Light per-series calibration context for raw-only mode: tells the learned
# detector how reliable the historical null is (n_hist), where in the online
# segment it is (relative position), and the historical dependence/tail shape.
RAW_CTX = ["t_over_nhist", "log_n_hist", "acf1_h", "kurt_h"]
SPLIT_SEED = 42
RAMP = 50.0
DROP = {"t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
        "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2"}
RANK_GW, W_LIN, W_GRU = 0.10, 0.10, 0.45


def val_mask(sid):
    uniq = np.unique(sid); rng = np.random.default_rng(SPLIT_SEED); rng.shuffle(uniq)
    return np.isin(sid, list(set(uniq[: int(0.2 * len(uniq))].tolist())))


def half_mask(sid_val):
    uniq = np.unique(sid_val); rng = np.random.default_rng(1); rng.shuffle(uniq)
    return np.isin(sid_val, list(set(uniq[: len(uniq) // 2].tolist())))


def ramp_weight(y_seg):
    w = np.ones(len(y_seg), dtype=np.float32)
    if y_seg.any():
        tau = int(y_seg.argmax()); idx = np.arange(len(y_seg))
        w[tau:] = np.clip((idx[tau:] - tau) / RAMP, 0.2, 1.0)
    return w


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra @ rb) / (np.sqrt(ra @ ra) * np.sqrt(rb @ rb) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dmodel", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=0, help="FFN dim (0=4*dmodel)")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--maxlen", type=int, default=0, help="train seq cap (0=none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--loss", choices=["bce", "rank"], default="bce",
                    help="bce=pointwise per-step; rank=within-step pairwise RankNet "
                         "(differentiable AUC surrogate, metric-aligned across the batch)")
    ap.add_argument("--no-logt", action="store_true")
    ap.add_argument("--raw", choices=["none", "concat", "only"], default="none",
                    help="none=151 calib; concat=calib+stream; only=stream+light ctx")
    ap.add_argument("--stream", default=RAW_NPZ,
                    help="extra per-step stream npz (aligned to v4) for --raw concat/only; "
                         "default is the raw z-stream, pass the PIT stream to swap input")
    ap.add_argument("--out", default="features/val_attn_logits.npz")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F_
    from torch.nn.utils.rnn import pad_sequence

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = (("mps" if torch.backends.mps.is_available() else "cpu")
           if args.device == "auto" else args.device)
    ff = args.ff or 4 * args.dmodel
    print(f"device={dev} layers={args.layers} d={args.dmodel} heads={args.heads} "
          f"ff={ff} epochs={args.epochs} batch={args.batch} maxlen={args.maxlen}")

    d = np.load(V4, allow_pickle=True)
    X, y, sid, t = d["X"], d["y"], d["series_id"], d["t_online"]
    names = [str(n) for n in d["feature_names"]]
    drop = set(DROP) | ({"log_t"} if args.no_logt else set())
    keep = [i for i, n in enumerate(names) if n not in drop]

    # Build the actual input matrix Xwork[:, keepwork] the model consumes.
    if args.raw == "none":
        Xwork, keepwork = X, keep
    else:
        rd = np.load(args.stream, allow_pickle=True)
        Xstream = rd["X"].astype(np.float32)  # (N,k) per-step extra channels
        snames = [str(n) for n in rd["feature_names"]]
        assert (np.array_equal(rd["series_id"], sid)
                and np.array_equal(rd["t_online"], t)), "stream order != v4"
        print(f"stream {args.stream} chans={snames}")
        if args.raw == "concat":
            Xwork = np.concatenate([X[:, keep].astype(np.float32), Xstream], axis=1)
        else:  # only: the stream trajectory + a light per-series calibration context
            ctx = [names.index(c) for c in RAW_CTX if c in names]
            Xwork = np.concatenate([Xstream, X[:, ctx].astype(np.float32)], axis=1)
        keepwork = list(range(Xwork.shape[1]))
        # Xwork is now self-contained; free the original 162-col X (~3.3GB) and the
        # stream handle to cut swap pressure (the v4 X is unused past this point).
        del X, Xstream, rd
        X = Xwork  # keep name bound for the shape print below
    F = len(keepwork)
    print(f"X={X.shape} raw={args.raw} -> using {F} input channels")

    va = val_mask(sid)
    uniq, start = np.unique(sid, return_index=True)
    bounds = list(start) + [len(sid)]
    series = [(int(start[i]), int(bounds[i + 1])) for i in range(len(uniq))]
    is_val = np.array([bool(va[s0]) for s0, _ in series])

    Xtr_rows = Xwork[~va][:, keepwork].astype(np.float32)
    np.nan_to_num(Xtr_rows, copy=False)
    mean = Xtr_rows.mean(0); scale = Xtr_rows.std(0); scale[scale < 1e-9] = 1.0
    del Xtr_rows

    def seq_tensor(s0, s1, cap=0):
        Xs = Xwork[s0:s1][:, keepwork].astype(np.float32)
        np.nan_to_num(Xs, copy=False)
        Xs = np.clip((Xs - mean) / scale, -8.0, 8.0)
        if cap and Xs.shape[0] > cap:
            Xs = Xs[:cap]
        return torch.from_numpy(Xs)

    train_idx = [i for i in range(len(series)) if not is_val[i]]
    val_idx = [i for i in range(len(series)) if is_val[i]]
    train_idx.sort(key=lambda i: series[i][1] - series[i][0])

    class AttnNet(nn.Module):
        def __init__(self, F, d, heads, layers, ff, dropout):
            super().__init__()
            self.proj = nn.Linear(F, d)
            enc = nn.TransformerEncoderLayer(
                d, heads, dim_feedforward=ff, dropout=dropout,
                batch_first=True, activation="gelu", norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(d, 1)

        def forward(self, x, lengths):
            B, T, _ = x.shape
            h = self.proj(x)
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            key_pad = (torch.arange(T, device=x.device)[None, :]
                       >= lengths[:, None].to(x.device))
            h = self.enc(h, mask=causal, src_key_padding_mask=key_pad)
            return self.head(self.drop(h)).squeeze(-1)

    net = AttnNet(F, args.dmodel, args.heads, args.layers, ff, args.dropout).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"params={n_par:,}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    yv = y[va].astype(np.int64); tv = t[va]; sv = sid[va]

    def eval_val():
        net.eval()
        if dev == "mps":
            torch.mps.empty_cache()
        val_pred = np.zeros(len(sid), dtype=np.float64)
        # one series at a time: attention is O(T^2) and MPS caches freed blocks,
        # so batching long (T up to 1000) VAL series accumulates GBs. B=1 keeps the
        # peak buffer tiny (1x heads x T^2) and avoids padding waste.
        with torch.no_grad():
            for n, i in enumerate(val_idx):
                s0, s1 = series[i]
                xb = seq_tensor(s0, s1).unsqueeze(0).to(dev)
                lengths = torch.tensor([s1 - s0])
                logits = net(xb, lengths).cpu().numpy()[0, : s1 - s0]
                val_pred[s0:s1] = logits
                del xb
                if dev == "mps" and (n & 255) == 0:
                    torch.mps.empty_cache()
        return val_pred[va]

    best = (-1.0, None, -1)
    for ep in range(args.epochs):
        net.train(); t0 = time.time()
        order = train_idx[:]
        rng = np.random.default_rng(1000 + ep)
        groups = [order[k:k + args.batch] for k in range(0, len(order), args.batch)]
        rng.shuffle(groups)
        tot = 0.0
        for g in groups:
            seqs = [seq_tensor(*series[i], cap=args.maxlen) for i in g]
            labels, weights = [], []
            for i in g:
                s0, s1 = series[i]
                yy = y[s0:s1].astype(np.float32)
                if args.maxlen and len(yy) > args.maxlen:
                    yy = yy[:args.maxlen]
                labels.append(torch.from_numpy(yy))
                weights.append(torch.from_numpy(ramp_weight(yy)))
            lengths = torch.tensor([s.shape[0] for s in seqs])
            xb = pad_sequence(seqs, batch_first=True).to(dev)
            yb = pad_sequence(labels, batch_first=True).to(dev)
            wb = pad_sequence(weights, batch_first=True).to(dev)
            maxT = xb.shape[1]
            m = (torch.arange(maxT, device=dev)[None, :] < lengths[:, None].to(dev)).float()
            logits = net(xb, lengths)
            if args.loss == "bce":
                loss = (bce(logits, yb) * wb * m).sum() / m.sum()
            else:
                # within-step pairwise RankNet: at each sequence position p (= the
                # online step t shared by every series in the batch) push positive
                # series above negative ones. -log sigmoid(s_pos - s_neg) summed over
                # all pos/neg pairs at p, ramp-weighted, is a differentiable surrogate
                # for the within-step Mann-Whitney AUC that TS-AUC averages.
                ylab = yb[:, 0]                                # (B,) per-series label
                pos = (ylab > 0.5).float()[:, None]           # (B,1)
                neg = (ylab < 0.5).float()[:, None]           # (B,1)
                pairmask = (pos @ neg.T)[:, :, None]          # (B,B,1) i=pos j=neg
                vp = m[:, None, :] * m[None, :, :]            # (B,B,T) both alive at p
                rampw = wb[None, :, :]                        # ramp from the neg row
                w = pairmask * vp * rampw                     # (B,B,T) pair weights
                diff = logits[:, None, :] - logits[None, :, :]  # s_i - s_j  (B,B,T)
                loss = (w * F_.softplus(-diff)).sum() / (w.sum() + 1e-8)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(g)
        sched.step()
        vl = eval_val()
        vauc = ts_auc_grouped(tv, yv, vl)
        print(f"  epoch {ep:2d}  loss {tot/len(train_idx):.4f}  "
              f"VAL TS-AUC {vauc:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        if vauc > best[0]:
            best = (vauc, vl.copy(), ep)

    print(f"\nBest VAL TS-AUC {best[0]:.4f} @ epoch {best[2]}")
    attn = best[1]
    np.savez(args.out, val_logit=attn, series_id=sv, t_online=tv)
    print(f"saved VAL logit -> {args.out}")

    # ---- blend gate from CACHED logits (no GBT/GRU retrain) ----
    base = np.load("features/val_base_logits.npz")
    assert np.array_equal(base["series_id"], sv) and np.array_equal(base["t_online"], tv)
    mean4 = base["gbt_logits"].mean(axis=1)
    log_logit = base["log_logit"]
    rk = np.load("features/val_rank_logits.npz")
    rk_key = {(int(a), int(bb)): float(v) for a, bb, v in
              zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank_score = np.array([rk_key[(int(a), int(bb))] for a, bb in zip(sv, tv)])
    grus = [np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"]
            for m in ("020", "021", "022")]
    gru_mean = np.mean(grus, axis=0)
    hA = half_mask(sv)

    def blend(neural):
        rag = (rank_score - rank_score.mean()) / (rank_score.std() + 1e-12)
        rag = rag * mean4.std() + mean4.mean()
        gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * rag
        b = (1 - W_LIN) * gbt5 + W_LIN * log_logit
        return (1 - W_GRU) * b + W_GRU * neural

    def rep(tag, neural):
        sh = blend(neural)
        f = ts_auc_grouped(tv, yv, sh)
        a = ts_auc_grouped(tv[hA], yv[hA], sh[hA])
        bb = ts_auc_grouped(tv[~hA], yv[~hA], sh[~hA])
        print(f"  [{tag}] full={f:.4f} halfA={a:.4f} halfB={bb:.4f}")
        return np.array([f, a, bb])

    az = (attn - attn.mean()) / (attn.std() + 1e-12)
    attn_g = az * gru_mean.std() + gru_mean.mean()   # rescale to GRU logit scale
    print(f"\nattn standalone VAL={best[0]:.4f}  rank-corr(attn,gru_mean)="
          f"{_spearman(attn, gru_mean):.3f}")
    r = rep("GRU x3 ", gru_mean)
    add = rep("+attn4 ", np.mean(grus + [attn_g], axis=0))
    print(f"Δ(add attn as 4th neural) full={add[0]-r[0]:+.4f} "
          f"halfA={add[1]-r[1]:+.4f} halfB={add[2]-r[2]:+.4f}")
    print("VERDICT:", "ATTN LIFTS BLEND" if (add - r > 0).all()
          else ("full+B up" if add[0] > r[0] and add[2] > r[2] else "no robust gain"))


if __name__ == "__main__":
    main()
