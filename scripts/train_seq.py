"""Train a deterministic GRU sequence member over the per-series v4 calibrated
feature stream, for a decorrelated ensemble member (the highest-EV next step:
the GBTs are saturated at corr 0.97-0.99; a recurrent net is a different
inductive bias that integrates evidence over time).

  * Input  : v4 calibrated features (minus pure-time/chi2 drop set), z-scored
             with frozen train-split stats, NaN->0, clipped.
  * Model  : 1-layer GRU (H hidden) -> Linear -> per-step logit P(broken by t).
  * Loss   : per-step BCEWithLogits, elapsed-ramp positive weighting (same as
             the GBTs: w = clip((i-tau)/RAMP, 0.2, 1)).
  * Select : epoch with best VAL TS-AUC (the competition metric), not loss.

Training runs in torch (MPS/CPU); only INFERENCE must be deterministic, which we
guarantee by exporting the weights to numpy and re-implementing the GRU forward
in pure numpy (parity-checked here, max|diff| printed). Saves:
  artifacts/models/model_020_gru/gru.npz   (numpy weights + input stats + names)
  features/val_seq_logits.npz              (VAL neural logit, aligned to va order)

Run: uv run python scripts/train_seq.py --hidden 96 --epochs 30
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

V4 = "features/train_features_v4.npz"
SPLIT_SEED = 42
RAMP = 50.0
DROP = {"t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
        "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2"}
OUTDIR = "artifacts/models/model_020_gru"


def val_mask(sid):
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids))


def ramp_weight(y_seg):
    """elapsed-ramp weight for one series segment (y_seg is 0..0 1..1)."""
    w = np.ones(len(y_seg), dtype=np.float32)
    if y_seg.any():
        tau = int(y_seg.argmax())
        idx = np.arange(len(y_seg))
        w[tau:] = np.clip((idx[tau:] - tau) / RAMP, 0.2, 1.0)
    return w


def numpy_gru_forward(X, p):
    """Pure-numpy GRU + linear head over a (T, F) sequence -> (T,) logits.

    Matches torch.nn.GRU (single layer) exactly. p holds the exported arrays.
    """
    Wih, Whh = p["Wih"], p["Whh"]          # (3H,F), (3H,H)
    bih, bhh = p["bih"], p["bhh"]          # (3H,), (3H,)
    Wo, bo = p["Wo"], p["bo"]              # (H,), ()
    mean, scale = p["mean"], p["scale"]
    H = Whh.shape[1]
    Xs = np.clip((X - mean) / scale, -8.0, 8.0)
    np.nan_to_num(Xs, copy=False)
    h = np.zeros(H, dtype=np.float64)
    out = np.empty(len(Xs), dtype=np.float64)
    Wir, Wiz, Win = Wih[:H], Wih[H:2 * H], Wih[2 * H:]
    Whr, Whz, Whn = Whh[:H], Whh[H:2 * H], Whh[2 * H:]
    bir, biz, bin_ = bih[:H], bih[H:2 * H], bih[2 * H:]
    bhr, bhz, bhn = bhh[:H], bhh[H:2 * H], bhh[2 * H:]
    for i in range(len(Xs)):
        x = Xs[i]
        r = 1.0 / (1.0 + np.exp(-(Wir @ x + bir + Whr @ h + bhr)))
        z = 1.0 / (1.0 + np.exp(-(Wiz @ x + biz + Whz @ h + bhz)))
        n = np.tanh(Win @ x + bin_ + r * (Whn @ h + bhn))
        h = (1.0 - z) * n + z * h
        out[i] = Wo @ h + bo
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=OUTDIR,
                    help="output dir for gru.npz/meta.json")
    ap.add_argument("--val-out", default="features/val_seq_logits.npz",
                    help="where to cache the VAL neural logit")
    args = ap.parse_args()
    out_dir = args.out
    val_out = args.val_out

    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = args.device
    print(f"device={dev} hidden={args.hidden} epochs={args.epochs}")

    d = np.load(V4, allow_pickle=True)
    X, y, sid, t = d["X"], d["y"], d["series_id"], d["t_online"]
    names = [str(n) for n in d["feature_names"]]
    keep = [i for i, n in enumerate(names) if n not in DROP]
    keep_names = [names[i] for i in keep]
    F = len(keep)
    print(f"X={X.shape} -> using {F} features")

    va = val_mask(sid)
    # series boundaries (matrix is sorted by (sid, t))
    uniq, start = np.unique(sid, return_index=True)
    bounds = list(start) + [len(sid)]
    series = [(int(start[i]), int(bounds[i + 1])) for i in range(len(uniq))]
    is_val = np.array([bool(va[s0]) for s0, _ in series])

    # frozen standardization on TRAIN rows
    Xtr_rows = X[~va][:, keep].astype(np.float32)
    np.nan_to_num(Xtr_rows, copy=False)
    mean = Xtr_rows.mean(0)
    scale = Xtr_rows.std(0)
    scale[scale < 1e-9] = 1.0
    del Xtr_rows

    def seq_tensor(s0, s1):
        Xs = X[s0:s1][:, keep].astype(np.float32)
        np.nan_to_num(Xs, copy=False)
        Xs = np.clip((Xs - mean) / scale, -8.0, 8.0)
        return torch.from_numpy(Xs)

    train_idx = [i for i in range(len(series)) if not is_val[i]]
    val_idx = [i for i in range(len(series)) if is_val[i]]
    # bucket train series by length for efficient padded batches
    train_idx.sort(key=lambda i: series[i][1] - series[i][0])

    class GRUNet(nn.Module):
        def __init__(self, F, H, dropout):
            super().__init__()
            self.gru = nn.GRU(F, H, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(H, 1)

        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            return self.head(self.drop(out)).squeeze(-1)

    net = GRUNet(F, args.hidden, args.dropout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    # precompute val labels/steps aligned to va order for TS-AUC
    yv = y[va].astype(np.int64)
    tv = t[va]

    def eval_val():
        net.eval()
        val_pred = np.zeros(len(sid), dtype=np.float64)
        with torch.no_grad():
            B = 128
            for k in range(0, len(val_idx), B):
                chunk = val_idx[k:k + B]
                seqs = [seq_tensor(*series[i]) for i in chunk]
                lengths = torch.tensor([s.shape[0] for s in seqs])
                xb = pad_sequence(seqs, batch_first=True).to(dev)
                logits = net(xb, lengths).cpu().numpy()
                for j, i in enumerate(chunk):
                    s0, s1 = series[i]
                    val_pred[s0:s1] = logits[j, : s1 - s0]
        return val_pred[va]

    best = (-1.0, None, -1)
    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        order = train_idx[:]
        # shuffle in length-buckets: group then shuffle group order (keeps pad low)
        rng = np.random.default_rng(1000 + ep)
        groups = [order[k:k + args.batch] for k in range(0, len(order), args.batch)]
        rng.shuffle(groups)
        tot = 0.0
        for g in groups:
            seqs = [seq_tensor(*series[i]) for i in g]
            labels, weights = [], []
            for i in g:
                s0, s1 = series[i]
                yy = y[s0:s1].astype(np.float32)
                labels.append(torch.from_numpy(yy))
                weights.append(torch.from_numpy(ramp_weight(yy)))
            lengths = torch.tensor([s.shape[0] for s in seqs])
            xb = pad_sequence(seqs, batch_first=True).to(dev)
            yb = pad_sequence(labels, batch_first=True).to(dev)
            wb = pad_sequence(weights, batch_first=True).to(dev)
            maxT = xb.shape[1]
            mask = (torch.arange(maxT, device=dev)[None, :] < lengths[:, None].to(dev)).float()
            logits = net(xb, lengths)
            loss = (bce(logits, yb) * wb * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(g)
        sched.step()
        val_logit = eval_val()
        vauc = ts_auc_grouped(tv, yv, val_logit)
        print(f"  epoch {ep:2d}  loss {tot/len(train_idx):.4f}  "
              f"VAL TS-AUC {vauc:.4f}  ({time.time()-t0:.1f}s)")
        if vauc > best[0]:
            sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best = (vauc, sd, ep)

    print(f"\nBest VAL TS-AUC {best[0]:.4f} @ epoch {best[2]}")
    net.load_state_dict(best[1])

    # export numpy weights
    sd = net.state_dict()
    p = dict(
        Wih=sd["gru.weight_ih_l0"].cpu().numpy().astype(np.float64),
        Whh=sd["gru.weight_hh_l0"].cpu().numpy().astype(np.float64),
        bih=sd["gru.bias_ih_l0"].cpu().numpy().astype(np.float64),
        bhh=sd["gru.bias_hh_l0"].cpu().numpy().astype(np.float64),
        Wo=sd["head.weight"].cpu().numpy()[0].astype(np.float64),
        bo=float(sd["head.bias"].cpu().numpy()[0]),
        mean=mean.astype(np.float64), scale=scale.astype(np.float64),
    )

    # parity check: numpy forward vs torch on a few val series
    net.eval()
    maxdiff = 0.0
    with __import__("torch").no_grad():
        for i in val_idx[:5]:
            s0, s1 = series[i]
            Xseq = X[s0:s1][:, keep].astype(np.float64)
            npy = numpy_gru_forward(Xseq, p)
            seqs = [seq_tensor(s0, s1)]
            lengths = __import__("torch").tensor([s1 - s0])
            xb = pad_sequence(seqs, batch_first=True).to(dev)
            tch = net(xb, lengths).cpu().numpy()[0, : s1 - s0]
            maxdiff = max(maxdiff, float(np.abs(npy - tch).max()))
    print(f"numpy<->torch parity max|diff| = {maxdiff:.2e}")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "gru.npz"),
             feature_names=np.array(keep_names), hidden=args.hidden, **p)
    # cache VAL neural logit (numpy forward, the deterministic one we ship)
    val_pred = np.zeros(len(sid), dtype=np.float64)
    for i in val_idx:
        s0, s1 = series[i]
        val_pred[s0:s1] = numpy_gru_forward(X[s0:s1][:, keep].astype(np.float64), p)
    np.savez(val_out, val_logit=val_pred[va])
    final_auc = ts_auc_grouped(tv, yv, val_pred[va])
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(dict(val_ts_auc=round(float(final_auc), 5), hidden=args.hidden,
                       epochs=args.epochs, best_epoch=best[2], n_features=F,
                       parity_maxdiff=maxdiff), fh, indent=2)
    print(f"numpy VAL TS-AUC {final_auc:.4f}  saved -> {out_dir}/gru.npz")


if __name__ == "__main__":
    main()
