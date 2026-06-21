"""Train one strong candidate sequence model:
causal dilated TCN + monotonic cumulative hazard head.

Why this model:
- The target is "break happened by step t" (a cumulative event probability).
- A hazard head enforces monotonicity by construction:
    hazard_t in (0,1),  P(break<=t) = 1 - prod_{i<=t}(1-hazard_i)
- A causal TCN is GPU-friendly (T4), parallel over time, and can capture
  multi-scale temporal patterns via dilation without recurrence.

Training protocol mirrors existing sequence members:
- Split: fixed VAL series split (seed=42, 20%).
- Input: v4 feature stream with optional drop-set and optional no-logt.
- Loss: per-step BCE on cumulative probability, with elapsed-ramp weighting.
- Select: best epoch by VAL TS-AUC (competition metric), not loss.

Outputs:
- artifacts/models/model_040_tcn_hazard/tcn_hazard.pt
- artifacts/models/model_040_tcn_hazard/meta.json
- features/val_tcn_hazard_logits.npz (VAL monotonic logit aligned to VAL rows)

Run example:
  uv run python scripts/train_tcn_hazard.py \
      --channels 128 --levels 5 --kernel 3 --epochs 35 --seed 0 --no-logt
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
EPS = 1e-9
DROP = {
    "t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
    "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2",
}
OUTDIR = "artifacts/models/model_040_tcn_hazard"


def val_mask(sid: np.ndarray) -> np.ndarray:
    uniq = np.unique(sid)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    return np.isin(sid, list(val_ids))


def ramp_weight(y_seg: np.ndarray) -> np.ndarray:
    """Elapsed-ramp weight for one series segment (y is 0..0, then 1..1)."""
    w = np.ones(len(y_seg), dtype=np.float32)
    if y_seg.any():
        tau = int(y_seg.argmax())
        idx = np.arange(len(y_seg))
        w[tau:] = np.clip((idx[tau:] - tau) / RAMP, 0.2, 1.0)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=V4)
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--val-out", default="features/val_tcn_hazard_logits.npz")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--raw", action="store_true",
                    help="use all features (no DROP); for controlled probes")
    ap.add_argument("--no-logt", action="store_true",
                    help="drop log_t from inputs")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = args.device

    print(
        f"device={dev} channels={args.channels} levels={args.levels} "
        f"kernel={args.kernel} epochs={args.epochs}"
    )

    d = np.load(args.features, allow_pickle=True)
    X = d["X"]
    y = d["y"].astype(np.int64)
    sid = d["series_id"].astype(np.int64)
    t = d["t_online"].astype(np.int64)
    names = [str(n) for n in d["feature_names"]]

    drop = set() if args.raw else set(DROP)
    if args.no_logt:
        drop = drop | {"log_t"}
    keep = [i for i, n in enumerate(names) if n not in drop]
    keep_names = [names[i] for i in keep]

    va = val_mask(sid)
    yv = y[va].astype(np.int64)
    tv = t[va]

    # Series boundaries (X is sorted by sid, t).
    uniq, start = np.unique(sid, return_index=True)
    bounds = list(start) + [len(sid)]
    series = [(int(start[i]), int(bounds[i + 1])) for i in range(len(uniq))]
    is_val = np.array([bool(va[s0]) for s0, _ in series])

    # Frozen z-score stats on TRAIN rows only.
    Xtr = X[~va][:, keep].astype(np.float32)
    np.nan_to_num(Xtr, copy=False)
    mean = Xtr.mean(0)
    scale = Xtr.std(0)
    scale[scale < 1e-9] = 1.0
    del Xtr

    def seq_tensor(s0: int, s1: int):
        xs = X[s0:s1][:, keep].astype(np.float32)
        np.nan_to_num(xs, copy=False)
        xs = np.clip((xs - mean) / scale, -8.0, 8.0)
        return torch.from_numpy(xs)

    train_idx = [i for i in range(len(series)) if not is_val[i]]
    val_idx = [i for i in range(len(series)) if is_val[i]]
    train_idx.sort(key=lambda i: series[i][1] - series[i][0])

    class CausalConv1d(nn.Module):
        def __init__(self, c_in: int, c_out: int, k: int, dila: int):
            super().__init__()
            self.pad = (k - 1) * dila
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=dila)

        def forward(self, x):
            # x: [B, C, T], left-pad only to keep causality.
            x = F.pad(x, (self.pad, 0))
            return self.conv(x)

    class TCNBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, k: int, dila: int, drop: float):
            super().__init__()
            self.c1 = CausalConv1d(c_in, c_out, k, dila)
            self.c2 = CausalConv1d(c_out, c_out, k, dila)
            self.act = nn.GELU()
            self.drop = nn.Dropout(drop)
            self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

        def forward(self, x):
            r = self.skip(x)
            z = self.drop(self.act(self.c1(x)))
            z = self.drop(self.act(self.c2(z)))
            return z + r

    class TCNHazard(nn.Module):
        def __init__(self, f_in: int, c: int, levels: int, k: int, drop: float):
            super().__init__()
            self.in_proj = nn.Conv1d(f_in, c, 1)
            blocks = []
            for i in range(levels):
                blocks.append(TCNBlock(c, c, k, 2 ** i, drop))
            self.blocks = nn.Sequential(*blocks)
            self.hazard_head = nn.Conv1d(c, 1, 1)

        def forward(self, x):
            # x: [B, T, F] -> hazard logits [B, T]
            z = x.transpose(1, 2)
            z = self.in_proj(z)
            z = self.blocks(z)
            hz = self.hazard_head(z).squeeze(1)
            return hz

        @staticmethod
        def hazard_to_cumprob(hazard_logits):
            # hazard_t = sigmoid(h_t)
            # S_t = prod_{i<=t}(1-h_i)
            # P_t = 1 - S_t (monotonic non-decreasing)
            log_surv = torch.cumsum(F.logsigmoid(-hazard_logits), dim=1)
            surv = torch.exp(log_surv)
            p = 1.0 - surv
            return torch.clamp(p, 1e-7, 1.0 - 1e-7)

        @staticmethod
        def cumprob_to_logit(p):
            return torch.log(p / (1.0 - p))

    net = TCNHazard(
        f_in=len(keep),
        c=args.channels,
        levels=args.levels,
        k=args.kernel,
        drop=args.dropout,
    ).to(dev)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def eval_val():
        net.eval()
        val_logit = np.zeros(len(sid), dtype=np.float64)
        with torch.no_grad():
            B = max(16, min(128, args.batch * 2))
            for k0 in range(0, len(val_idx), B):
                chunk = val_idx[k0:k0 + B]
                seqs = [seq_tensor(*series[i]) for i in chunk]
                lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.int64)
                xb = pad_sequence(seqs, batch_first=True).to(dev)
                hz = net(xb)
                p = net.hazard_to_cumprob(hz)
                l = net.cumprob_to_logit(p).cpu().numpy()
                for j, i in enumerate(chunk):
                    s0, s1 = series[i]
                    val_logit[s0:s1] = l[j, : s1 - s0]
        return val_logit[va]

    best_auc = -1.0
    best_state = None
    best_ep = -1

    for ep in range(args.epochs):
        t0 = time.time()
        net.train()

        order = train_idx[:]
        rng = np.random.default_rng(1000 + ep)
        groups = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        rng.shuffle(groups)

        total_loss = 0.0
        n_batches = 0

        for g in groups:
            seqs = [seq_tensor(*series[i]) for i in g]
            labels, weights = [], []
            for i in g:
                s0, s1 = series[i]
                yy = y[s0:s1].astype(np.float32)
                labels.append(torch.from_numpy(yy))
                weights.append(torch.from_numpy(ramp_weight(yy)))

            lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.int64)
            xb = pad_sequence(seqs, batch_first=True).to(dev)
            yb = pad_sequence(labels, batch_first=True).to(dev)
            wb = pad_sequence(weights, batch_first=True).to(dev)

            Tmax = xb.shape[1]
            mask = (torch.arange(Tmax, device=dev)[None, :] < lengths[:, None].to(dev)).float()

            hz = net(xb)
            p = net.hazard_to_cumprob(hz)
            loss = (F.binary_cross_entropy(p, yb, reduction="none") * wb * mask).sum()
            loss = loss / (mask.sum() + 1e-9)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        sched.step()
        val_logit = eval_val()
        val_auc = ts_auc_grouped(tv, yv, val_logit)

        if val_auc > best_auc:
            best_auc = val_auc
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        print(
            f"ep={ep + 1:02d}/{args.epochs} "
            f"loss={total_loss / max(1, n_batches):.5f} "
            f"val_ts_auc={val_auc:.5f} "
            f"best={best_auc:.5f}@{best_ep + 1} "
            f"dt={time.time() - t0:.1f}s"
        )

    assert best_state is not None
    net.load_state_dict(best_state)
    val_logit = eval_val()
    final_auc = ts_auc_grouped(tv, yv, val_logit)

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "tcn_hazard.pt")
    torch.save(
        {
            "state_dict": best_state,
            "in_features": len(keep),
            "keep_names": keep_names,
            "mean": mean.astype(np.float32),
            "scale": scale.astype(np.float32),
            "channels": args.channels,
            "levels": args.levels,
            "kernel": args.kernel,
            "dropout": args.dropout,
            "seed": args.seed,
            "no_logt": bool(args.no_logt),
            "raw": bool(args.raw),
        },
        ckpt_path,
    )

    np.savez_compressed(
        args.val_out,
        val_logit=val_logit.astype(np.float64),
        t_online=tv.astype(np.int64),
        y=yv.astype(np.int64),
    )

    meta = {
        "model_id": os.path.basename(args.out),
        "family": "tcn_hazard_monotonic",
        "best_epoch": int(best_ep + 1),
        "best_val_ts_auc": float(best_auc),
        "final_val_ts_auc": float(final_auc),
        "features": int(len(keep)),
        "channels": int(args.channels),
        "levels": int(args.levels),
        "kernel": int(args.kernel),
        "dropout": float(args.dropout),
        "seed": int(args.seed),
        "no_logt": bool(args.no_logt),
        "raw": bool(args.raw),
        "checkpoint": ckpt_path,
        "val_logits": args.val_out,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== DONE ===")
    print(f"best epoch: {best_ep + 1}")
    print(f"best VAL TS-AUC: {best_auc:.5f}")
    print(f"saved checkpoint: {ckpt_path}")
    print(f"saved val logits: {args.val_out}")


if __name__ == "__main__":
    main()
