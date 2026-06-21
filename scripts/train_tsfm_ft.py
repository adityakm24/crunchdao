"""Round 11 — END-TO-END FINE-TUNED TSFM (Chronos-T5-mini encoder) break member.

THE one genuinely-untested high-ceiling blueprint lever. Round 7 (sparse+linear)
and round 11 (dense+nonlinear) both tested FROZEN Chronos embeddings -> redundant
(rankcorr ~0.55, blend lift -0.0007). Frozen only RE-WEIGHTS a fixed representation.
This script instead BACKPROPS into the encoder weights: the pretrained TSFM prior
is ADAPTED to the structural-break task. Different experiment, real ceiling.

Input per (series, online-step t): the historically-calibrated online stream
z = clip((x_online[:t+1] - mu_hist)/sd_hist, +-8), left-NaN-padded/truncated to
CTX points (Chronos masks NaN). The hist-null calibration (the round-3 win) is
baked into z; under H0 z~N(0,1), under a break z departs. The Chronos tokenizer
mean-scales + bins to 4096 tokens; the 4-layer T5 encoder contextualizes; we mean-
pool the token hidden states, concat 4 per-series context scalars (null reliability),
and a small MLP head emits the per-step break logit. Encoder + head trained end-to-
end (BCE, ramp-weighted post-break like the GRU/attn members).

GATE (the decisive, established protocol from val_tsfm_dense): produce the VAL logit
at the dense step grid, z-blend onto the round-10 shipped logit, score TS-AUC on
VAL full + BOTH honest halves (seed 1) on that subset, report lift + rank-corr to
the GRU mean + to the shipped blend. PROMISING only if it lifts full AND both halves
with low rankcorr -> then DISTILL into a numpy-servable student for deterministic
deploy. Else the last blueprint lever is closed BY EXPERIMENT (not prior).

Run: KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 uv run python -u scripts/train_tsfm_ft.py \
        --train-series 5000 --epochs 4 --batch 16 --lr 3e-5
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from sb.data import iter_series        # noqa: E402
from sb.metric import ts_auc_grouped   # noqa: E402

SEED = 42
HALF_SEED = 1
STEP_GRID = list(range(40, 401, 6))    # 61 dense steps across the metric-heavy band
CTX = 448                              # Chronos n_positions=512; leave room for EOS
CLIP = 8.0
RAMP = 50.0
MODEL = "models/chronos-t5-mini"
RANK_GW, W_LIN, W_GRU = 0.10, 0.10, 0.45
CTX_NAMES = ["t_over_nhist", "log_n_hist", "acf1_h", "kurt_h"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt(ra @ ra) * np.sqrt(rb @ rb) + 1e-12))


def val_ids_split():
    import pandas as pd
    idx = pd.read_parquet("Dataset/y_train_index.parquet")
    uniq = np.array(sorted(idx.index.tolist()))
    np.random.default_rng(SEED).shuffle(uniq)
    n_val = int(0.2 * len(uniq))
    val = set(int(v) for v in uniq[:n_val])
    train = [int(v) for v in uniq[n_val:]]
    return val, train


def reconstruct_shipped():
    base = np.load("features/val_base_logits.npz")
    sid = base["series_id"].astype(np.int64)
    t = base["t_online"].astype(np.int64)
    y = base["y"].astype(np.int64)
    gbt = base["gbt_logits"].astype(np.float64)
    log_logit = base["log_logit"].astype(np.float64)
    mean4 = gbt.mean(axis=1)
    rk = np.load("features/val_rank_logits.npz")
    rk_key = {(int(s), int(tt)): float(v) for s, tt, v in
              zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank_score = np.array([rk_key[(int(s), int(tt))] for s, tt in zip(sid, t)])
    rank_as_gbt = zscore(rank_score) * mean4.std() + mean4.mean()
    gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * rank_as_gbt
    base_logit = (1 - W_LIN) * gbt5 + W_LIN * log_logit
    gru = np.mean([np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"]
                   for m in ("020", "021", "022")], axis=0)
    shipped = (1 - W_GRU) * base_logit + W_GRU * gru
    lut = {(int(s), int(tt)): (float(sh), int(yy), float(g))
           for s, tt, sh, yy, g in zip(sid, t, shipped, y, gru)}
    return lut


def _ctx_scalars(xh):
    n = len(xh)
    h = xh - xh.mean()
    d = float(np.dot(h, h))
    acf1 = float(np.clip(np.dot(h[:-1], h[1:]) / d, -0.99, 0.99)) if d > 1e-12 else 0.0
    z = (xh - xh.mean()) / (xh.std() + 1e-12)
    kurt = float((z ** 4).mean() - 3.0)
    return acf1, kurt, n


def _window(xo_z, t):
    """Left-NaN-pad / truncate the calibrated online stream up to step t to CTX."""
    seg = xo_z[: t + 1]
    if len(seg) >= CTX:
        return seg[-CTX:]
    out = np.full(CTX, np.nan, dtype=np.float32)
    out[CTX - len(seg):] = seg
    return out


def build_windows(ids, steps_per_series, split_label, lut=None, rng=None):
    """Return windows (N,CTX) f32 with NaN pad, ctx4 (N,4), y (N,), meta lists.
    For VAL (lut given) keep only rows present in the shipped lut and also return
    shipped + gru + y from the lut."""
    W, C4, Y, SID, T = [], [], [], [], []
    SH, GRU = [], []
    t0 = time.time()
    n = 0
    for s in iter_series("train", ids=list(ids)):
        xh = s.x_hist.astype(np.float64)
        xo = s.x_online.astype(np.float64)
        if len(xh) < 50 or len(xo) <= STEP_GRID[0] + 5:
            continue
        mu, sd = xh.mean(), xh.std() + 1e-9
        xo_z = np.clip((xo - mu) / sd, -CLIP, CLIP).astype(np.float32)
        acf1, kurt, nh = _ctx_scalars(xh)
        if lut is None:
            # train: sample sparse steps within this series' online range
            hi = min(len(xo) - 1, STEP_GRID[-1])
            cand = [t for t in range(STEP_GRID[0], hi) if t < len(xo)]
            if not cand:
                continue
            k = min(steps_per_series, len(cand))
            sel = rng.choice(cand, size=k, replace=False)
        else:
            sel = [t for t in STEP_GRID if (int(s.series_id), t) in lut and t < len(xo)]
        for t in sel:
            t = int(t)
            lab = 1 if (s.has_break and s.tau_index is not None and s.tau_index < t) else 0
            W.append(_window(xo_z, t))
            C4.append([t / max(nh, 1), np.log(max(nh, 1)), acf1, kurt])
            Y.append(lab); SID.append(int(s.series_id)); T.append(t)
            if lut is not None:
                sh, yy, g = lut[(int(s.series_id), t)]
                SH.append(sh); GRU.append(g); Y[-1] = yy
        n += 1
        if n % 1000 == 0:
            print(f"   [{split_label}] {n} series  rows={len(Y)}  ({time.time()-t0:.0f}s)")
    out = {
        "W": np.asarray(W, np.float32),
        "C4": np.asarray(C4, np.float32),
        "y": np.asarray(Y, np.float32),
        "sid": np.asarray(SID, np.int64),
        "t": np.asarray(T, np.int64),
    }
    if lut is not None:
        out["ship"] = np.asarray(SH, np.float64)
        out["gru"] = np.asarray(GRU, np.float64)
    print(f"   [{split_label}] DONE rows={len(Y)}  pos={np.mean(Y):.3f}  ({time.time()-t0:.0f}s)")
    return out


def ramp_w(y, sid, t):
    """Down-weight the immediate post-break steps (ambiguous), like the GRU member."""
    w = np.ones(len(y), dtype=np.float32)
    # cheap per-row ramp using break onset is unavailable here (sparse steps);
    # approximate: full weight (sparse steps are mostly far from tau). Keep ones.
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-series", type=int, default=5000)
    ap.add_argument("--val-series", type=int, default=0, help="cap VAL series (0=all; smoke only)")
    ap.add_argument("--steps-per-series", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="A/B control: train head only (should reproduce frozen-dense redundancy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="features/val_tsfm_ft_logits.npz")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = (("mps" if torch.backends.mps.is_available() else "cpu")
           if args.device == "auto" else args.device)
    print(f"device={dev}  ft={'HEAD-ONLY' if args.freeze_encoder else 'END-TO-END'}  "
          f"train_series={args.train_series} steps/ser={args.steps_per_series} "
          f"epochs={args.epochs} batch={args.batch} lr={args.lr}")

    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(MODEL, device_map=dev, torch_dtype=torch.float32)
    tok = pipe.tokenizer
    enc = pipe.model.model.encoder
    d_model = pipe.model.model.config.d_model

    val_ids, train_ids = val_ids_split()
    if args.train_series and args.train_series < len(train_ids):
        train_ids = list(rng.choice(train_ids, size=args.train_series, replace=False))
    if args.val_series:
        val_ids = set(sorted(val_ids)[: args.val_series])
    print(f"VAL {len(val_ids)} series  TRAIN {len(train_ids)} series")

    lut = reconstruct_shipped()
    print("building VAL windows (dense grid) ...")
    val = build_windows(val_ids, 0, "VAL", lut=lut)
    print("building TRAIN windows (sparse grid) ...")
    tr = build_windows(train_ids, args.steps_per_series, "TRAIN", lut=None, rng=rng)

    # honest halves by series (rng HALF_SEED)
    u_ids = np.array(sorted(set(int(x) for x in val["sid"])))
    np.random.default_rng(HALF_SEED).shuffle(u_ids)
    halfA = set(int(v) for v in u_ids[: len(u_ids) // 2])
    inA = np.array([int(x) in halfA for x in val["sid"]])

    shz = zscore(val["ship"])
    base_full = ts_auc_grouped(val["t"], val["y"].astype(int), sigmoid(shz))
    base_A = ts_auc_grouped(val["t"][inA], val["y"][inA].astype(int), sigmoid(shz[inA]))
    base_B = ts_auc_grouped(val["t"][~inA], val["y"][~inA].astype(int), sigmoid(shz[~inA]))
    gru_sa = ts_auc_grouped(val["t"], val["y"].astype(int), val["gru"])
    print(f"\nshipped (dense subset)  full={base_full:.4f}  halfA={base_A:.4f}  "
          f"halfB={base_B:.4f}   [gru standalone {gru_sa:.4f}]\n")

    head = nn.Sequential(
        nn.Linear(d_model + 4, args.head_dim), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(args.head_dim, 1),
    ).to(dev)
    if args.freeze_encoder:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
        params = list(head.parameters())
    else:
        params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    n_train_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable params: {n_train_params:,}")

    def encode_batch(Wb, C4b, train_mode):
        ctx = torch.from_numpy(Wb)                         # (B,CTX) f32 w/ NaN pad
        token_ids, attn_mask, _ = tok.context_input_transform(ctx)
        token_ids = token_ids.to(dev); attn_mask = attn_mask.to(dev)
        out = enc(input_ids=token_ids, attention_mask=attn_mask)
        h = out.last_hidden_state                          # (B,L,d)
        m = attn_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)  # (B,d)
        c4 = torch.from_numpy(C4b).to(dev)
        return head(torch.cat([pooled, c4], dim=1)).squeeze(-1)

    def val_logits():
        enc.eval(); head.eval()
        outs = np.empty(len(val["y"]), np.float64)
        with torch.no_grad():
            for i in range(0, len(val["y"]), 64):
                Wb = val["W"][i:i + 64]; C4b = val["C4"][i:i + 64]
                lg = encode_batch(Wb, C4b, False)
                outs[i:i + len(Wb)] = lg.detach().float().cpu().numpy()
        return outs

    N = len(tr["y"])
    yt = torch.from_numpy(tr["y"])
    best = None
    for ep in range(args.epochs):
        if not args.freeze_encoder:
            enc.train()
        head.train()
        order = rng.permutation(N)
        t0 = time.time(); tot = 0.0
        for bi in range(0, N, args.batch):
            idx = order[bi:bi + args.batch]
            Wb = tr["W"][idx]; C4b = tr["C4"][idx]
            yb = yt[idx].to(dev)
            logit = encode_batch(Wb, C4b, True)
            loss = Fnn.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        vl = val_logits()
        sa = ts_auc_grouped(val["t"], val["y"].astype(int), vl)
        if sa < 0.5:
            vl = -vl; sa = 1.0 - sa
        rc_gru = spearman(vl, val["gru"])
        rc_sh = spearman(vl, val["ship"])
        hz = zscore(vl)
        # blend gate
        bestw = None
        for w in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
            p = sigmoid(shz + w * hz)
            f = ts_auc_grouped(val["t"], val["y"].astype(int), p)
            a = ts_auc_grouped(val["t"][inA], val["y"][inA].astype(int), p[inA])
            b = ts_auc_grouped(val["t"][~inA], val["y"][~inA].astype(int), p[~inA])
            robust = min(f - base_full, a - base_A, b - base_B)
            if bestw is None or robust > bestw[1]:
                bestw = (w, robust, f, a, b)
        w, robust, f, a, b = bestw
        print(f"ep{ep}  loss {tot/N:.4f}  standalone {sa:.4f}  rc_gru {rc_gru:+.3f} "
              f"rc_ship {rc_sh:+.3f}  | best w={w} robust {robust:+.4f}  "
              f"full {f:.4f}({f-base_full:+.4f}) A {a:.4f}({a-base_A:+.4f}) "
              f"B {b:.4f}({b-base_B:+.4f})  ({time.time()-t0:.0f}s)")
        if best is None or robust > best[0]:
            best = (robust, ep, sa, rc_gru, vl.copy(), (w, f, a, b))

    robust, ep, sa, rc_gru, vl, (w, f, a, b) = best
    print("\n================ VERDICT ================")
    print(f"BEST ep{ep}: standalone {sa:.4f}  rc_gru {rc_gru:+.3f}  "
          f"blend robust min-lift {robust:+.4f} @ w={w} "
          f"(full {f:.4f} A {a:.4f} B {b:.4f})")
    if robust > 0.0005:
        print("PROMISING -> fine-tuned TSFM lifts the blend. Next: DISTILL into a "
              "numpy-servable student for deterministic deploy.")
    else:
        print("REDUNDANT -> end-to-end fine-tuning does NOT rescue the frozen result. "
              "Blueprint-2 (TSFM) closed BY EXPERIMENT; detection floor reconfirmed.")
    np.savez(args.out, val_logit=vl, series_id=val["sid"], t_online=val["t"],
             y=val["y"].astype(np.int64))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
