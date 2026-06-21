"""Deterministic float64 numpy serve for the causal-attention member (round 11, A).

Mirrors StreamingGRU/StreamingLSTM in the submission: O(t)-per-step causal self-
attention with a growing per-layer K/V cache (causal => past K/V are frozen, so
cacheable). Reproduces the EXACT train_attn.py eval forward:
  norm_first TransformerEncoderLayer = h + MHA(LN1(h)); h + FFN(LN2(h)),
  activation = exact erf-GELU, LayerNorm eps=1e-5 (biased var),
  MHA scale = 1/sqrt(head_dim), heads split d contiguously.
INPUT-CALIBRATION ORDER MATCHES train_attn EVAL (not the GRU): nan_to_num FIRST,
then (x-mean)/scale, then clip +-8  (NaN feature -> 0 -> (0-mean)/scale).

Usage:
  uv run python scripts/serve_attn.py parity  artifacts/models/model_041_attn/attn_s1.npz [n]
  uv run python scripts/serve_attn.py valgate artifacts/models/model_041_attn/attn_s1.npz [s2.npz ...]
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from scipy.special import erf as _erf
except Exception:  # pragma: no cover - cloud has scipy via scikit-learn
    def _erf(x):
        # high-accuracy fallback (Abramowitz-Stegun 7.1.26), |err|<1.5e-7
        x = np.asarray(x, dtype=np.float64)
        s = np.sign(x); ax = np.abs(x)
        tt = 1.0 / (1.0 + 0.3275911 * ax)
        poly = tt * (0.254829592 + tt * (-0.284496736 + tt * (
            1.421413741 + tt * (-1.453152027 + tt * 1.061405429))))
        return s * (1.0 - poly * np.exp(-ax * ax))

_SQRT2 = np.sqrt(2.0)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / _SQRT2))


def _layernorm(x, w, b, eps=1e-5):
    mu = x.mean()
    var = ((x - mu) ** 2).mean()          # biased var, matches torch LayerNorm
    return (x - mu) / np.sqrt(var + eps) * w + b


class _Cache:
    __slots__ = ("buf", "n")

    def __init__(self, d):
        self.buf = np.zeros((64, d), dtype=np.float64)
        self.n = 0

    def append(self, v):
        if self.n == self.buf.shape[0]:
            self.buf = np.concatenate([self.buf, np.zeros_like(self.buf)], axis=0)
        self.buf[self.n] = v
        self.n += 1

    @property
    def arr(self):
        return self.buf[: self.n]


class StreamingAttn:
    """O(t)-per-step causal Transformer member -> per-step logit (float64)."""

    CLIP = 8.0

    def __init__(self, p, name_to_col):
        F, d, heads, layers, ff = (int(v) for v in p["__config"])
        self.d, self.heads, self.layers = d, heads, layers
        self.hd = d // heads
        self.attn_scale = 1.0 / np.sqrt(self.hd)
        self.Wproj = np.ascontiguousarray(p["proj.weight"], dtype=np.float64)   # (d,F)
        self.bproj = np.asarray(p["proj.bias"], dtype=np.float64)
        self.Whead = np.asarray(p["head.weight"], dtype=np.float64)[0]          # (d,)
        self.bhead = float(np.asarray(p["head.bias"])[0])
        self.layer = []
        for l in range(layers):
            pre = f"enc.layers.{l}."
            self.layer.append({
                "inW": np.ascontiguousarray(p[pre + "self_attn.in_proj_weight"], np.float64),
                "inB": np.asarray(p[pre + "self_attn.in_proj_bias"], np.float64),
                "outW": np.ascontiguousarray(p[pre + "self_attn.out_proj.weight"], np.float64),
                "outB": np.asarray(p[pre + "self_attn.out_proj.bias"], np.float64),
                "l1W": np.ascontiguousarray(p[pre + "linear1.weight"], np.float64),
                "l1B": np.asarray(p[pre + "linear1.bias"], np.float64),
                "l2W": np.ascontiguousarray(p[pre + "linear2.weight"], np.float64),
                "l2B": np.asarray(p[pre + "linear2.bias"], np.float64),
                "n1W": np.asarray(p[pre + "norm1.weight"], np.float64),
                "n1B": np.asarray(p[pre + "norm1.bias"], np.float64),
                "n2W": np.asarray(p[pre + "norm2.weight"], np.float64),
                "n2B": np.asarray(p[pre + "norm2.bias"], np.float64),
            })
        self.mean = np.asarray(p["__mean"], dtype=np.float64)
        self.scale = np.asarray(p["__scale"], dtype=np.float64)
        self.keep = np.array(
            [name_to_col[n] for n in [str(s) for s in p["__feature_names"]]],
            dtype=np.int64)
        self.reset()

    def reset(self):
        self.Kc = [_Cache(self.d) for _ in range(self.layers)]
        self.Vc = [_Cache(self.d) for _ in range(self.layers)]

    def step(self, feats):
        d, hd = self.d, self.hd
        xs = feats[self.keep].astype(np.float64)
        np.nan_to_num(xs, copy=False)                 # NaN->0 FIRST (matches eval)
        xs = (xs - self.mean) / self.scale
        np.clip(xs, -self.CLIP, self.CLIP, out=xs)
        h = self.Wproj @ xs + self.bproj              # (d,)
        for l in range(self.layers):
            P = self.layer[l]
            a = _layernorm(h, P["n1W"], P["n1B"])
            qkv = P["inW"] @ a + P["inB"]             # (3d,)
            q = qkv[:d]; k = qkv[d:2 * d]; v = qkv[2 * d:]
            self.Kc[l].append(k); self.Vc[l].append(v)
            K = self.Kc[l].arr; V = self.Vc[l].arr    # (t,d)
            ctx = np.empty(d, dtype=np.float64)
            for hh in range(self.heads):
                sl = slice(hh * hd, (hh + 1) * hd)
                sc = (K[:, sl] @ q[sl]) * self.attn_scale   # (t,)
                sc -= sc.max()
                w = np.exp(sc); w /= w.sum()
                ctx[sl] = w @ V[:, sl]
            h = h + (P["outW"] @ ctx + P["outB"])
            f = _layernorm(h, P["n2W"], P["n2B"])
            h = h + (P["l2W"] @ _gelu(P["l1W"] @ f + P["l1B"]) + P["l2B"])
        return float(self.Whead @ h + self.bhead)


# ----------------------------------------------------------------------------- #
#  Parity + VAL re-gate harness (local only)
# ----------------------------------------------------------------------------- #
V4 = "features/train_features_v4.npz"
SPLIT_SEED = 42
DROP = {"t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
        "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2", "log_t"}
RANK_GW, W_LIN, W_GRU = 0.10, 0.10, 0.45


def _val_series():
    d = np.load(V4, allow_pickle=True)
    X, y, sid, t = d["X"], d["y"], d["series_id"], d["t_online"]
    names = [str(n) for n in d["feature_names"]]
    n2c = {n: i for i, n in enumerate(names)}
    uniq = np.unique(sid); rng = np.random.default_rng(SPLIT_SEED); rng.shuffle(uniq)
    val_ids = set(uniq[: int(0.2 * len(uniq))].tolist())
    va = np.isin(sid, list(val_ids))
    u2, start = np.unique(sid, return_index=True)
    bounds = list(start) + [len(sid)]
    series = [(int(start[i]), int(bounds[i + 1])) for i in range(len(u2))]
    is_val = np.array([bool(va[s0]) for s0, _ in series])
    return X, y, sid, t, n2c, series, is_val, va


def _build_torch(p):
    import torch
    import torch.nn as nn
    F, d, heads, layers, ff = (int(v) for v in p["__config"])

    class AttnNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(F, d)
            enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ff, dropout=0.1,
                                             batch_first=True, activation="gelu", norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            T = x.shape[1]
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            h = self.enc(self.proj(x), mask=causal)
            return self.head(h).squeeze(-1)

    net = AttnNet().double().eval()
    sd = {k: torch.from_numpy(np.asarray(p[k], np.float64))
          for k in net.state_dict().keys()}
    net.load_state_dict(sd)
    return net, torch


def parity(wpath, n=6):
    p = np.load(wpath, allow_pickle=True)
    X, y, sid, t, n2c, series, is_val, va = _val_series()
    sa = StreamingAttn(p, n2c)
    net, torch = _build_torch(p)
    mean, scale, keep = sa.mean, sa.scale, sa.keep
    vidx = [i for i in range(len(series)) if is_val[i]]
    rng = np.random.default_rng(0); rng.shuffle(vidx)
    worst = 0.0
    for i in vidx[:n]:
        s0, s1 = series[i]
        feats = X[s0:s1]
        # torch full-sequence forward (float64)
        Xs = feats[:, keep].astype(np.float64)
        np.nan_to_num(Xs, copy=False)
        Xs = np.clip((Xs - mean) / scale, -8.0, 8.0)
        with torch.no_grad():
            tl = net(torch.from_numpy(Xs).unsqueeze(0)).numpy()[0]
        # streaming numpy
        sa.reset()
        nl = np.array([sa.step(feats[k]) for k in range(s1 - s0)])
        diff = np.max(np.abs(tl - nl))
        worst = max(worst, diff)
        print(f"  series {i:5d}  T={s1-s0:4d}  max|torch-numpy|={diff:.2e}")
    print(f"\n>>> WORST per-step parity diff = {worst:.2e}  "
          f"({'PASS <1e-7' if worst < 1e-7 else 'FAIL'})")


def valgate(wpaths):
    import sys as _sys
    _sys.path.insert(0, "src")
    from sb.metric import ts_auc_grouped
    X, y, sid, t, n2c, series, is_val, va = _val_series()
    vidx = [i for i in range(len(series)) if is_val[i]]
    member_logits = []
    for wp in wpaths:
        p = np.load(wp, allow_pickle=True)
        sa = StreamingAttn(p, n2c)
        vl = np.zeros(len(sid), dtype=np.float64)
        for n, i in enumerate(vidx):
            s0, s1 = series[i]
            sa.reset()
            for k in range(s1 - s0):
                vl[s0 + k] = sa.step(X[s0 + k])
            if (n & 255) == 0:
                print(f"  [{wp.split('/')[-1]}] {n}/{len(vidx)} series", flush=True)
        member_logits.append(vl[va])
        np.savez(wp.replace(".npz", "_valserve.npz"),
                 val_logit=vl[va], series_id=sid[va], t_online=t[va])
    attn_serve = np.mean(member_logits, axis=0)

    # blend gate from cached logits
    base = np.load("features/val_base_logits.npz")
    yv = base["y"].astype(np.int64); tv = base["t_online"]; sv = base["series_id"]
    mean4 = base["gbt_logits"].mean(axis=1); log_logit = base["log_logit"]
    rk = np.load("features/val_rank_logits.npz")
    rkey = {(int(a), int(b)): float(v)
            for a, b, v in zip(rk["series_id"], rk["t_online"], rk["rank_score"])}
    rank = np.array([rkey[(int(a), int(b))] for a, b in zip(sv, tv)])
    grus = [np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"]
            for m in ("020", "021", "022")]
    gru = np.mean(grus, axis=0)
    uniq = np.unique(sv); rng = np.random.default_rng(1); rng.shuffle(uniq)
    hA = np.isin(sv, list(set(uniq[: len(uniq) // 2].tolist())))

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-12)

    def blend(neural):
        rag = z(rank) * mean4.std() + mean4.mean()
        gbt5 = (1 - RANK_GW) * mean4 + RANK_GW * rag
        b = (1 - W_LIN) * gbt5 + W_LIN * log_logit
        return (1 - W_GRU) * b + W_GRU * neural

    def trio(neural):
        s = blend(neural)
        return np.array([ts_auc_grouped(tv, yv, s),
                         ts_auc_grouped(tv[hA], yv[hA], s[hA]),
                         ts_auc_grouped(tv[~hA], yv[~hA], s[~hA])])

    assert np.array_equal(sv, sid[va]) and np.array_equal(tv, t[va]), "order mismatch"
    r = trio(gru)
    attn_g = z(attn_serve) * gru.std() + gru.mean()
    a4 = np.mean(grus + [attn_g], axis=0)
    d = trio(a4) - r
    print(f"\nshipped 3-GRU baseline: full={r[0]:.4f} A={r[1]:.4f} B={r[2]:.4f}")
    print(f"+attn(serve) as 4th:    full={d[0]+r[0]:.4f} A={d[1]+r[1]:.4f} B={d[2]+r[2]:.4f}")
    print(f"Δ full={d[0]:+.4f} A={d[1]:+.4f} B={d[2]:+.4f}   robust-min={min(d):+.4f}")
    ship = min(d) > 0.0005
    print("VERDICT:", "SHIP (robust>+0.0005)" if ship else "BELOW BAR")

    # Bake the blend rescale (A*x + B == attn_g) into the head so the served member,
    # averaged RAW with the GRUs, reproduces the verified a4 blend EXACTLY:
    #   mean_over_seeds(baked.step) = A*attn_serve + B = attn_g.
    # (B is a global additive constant => AUC-invariant; A = gru.std/attn.std is the
    #  scale that puts the attn logits on the GRU scale.)
    import os as _os
    A = gru.std() / (attn_serve.std() + 1e-12)
    B = float(gru.mean() - attn_serve.mean() * A)
    saved = []
    for wp in wpaths:
        p = np.load(wp, allow_pickle=True)
        out = {k: p[k] for k in p.files}
        out["head.weight"] = np.asarray(p["head.weight"], np.float64) * A
        out["head.bias"] = np.asarray(p["head.bias"], np.float64) * A + B
        sdir = _os.path.splitext(wp)[0] + "_serve"
        _os.makedirs(sdir, exist_ok=True)
        np.savez(_os.path.join(sdir, "attn.npz"), **out)
        # save-integrity: reload and confirm baked head + ALL original keys intact
        chk = np.load(_os.path.join(sdir, "attn.npz"), allow_pickle=True)
        assert set(chk.files) == set(p.files), "key set changed on save"
        assert np.allclose(chk["head.weight"], out["head.weight"], atol=0, rtol=0)
        assert np.allclose(chk["head.bias"], out["head.bias"], atol=0, rtol=0)
        saved.append(sdir)
        print(f"  baked serve member -> {sdir}/attn.npz")
    print(f"\nA={A:.5f} B={B:+.5f}  (baked into head; serve averages RAW)")
    print('SB_ATTN_DIRS="' + ",".join(saved) + '"')



if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "parity":
        parity(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    elif mode == "valgate":
        valgate(sys.argv[2:])
    else:
        raise SystemExit("mode must be parity|valgate")
