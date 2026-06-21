"""Round 11 probe — DGP-matched Bayesian Online Change-Point posterior (Blueprint 3).

THE one genuinely-untried online class. info.md Blueprint-3 names an explicit
two-state, unidirectional (absorbing) change-point HMM whose forward filter gives
a posterior P(S_t = 1 | x_1:t) -- naturally bounded in [0, 1], so globally
scaled it is directly a TS-AUC score. Round-6 eda_sr.py tried the cousin
Shiryaev-Roberts statistic and FAILED (standalone 0.52, weight 0) -- but it used
(a) GAUSSIAN likelihoods and (b) HARD-CODED round-number alternatives (+-0.5/1/2
sigma). Neither is the true DGP. By Neyman-Pearson, a likelihood-ratio test is
most powerful ONLY when matched to the true alternative; KS / moment features are
not optimal for the subtle cohort that dominates the loss.

What is NEW here (and only possible after we measured the DGP this round):
  1. The H1 mixture (mean-shift, log-variance-ratio) is SAMPLED FROM THE MEASURED
     break-magnitude joint distribution in features/dgp_params.npz -- the actual
     alternative, not round numbers.
  2. Heavy-tailed Student-t point likelihoods with df matched to the measured
     historical-innovation excess kurtosis (the 68% subtle cohort is non-Gaussian;
     a Gaussian likelihood under-weights tail evidence).
  3. AR(1) pre-whitening with the per-series measured rho_h.
  4. A proper HMM forward-filter POSTERIOR (bounded [0,1]) rather than the
     unbounded SR statistic.

Everything is pure-numpy (scipy only for gammaln, local-probe only), O(1)/step,
deterministic -> trivially cloud-deployable if it earns a weight.

GATE (identical to every member): standalone cross-sectional TS-AUC, rank-corr to
the 3-GRU mean (decorrelation axis) and to the shipped blend (redundancy axis),
and the blend lift onto the round-10 shipped logit on VAL full + BOTH honest
halves (seed 1). PROMISING only if it lifts full AND both halves; else Blueprint-3
is closed too and the per-series detection floor is confirmed once more.

Run:  uv run python scripts/val_bocpd.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, "src")

from sb.data import iter_series        # noqa: E402
from sb.metric import ts_auc_grouped   # noqa: E402

SEED = 42
HALF_SEED = 1
RANK_GW = 0.10
W_LIN = 0.10
W_GRU = 0.45

K_ATOMS = 48          # H1 mixture size (sampled from measured break joint)
HAZARDS = (0.003, 0.01, 0.03)   # geometric change-point prior (tau ~ Uniform approx)
MIN_HIST = 50
DGP = "features/dgp_params.npz"


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


def ar1(x, mu):
    h = x - mu
    d = float(np.dot(h, h))
    if d < 1e-12:
        return 0.0
    return float(np.clip(np.dot(h[:-1], h[1:]) / d, -0.95, 0.95))


def t_logpdf(u, df):
    """log pdf of a standardized (unit-scale) Student-t with given df, vectorized."""
    return (
        gammaln(0.5 * (df + 1.0)) - gammaln(0.5 * df)
        - 0.5 * np.log(df * np.pi)
        - 0.5 * (df + 1.0) * np.log1p(u * u / df)
    )


def df_from_kurt(kurt_ex):
    """Student-t df from excess kurtosis: kurt_ex = 6/(df-4) -> df = 4 + 6/kurt_ex.
    Near-Gaussian (small/neg kurt) -> large df. Clipped to a stable heavy-tail band."""
    if not np.isfinite(kurt_ex) or kurt_ex <= 0.2:
        return 40.0
    return float(np.clip(4.0 + 6.0 / kurt_ex, 4.5, 40.0))


# ---------------------------------------------------------------- DGP alternatives
def build_alternatives(k=K_ATOMS):
    """Sample k representative (dmean, log_var_ratio) atoms from the MEASURED break
    joint distribution -> the true H1 mixture. Deterministic (fixed rng)."""
    d = np.load(DGP)
    dmean = np.asarray(d["brk_dmean"], np.float64)
    lvr = np.asarray(d["brk_lvr"], np.float64)
    m = np.isfinite(dmean) & np.isfinite(lvr)
    dmean, lvr = dmean[m], lvr[m]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(dmean), size=k, replace=len(dmean) < k)
    return dmean[idx], lvr[idx]


# ---------------------------------------------------------------- per-series filter
def bocpd_scores(xh, xo, atom_dmean, atom_lvr, hazards):
    """Return {hazard: posterior p_t array} and the DGP-matched cumulative-LLR array.
    xh history, xo online. All O(T*K) vectorized over the K alternatives + a scalar
    forward recursion over T."""
    mu = float(xh.mean())
    sd = float(xh.std()) + 1e-9
    rho = ar1(xh, mu)
    # historical AR(1) innovations -> innovation scale + excess kurtosis -> df
    eh = (xh[1:] - mu) - rho * (xh[:-1] - mu)
    s_inn = float(eh.std()) + 1e-9
    uh = eh / s_inn
    kurt_ex = float((uh ** 4).mean() - 3.0) if len(uh) > 8 else 0.0
    df = df_from_kurt(kurt_ex)

    # online innovations u_t (causal: first online point uses last hist value as prev)
    prev = np.empty_like(xo)
    prev[0] = xh[-1]
    prev[1:] = xo[:-1]
    e = (xo - mu) - rho * (prev - mu)
    u = e / s_inn                                  # (T,)

    # alternative parameters in innovation (u) units:
    #   level shift dmean*sd in x  -> innovation-mean shift (1-rho)*dmean*sd / s_inn
    #                               = dmean * sqrt((1-rho)/(1+rho))
    #   variance ratio exp(lvr)    -> innovation-scale c = exp(lvr/2)
    m_j = atom_dmean * np.sqrt((1.0 - rho) / (1.0 + rho))      # (K,)
    c_j = np.exp(0.5 * atom_lvr)                                # (K,)

    logf0 = t_logpdf(u, df)                                     # (T,)
    uu = (u[:, None] - m_j[None, :]) / c_j[None, :]            # (T,K)
    logf1_j = t_logpdf(uu, df) - np.log(c_j)[None, :]          # (T,K)
    # mixture over K (uniform = empirical frequency of the measured atoms)
    mx = logf1_j.max(axis=1)
    logf1 = mx + np.log(np.exp(logf1_j - mx[:, None]).mean(axis=1) + 1e-300)
    logLR = logf1 - logf0                                       # (T,)
    LR = np.exp(np.clip(logLR, -50.0, 50.0))

    posts = {}
    for h in hazards:
        p = 0.0
        out = np.empty(len(xo))
        for i in range(len(xo)):
            prior_brk = p + (1.0 - p) * h          # P(S_t=1 | x_1:t-1)
            a1 = prior_brk * LR[i]
            a0 = (1.0 - p) * (1.0 - h)             # P(S_t=0 | x_1:t-1) * f0/f0
            p = a1 / (a1 + a0 + 1e-300)
            out[i] = p
        posts[h] = out
    cumllr = np.cumsum(logLR)                       # DGP-matched SR analog
    return posts, cumllr


# ---------------------------------------------------------------- shipped reference
def load_reference():
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
    return sid, t, y, shipped, gru


# ---------------------------------------------------------------- main
def main():
    sid, t, y, shipped, gru = load_reference()
    idx_of = {(int(s), int(tt)): i for i, (s, tt) in enumerate(zip(sid, t))}
    N = len(sid)
    val_ids = sorted(set(int(s) for s in sid))
    print(f"VAL rows {N}  series {len(val_ids)}  pos_rate {y.mean():.3f}")

    atom_dmean, atom_lvr = build_alternatives()
    print(f"H1 mixture: {len(atom_dmean)} atoms from measured break joint  "
          f"(|dmean| med {np.median(np.abs(atom_dmean)):.3f}, lvr med {np.median(atom_lvr):+.3f})")

    # score arrays aligned to base rows
    post = {h: np.full(N, np.nan) for h in HAZARDS}
    cll = np.full(N, np.nan)
    t0 = time.time()
    n = 0
    for s in iter_series("train", ids=val_ids):
        xh = s.x_hist.astype(np.float64)
        xo = s.x_online.astype(np.float64)
        if len(xh) < MIN_HIST or len(xo) == 0:
            continue
        posts, cumllr = bocpd_scores(xh, xo, atom_dmean, atom_lvr, HAZARDS)
        sidv = int(s.series_id)
        for tt in range(len(xo)):
            j = idx_of.get((sidv, tt))
            if j is None:
                continue
            for h in HAZARDS:
                post[h][j] = posts[h][tt]
            cll[j] = cumllr[tt]
        n += 1
        if n % 400 == 0:
            print(f"   {n}/{len(val_ids)}  {time.time()-t0:.0f}s")

    # fill any gaps (series too short to estimate) with neutral median
    for h in HAZARDS:
        med = np.nanmedian(post[h])
        post[h] = np.where(np.isfinite(post[h]), post[h], med)
    cll = np.where(np.isfinite(cll), cll, np.nanmedian(cll))
    print(f"scored in {time.time()-t0:.0f}s\n")

    # honest halves by series (rng HALF_SEED)
    u_ids = np.array(sorted(val_ids))
    np.random.default_rng(HALF_SEED).shuffle(u_ids)
    halfA = set(int(v) for v in u_ids[: len(u_ids) // 2])
    inA = np.array([int(x) in halfA for x in sid])

    shz = zscore(shipped)
    base_full = ts_auc_grouped(t, y, sigmoid(shz))
    base_A = ts_auc_grouped(t[inA], y[inA], sigmoid(shz[inA]))
    base_B = ts_auc_grouped(t[~inA], y[~inA], sigmoid(shz[~inA]))
    print(f"shipped blend  full={base_full:.4f}  halfA={base_A:.4f}  halfB={base_B:.4f}\n")

    candidates = {f"hmm_post_h{h}": post[h] for h in HAZARDS}
    candidates["cumllr"] = cll

    best_overall = (-1.0, None)
    best_logit = None
    for name, fv in candidates.items():
        sa = ts_auc_grouped(t, y, fv)
        if sa < 0.5:
            fv = -fv
            sa = 1.0 - sa
        rc_gru = spearman(fv, gru)
        rc_ship = spearman(fv, shipped)
        hz = zscore(fv)
        print(f"=== {name}  standalone TS-AUC={sa:.4f}  rankcorr(gru)={rc_gru:+.3f}  "
              f"rankcorr(ship)={rc_ship:+.3f} ===")
        best = None
        for w in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
            p = sigmoid(shz + w * hz)
            f = ts_auc_grouped(t, y, p)
            a = ts_auc_grouped(t[inA], y[inA], p[inA])
            b = ts_auc_grouped(t[~inA], y[~inA], p[~inA])
            flag = "  <-- both halves up" if (a > base_A + 1e-5 and b > base_B + 1e-5) else ""
            print(f"  w={w:4.2f}  full={f:.4f} ({f-base_full:+.4f})  "
                  f"halfA={a:.4f} ({a-base_A:+.4f})  halfB={b:.4f} ({b-base_B:+.4f}){flag}")
            robust = min(f - base_full, a - base_A, b - base_B)
            if best is None or robust > best[1]:
                best = (w, robust)
        print(f"  BEST robust min-lift (full,A,B): {best[1]:+.4f} @ w={best[0]}\n")
        if best[1] > best_overall[0]:
            best_overall = (best[1], name)
            best_logit = (name, fv, sa, rc_gru)

    print("================ VERDICT ================")
    bl, bn = best_overall
    if bl > 0.0005:
        print(f"PROMISING: {bn} robust min-lift {bl:+.4f} -> gate in 3-way neural stack.")
    else:
        print(f"REDUNDANT: best {bn} robust min-lift {bl:+.4f} (<= +0.0005). DGP-matched BOCPD "
              f"adds nothing at the blend -> Blueprint-3 closed; detection floor reconfirmed.")

    # cache the best HMM posterior for downstream attn_probe / attn_stack_search
    name, fv, sa, rc = best_logit
    np.savez("features/val_bocpd_logits.npz",
             val_logit=fv, series_id=sid, t_online=t, y=y)
    print(f"\nsaved features/val_bocpd_logits.npz  (best member '{name}', "
          f"standalone {sa:.4f}, rankcorr-gru {rc:+.3f})")


if __name__ == "__main__":
    main()
