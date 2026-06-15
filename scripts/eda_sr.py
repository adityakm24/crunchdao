"""Round 6 — Shiryaev-Roberts / Bayesian mixture changepoint statistic test.

The existing features have only MAX-based changepoint detectors (CUSUM /
Page-Hinkley = sup_k of a cumulative sum, mean & variance, raw & prewhitened).
They do NOT have the complementary OPTIMAL Bayesian detector: the
Shiryaev-Roberts statistic R_t = sum_k prod_{i=k}^{t} LR_i (average of likelihood
ratios over ALL candidate change times), which is provably different from CUSUM
(sum vs max) and is a posterior-odds -> naturally [0,1] calibrated (Blueprint 3).

We maintain a BANK of SR recursions, one per alternative (mean-shift delta and/or
variance-scale rho), each O(1):  logR <- softplus(logR) + logLR(x).  The mixture
score = logsumexp_j (log w_j + logR_j) is the marginal posterior odds. We test it
both on RAW calibrated z and on AR(1)-PREWHITENED z (serial-correlation robust).

Cheap, no training. We measure standalone cross-sectional AUC, rank-correlation
with the base model, and the INCREMENTAL blend lift over the cached base logits.
Decision rule identical to eda_incremental.py: ship-worthy only if it shows a
consistent positive incremental lift with modest rank-corr. Run:
    uv run python scripts/eda_sr.py
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from sb.data import iter_series  # noqa: E402

SEED = 42
SNAP_STEPS = [100, 150, 200, 250, 300, 400]
LAMBDAS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]

# alternative grid (delta = mean shift in hist-sigma units; rho = std scale)
MEAN_DELTAS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
VAR_RHOS = (0.5, 1.5, 2.5, 4.0)
# joint corners (both mean and variance move)
JOINT = ((1.0, 2.0), (-1.0, 2.0), (1.5, 0.5), (-1.5, 0.5))


def _loglr_point(z, delta, rho):
    """Per-point Gaussian log-likelihood ratio: alt N(delta, rho^2) vs null N(0,1).
    z is the per-series-calibrated standardized value."""
    return -np.log(rho) - (z - delta) ** 2 / (2.0 * rho * rho) + z * z / 2.0


def _sr_mixture(z_online, alts, logw):
    """Streaming Shiryaev-Roberts mixture. Returns logR_mixture at each step.
    z_online: 1d standardized online stream. alts: list of (delta, rho)."""
    J = len(alts)
    logR = np.full(J, -50.0)  # ~ log(0)
    out = np.empty(len(z_online))
    deltas = np.array([a[0] for a in alts])
    rhos = np.array([a[1] for a in alts])
    lw = np.asarray(logw)
    for i, z in enumerate(z_online):
        ll = -np.log(rhos) - (z - deltas) ** 2 / (2.0 * rhos * rhos) + z * z / 2.0
        # softplus(logR) = log(1 + exp(logR)) stable
        sp = np.logaddexp(0.0, logR)
        logR = sp + ll
        m = np.max(logR + lw)
        out[i] = m + np.log(np.sum(np.exp(logR + lw - m)))
    return out


def _ar1_prewhiten(xh, xo):
    """Prewhiten online with AR(1) coef from historical; return calibrated resid z."""
    h = xh - xh.mean()
    d = float(np.dot(h, h))
    phi = float(np.dot(h[:-1], h[1:]) / d) if d > 1e-12 else 0.0
    phi = max(-0.95, min(0.95, phi))
    mu = xh.mean()
    sd = xh.std() + 1e-12
    # residual e_t = (x_t - mu) - phi*(x_{t-1} - mu); innovation sd = sd*sqrt(1-phi^2)
    isd = sd * np.sqrt(max(1e-6, 1.0 - phi * phi))
    prev = xo[0] - mu
    res = np.empty(len(xo))
    res[0] = (xo[0] - mu) / sd
    for i in range(1, len(xo)):
        cur = xo[i] - mu
        res[i] = (cur - phi * prev) / isd
        prev = cur
    return res


def cross_auc(score, label):
    label = np.asarray(label)
    n_pos = int(label.sum()); n_neg = int(len(label) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan, n_pos, n_neg
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sl = score[order]
    ranks[order] = np.arange(1, len(score) + 1)
    i = 0
    while i < len(sl):
        j = i
        while j + 1 < len(sl) and sl[j + 1] == sl[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    auc = (ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else v * 0.0


def main():
    print("loading base logits cache ...")
    d = np.load("features/val_base_logits.npz", allow_pickle=True)
    sid_arr = d["series_id"]; t_arr = d["t_online"]; base = d["base_logit"]
    base_lookup = {}
    for i in range(len(sid_arr)):
        base_lookup[(int(sid_arr[i]), int(t_arr[i]))] = float(base[i])
    val_ids = sorted(set(int(s) for s in sid_arr))
    print(f"  {len(val_ids)} VAL series")

    # build alternative banks
    alts_mean = [(dl, 1.0) for dl in MEAN_DELTAS]
    alts_var = [(0.0, r) for r in VAR_RHOS]
    alts_joint = list(JOINT)
    alts_full = alts_mean + alts_var + alts_joint
    banks = {
        "sr_mean_raw": (alts_mean, False),
        "sr_var_raw": (alts_var, False),
        "sr_full_raw": (alts_full, False),
        "sr_full_pw": (alts_full, True),   # prewhitened
    }

    stash = {name: {t: [] for t in SNAP_STEPS} for name in banks}
    stash_base = {t: [] for t in SNAP_STEPS}

    processed = 0
    for s in iter_series("train", ids=val_ids):
        xh = s.x_hist.astype(np.float64)
        xo = s.x_online.astype(np.float64)
        sid = s.series_id
        mu = xh.mean(); sd = xh.std() + 1e-12
        z_raw = (xo - mu) / sd
        z_pw = _ar1_prewhiten(xh, xo)
        for name, (alts, pw) in banks.items():
            zz = z_pw if pw else z_raw
            logw = np.zeros(len(alts))  # uniform prior
            sr = _sr_mixture(zz, alts, logw)
            for t in SNAP_STEPS:
                if t > len(xo):
                    continue
                key = (sid, t)
                if key not in base_lookup:
                    continue
                lab = 1 if (s.has_break and s.tau_index is not None and s.tau_index < t) else 0
                stash[name][t].append((sr[t - 1], lab, base_lookup[key]))
        processed += 1
        if processed % 250 == 0:
            print(f"  ...{processed} series")

    # build base accumulator once (from any bank's rows -- same keys)
    any_bank = next(iter(banks))
    for t in SNAP_STEPS:
        for (_, lab, b) in stash[any_bank][t]:
            stash_base[t].append((b, lab))

    def wmean(pairs):
        num = sum(a * w for a, w in pairs); den = sum(w for _, w in pairs)
        return num / den if den else np.nan

    # base TS-AUC
    base_pairs = []
    for t in SNAP_STEPS:
        rows = stash_base[t]
        if len(rows) < 30:
            continue
        b = np.array([r[0] for r in rows]); lab = np.array([r[1] for r in rows])
        a0, npos, nneg = cross_auc(b, lab)
        if np.isfinite(a0):
            base_pairs.append((a0, npos * nneg))
    base_ts = wmean(base_pairs)
    print(f"\nbase TS-AUC (snapshot approx): {base_ts:.4f}\n")
    print(f"{'bank':<16}{'standalone':>11}{'blend':>8}{'lift':>8}{'rankcorr':>10}{'best_lam':>9}")

    for name in banks:
        sa_pairs = []; bl_pairs = []; rcs = []; lams = []
        for t in SNAP_STEPS:
            rows = stash[name][t]
            if len(rows) < 30:
                continue
            sv = np.array([r[0] for r in rows], dtype=np.float64)
            lab = np.array([r[1] for r in rows], dtype=np.int64)
            b = np.array([r[2] for r in rows], dtype=np.float64)
            asa, npos, nneg = cross_auc(sv, lab)
            if not np.isfinite(asa):
                continue
            w = npos * nneg
            sa_pairs.append((asa, w))
            rcs.append(np.corrcoef(zscore(b), zscore(sv))[0, 1] if sv.std() > 1e-12 else 0.0)
            a0, _, _ = cross_auc(b, lab)
            best = (a0, 0.0)
            for lam in LAMBDAS:
                blended = b + lam * zscore(sv) * b.std()
                ab, _, _ = cross_auc(blended, lab)
                if ab > best[0]:
                    best = (ab, lam)
            bl_pairs.append((best[0], w)); lams.append(best[1])
        sa = wmean(sa_pairs); bl = wmean(bl_pairs)
        rc = float(np.nanmean(rcs)); lam_mode = max(set(lams), key=lams.count) if lams else 0.0
        print(f"{name:<16}{sa:>11.4f}{bl:>8.4f}{bl - base_ts:>+8.4f}{rc:>10.3f}{lam_mode:>9.2f}")


if __name__ == "__main__":
    main()
