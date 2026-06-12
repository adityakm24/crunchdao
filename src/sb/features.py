"""O(1)-per-step streaming feature extractor for structural-break detection.

EXP-007 feature set
-------------------
* Context (dropped at train time): online step / historical length.
* Cumulative online moments vs the historical baseline (mean z, mean dev,
  log-var ratio, skew, kurt, skew/kurt diffs).
* EWMA mean z-scores (short-term mean shift).
* Tail-mass excess vs Gaussian expectation.
* Cumulative online lag-1 autocorrelation + diff vs historical.
* Cumulative eCDF distance (KS / L1) vs historical deciles.
* Multi-sensitivity mean CUSUM (k=0.25/0.5/1.0), variance CUSUM (z^2-1), and
  Page-Hinkley drift -- raw and sqrt(n)-standardized.
* Variance-of-first-differences ratio (cumulative + window) and mean-abs-diff
  ratio: robust detectors of dependence / smoothness (autocorrelation) breaks.
* Trailing-window two-sample stats vs the clean historical reference at multiple
  scales (25/50/100/200): mean z, log-var ratio, skew, kurt, eCDF KS, lag-1 acf
  diff. Multi-scale lets the model pick the right horizon for each break's age.

All state updates are ~O(1) per observation (deque windows, running power sums),
keeping total cost ~ O(total points) within the 15h budget, and fully
deterministic (no RNG in the feature path).
"""
from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np

EWMA_ALPHAS = (0.05, 0.20)
CUSUM_KS = (0.25, 0.5, 1.0)   # mean-CUSUM allowances (sensitivity ladder)
CUSUM_VAR_K = 0.5
DIST_WINDOWS = (25, 50, 100, 200)
DIFF_WINDOW = 50
N_EDGES = 19

_BASE_NAMES = [
    "t", "log_t", "log_n_hist", "t_over_nhist",
    "cum_mean_z", "cum_mean_dev", "cum_logvar_ratio",
    "cum_skew", "cum_kurt", "cum_skew_diff", "cum_kurt_diff",
    "ewma_z_fast", "ewma_z_slow",
    "tail2_excess", "tail3_excess",
    "online_acf1", "acf1_diff",
    "ks_stat", "ecdf_l1", "below_hist_med_excess",
    "cusum_var_absmax", "cusum_var_absmax_n", "ph_mean", "ph_mean_n",
    "dvar_ratio_cum", "dvar_ratio_win", "dabs_ratio_cum",
    "chi2_cum", "online_acf2", "acf2_diff",
]
_CUSUM_NAMES = []
for _k in CUSUM_KS:
    tag = str(_k).replace(".", "")
    _CUSUM_NAMES += [f"cusum_absmax_k{tag}", f"cusum_absmax_k{tag}_n"]
_WIN_NAMES = []
for _W in DIST_WINDOWS:
    _WIN_NAMES += [f"w{_W}_mean_z", f"w{_W}_logvar", f"w{_W}_skew",
                   f"w{_W}_kurt", f"w{_W}_ks", f"w{_W}_acf1_diff", f"w{_W}_chi2"]

FEATURE_NAMES = _BASE_NAMES + _CUSUM_NAMES + _WIN_NAMES
N_FEATURES = len(FEATURE_NAMES)


class WindowDist:
    """Fixed-size trailing window with O(1) moment / eCDF / acf maintenance."""

    __slots__ = ("W", "q", "s1", "s2", "s3", "s4", "bins", "bin_counts",
                 "prod_sum", "n_pairs", "n_edges")

    def __init__(self, W: int, n_edges: int) -> None:
        self.W = W
        self.n_edges = n_edges
        self.reset()

    def reset(self) -> None:
        self.q = deque()
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0
        self.bins = deque()
        self.bin_counts = [0] * (self.n_edges + 1)
        self.prod_sum = 0.0
        self.n_pairs = 0

    def push(self, x: float, b: int) -> None:
        q = self.q
        if q:
            self.prod_sum += q[-1] * x
            self.n_pairs += 1
        q.append(x)
        self.s1 += x
        xx = x * x
        self.s2 += xx
        self.s3 += xx * x
        self.s4 += xx * xx
        self.bins.append(b)
        self.bin_counts[b] += 1
        if len(q) > self.W:
            ox = q.popleft()
            self.prod_sum -= ox * q[0]
            self.n_pairs -= 1
            oxx = ox * ox
            self.s1 -= ox
            self.s2 -= oxx
            self.s3 -= oxx * ox
            self.s4 -= oxx * oxx
            self.bin_counts[self.bins.popleft()] -= 1

    def stats(self, mu_h, sd_h, var_h, hist_cdf, acf1_h):
        cw = len(self.q)
        mean = self.s1 / cw
        var = max(self.s2 / cw - mean * mean, 0.0)
        mean_z = (mean - mu_h) / max(sd_h / math.sqrt(cw), 1e-9)
        logvar = math.log(max(var, 1e-12) / max(var_h, 1e-12))
        if cw > 2 and var > 1e-12:
            sd = math.sqrt(var)
            m3 = self.s3 / cw - 3 * mean * (self.s2 / cw) + 2 * mean ** 3
            m4 = (self.s4 / cw - 4 * mean * (self.s3 / cw)
                  + 6 * mean ** 2 * (self.s2 / cw) - 3 * mean ** 4)
            skew = m3 / (sd ** 3)
            kurt = m4 / (var * var) - 3.0
        else:
            skew = kurt = 0.0
        run = 0
        ks = 0.0
        inv = 1.0 / cw
        bc = self.bin_counts
        nb = self.n_edges + 1
        exp = cw / nb            # historical bins are ~equiprobable by construction
        inv_exp = 1.0 / exp
        chi2 = 0.0
        for j in range(self.n_edges):
            run += bc[j]
            d = abs(run * inv - hist_cdf[j])
            if d > ks:
                ks = d
            dd = bc[j] - exp
            chi2 += dd * dd
        # last bin (above top edge)
        dd = bc[self.n_edges] - exp
        chi2 = (chi2 + dd * dd) * inv_exp
        if self.n_pairs > 1 and var > 1e-12:
            acf1 = (self.prod_sum / self.n_pairs - mean * mean) / var
            acf1 = max(min(acf1, 1.0), -1.0)
        else:
            acf1 = 0.0
        return mean_z, logvar, skew, kurt, ks, acf1 - acf1_h, chi2


class StreamingDetector:
    """Maintains streaming state for a single series; reset per series."""

    __slots__ = (
        "mu_h", "sd_h", "var_h", "skew_h", "kurt_h", "acf1_h", "acf2_h", "median_h",
        "dvar_h", "dabs_h", "edges", "hist_cdf", "n_hist",
        "n", "s1", "s2", "s3", "s4", "ewma", "neff",
        "cusum_pos", "cusum_neg", "cusum_absmax",
        "cusum_var_pos", "cusum_var_neg", "cusum_var_absmax",
        "cumz", "mincum", "maxcum", "tail2", "tail3",
        "ac_cross", "ac_n", "ac_lastx", "ac_lastx2", "ac2_cross", "ac2_n",
        "ac_sx", "ac_sx2",
        "bin_counts", "below_med",
        "d_n", "d_s1", "d_s2", "d_abs", "dwin_q", "dwin_s1", "dwin_s2",
        "windows",
    )

    def __init__(self) -> None:
        self.windows = [WindowDist(W, N_EDGES) for W in DIST_WINDOWS]
        self.reset_baseline()

    def reset_baseline(self) -> None:
        self.mu_h = 0.0
        self.sd_h = 1.0
        self.var_h = 1.0
        self.skew_h = self.kurt_h = self.acf1_h = self.acf2_h = self.median_h = 0.0
        self.dvar_h = 1.0
        self.dabs_h = 1.0
        self.n_hist = 0
        self.edges = list(np.linspace(-1, 1, N_EDGES))
        self.hist_cdf = [k / (N_EDGES + 1) for k in range(1, N_EDGES + 1)]
        self._reset_online()

    def _reset_online(self) -> None:
        self.n = 0
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0
        self.ewma = [0.0 for _ in EWMA_ALPHAS]
        self.neff = [0.0 for _ in EWMA_ALPHAS]
        self.cusum_pos = [0.0 for _ in CUSUM_KS]
        self.cusum_neg = [0.0 for _ in CUSUM_KS]
        self.cusum_absmax = [0.0 for _ in CUSUM_KS]
        self.cusum_var_pos = self.cusum_var_neg = self.cusum_var_absmax = 0.0
        self.cumz = self.mincum = self.maxcum = 0.0
        self.tail2 = self.tail3 = 0
        self.ac_cross = 0.0
        self.ac_n = 0
        self.ac_lastx = None
        self.ac_lastx2 = None
        self.ac2_cross = 0.0
        self.ac2_n = 0
        self.ac_sx = self.ac_sx2 = 0.0
        self.bin_counts = [0] * (N_EDGES + 1)
        self.below_med = 0
        self.d_n = 0
        self.d_s1 = self.d_s2 = self.d_abs = 0.0
        self.dwin_q = deque()
        self.dwin_s1 = self.dwin_s2 = 0.0
        for w in self.windows:
            w.reset()

    def calibrate(self, x_hist) -> None:
        x = np.asarray(x_hist, dtype=np.float64)
        n = x.size
        if n == 0:
            self.reset_baseline()
            return
        self.n_hist = int(n)
        mu = float(x.mean())
        sd = float(x.std(ddof=1)) if n > 1 else 1.0
        sd = max(sd, 1e-9)
        z = (x - mu) / sd
        self.mu_h = mu
        self.sd_h = sd
        self.var_h = sd * sd
        self.skew_h = float((z ** 3).mean()) if n > 2 else 0.0
        self.kurt_h = float((z ** 4).mean() - 3.0) if n > 3 else 0.0
        self.median_h = float(np.median(x))
        if n > 2:
            xc = x - mu
            denom = float(np.dot(xc, xc))
            self.acf1_h = float(np.dot(xc[:-1], xc[1:]) / denom) if denom > 0 else 0.0
            self.acf2_h = float(np.dot(xc[:-2], xc[2:]) / denom) if (denom > 0 and n > 3) else 0.0
        else:
            self.acf1_h = 0.0
            self.acf2_h = 0.0
        if n > 2:
            dx = np.diff(x)
            self.dvar_h = max(float(dx.var()), 1e-9)
            self.dabs_h = max(float(np.abs(dx).mean()), 1e-9)
        else:
            self.dvar_h = 1.0
            self.dabs_h = 1.0
        qs = np.linspace(1.0 / (N_EDGES + 1), N_EDGES / (N_EDGES + 1), N_EDGES)
        edges = np.quantile(x, qs)
        edges = np.maximum.accumulate(edges + np.arange(N_EDGES) * 1e-12)
        self.edges = edges.tolist()
        self.hist_cdf = qs.tolist()
        self._reset_online()

    def update(self, x: float) -> np.ndarray:
        mu_h, sd_h = self.mu_h, self.sd_h
        z = (x - mu_h) / sd_h
        self.n += 1
        self.s1 += x
        xx = x * x
        self.s2 += xx
        self.s3 += xx * x
        self.s4 += xx * xx

        for i, a in enumerate(EWMA_ALPHAS):
            self.ewma[i] = (1.0 - a) * self.ewma[i] + a * x if self.neff[i] > 0 else x
            self.neff[i] = (1.0 - a) * self.neff[i] + 1.0

        # multi-k mean CUSUM
        for i, kk in enumerate(CUSUM_KS):
            cp = self.cusum_pos[i] + z - kk
            if cp < 0.0:
                cp = 0.0
            cn = self.cusum_neg[i] - z - kk
            if cn < 0.0:
                cn = 0.0
            self.cusum_pos[i] = cp
            self.cusum_neg[i] = cn
            m = cp if cp > cn else cn
            if m > self.cusum_absmax[i]:
                self.cusum_absmax[i] = m

        # variance CUSUM on (z^2 - 1)
        zz1 = z * z - 1.0
        self.cusum_var_pos = max(0.0, self.cusum_var_pos + zz1 - CUSUM_VAR_K)
        self.cusum_var_neg = max(0.0, self.cusum_var_neg - zz1 - CUSUM_VAR_K)
        mv = self.cusum_var_pos if self.cusum_var_pos > self.cusum_var_neg else self.cusum_var_neg
        if mv > self.cusum_var_absmax:
            self.cusum_var_absmax = mv

        # Page-Hinkley two-sided drift
        self.cumz += z
        if self.cumz < self.mincum:
            self.mincum = self.cumz
        if self.cumz > self.maxcum:
            self.maxcum = self.cumz

        az = -z if z < 0 else z
        if az > 2.0:
            self.tail2 += 1
        if az > 3.0:
            self.tail3 += 1

        # autocorr (lag-1 & lag-2) accumulators + first-difference trackers
        prev1 = self.ac_lastx
        prev2 = self.ac_lastx2
        if prev1 is not None:
            self.ac_cross += prev1 * x
            self.ac_n += 1
            dx = x - prev1
            self.d_n += 1
            self.d_s1 += dx
            self.d_s2 += dx * dx
            self.d_abs += -dx if dx < 0 else dx
            dq = self.dwin_q
            dq.append(dx)
            self.dwin_s1 += dx
            self.dwin_s2 += dx * dx
            if len(dq) > DIFF_WINDOW:
                od = dq.popleft()
                self.dwin_s1 -= od
                self.dwin_s2 -= od * od
        if prev2 is not None:
            self.ac2_cross += prev2 * x
            self.ac2_n += 1
        self.ac_lastx2 = prev1
        self.ac_lastx = x
        self.ac_sx += x
        self.ac_sx2 += xx

        b = bisect.bisect_right(self.edges, x)
        self.bin_counts[b] += 1
        if x < self.median_h:
            self.below_med += 1

        for w in self.windows:
            w.push(x, b)

        return self._features()

    def _features(self) -> np.ndarray:
        n = self.n
        mu_h, sd_h, var_h = self.mu_h, self.sd_h, self.var_h
        f = np.empty(N_FEATURES, dtype=np.float64)
        mean_o = self.s1 / n
        var_o = max(self.s2 / n - mean_o * mean_o, 0.0)
        cum_mean_dev = (mean_o - mu_h) / sd_h
        cum_mean_z = cum_mean_dev * math.sqrt(n)
        cum_logvar = math.log(max(var_o, 1e-12) / max(var_h, 1e-12))
        if n > 2 and var_o > 1e-12:
            sd_o = math.sqrt(var_o)
            m3 = self.s3 / n - 3 * mean_o * (self.s2 / n) + 2 * mean_o ** 3
            m4 = (self.s4 / n - 4 * mean_o * (self.s3 / n)
                  + 6 * mean_o ** 2 * (self.s2 / n) - 3 * mean_o ** 4)
            cum_skew = m3 / (sd_o ** 3)
            cum_kurt = m4 / (var_o * var_o) - 3.0
        else:
            cum_skew = cum_kurt = 0.0

        ewma_z = []
        for i in range(len(EWMA_ALPHAS)):
            se = sd_h / math.sqrt(max(self.neff[i], 1.0))
            ewma_z.append((self.ewma[i] - mu_h) / max(se, 1e-9))

        tail2_excess = self.tail2 / n - 0.0455
        tail3_excess = self.tail3 / n - 0.0027

        if self.ac_n > 1:
            mac = self.ac_sx / n
            var_ac = self.ac_sx2 / n - mac * mac
            online_acf1 = (self.ac_cross / self.ac_n - mac * mac) / max(var_ac, 1e-12)
            online_acf1 = max(min(online_acf1, 1.0), -1.0)
            if self.ac2_n > 1:
                online_acf2 = (self.ac2_cross / self.ac2_n - mac * mac) / max(var_ac, 1e-12)
                online_acf2 = max(min(online_acf2, 1.0), -1.0)
            else:
                online_acf2 = 0.0
        else:
            online_acf1 = 0.0
            online_acf2 = 0.0

        run = 0
        ks_stat = 0.0
        ecdf_l1 = 0.0
        chi2_cum = 0.0
        inv_n = 1.0 / n
        exp_n = n / (N_EDGES + 1)
        bc = self.bin_counts
        hcdf = self.hist_cdf
        for j in range(N_EDGES):
            run += bc[j]
            d = abs(run * inv_n - hcdf[j])
            ecdf_l1 += d
            if d > ks_stat:
                ks_stat = d
            dd = bc[j] - exp_n
            chi2_cum += dd * dd
        dd = bc[N_EDGES] - exp_n
        chi2_cum = (chi2_cum + dd * dd) / exp_n
        below_excess = self.below_med / n - 0.5

        ph_mean = max(self.cumz - self.mincum, self.maxcum - self.cumz)
        inv_sqrt_n = 1.0 / math.sqrt(n)

        # first-difference variance / abs ratios (dependence breaks)
        if self.d_n > 1:
            dvar_cum = max(self.d_s2 / self.d_n - (self.d_s1 / self.d_n) ** 2, 0.0)
            dvar_ratio_cum = math.log(max(dvar_cum, 1e-12) / self.dvar_h)
            dabs_ratio_cum = (self.d_abs / self.d_n) / self.dabs_h
        else:
            dvar_ratio_cum = 0.0
            dabs_ratio_cum = 1.0
        cwd = len(self.dwin_q)
        if cwd > 1:
            dwin_var = max(self.dwin_s2 / cwd - (self.dwin_s1 / cwd) ** 2, 0.0)
            dvar_ratio_win = math.log(max(dwin_var, 1e-12) / self.dvar_h)
        else:
            dvar_ratio_win = 0.0

        f[0] = n - 1
        f[1] = math.log(n)
        f[2] = math.log(self.n_hist) if self.n_hist > 0 else 0.0
        f[3] = (n - 1) / self.n_hist if self.n_hist > 0 else 0.0
        f[4] = cum_mean_z
        f[5] = cum_mean_dev
        f[6] = cum_logvar
        f[7] = cum_skew
        f[8] = cum_kurt
        f[9] = cum_skew - self.skew_h
        f[10] = cum_kurt - self.kurt_h
        f[11] = ewma_z[0]
        f[12] = ewma_z[1]
        f[13] = tail2_excess
        f[14] = tail3_excess
        f[15] = online_acf1
        f[16] = online_acf1 - self.acf1_h
        f[17] = ks_stat
        f[18] = ecdf_l1
        f[19] = below_excess
        f[20] = self.cusum_var_absmax
        f[21] = self.cusum_var_absmax * inv_sqrt_n
        f[22] = ph_mean
        f[23] = ph_mean * inv_sqrt_n
        f[24] = dvar_ratio_cum
        f[25] = dvar_ratio_win
        f[26] = dabs_ratio_cum
        f[27] = chi2_cum
        f[28] = online_acf2
        f[29] = online_acf2 - self.acf2_h
        k = 30
        for i in range(len(CUSUM_KS)):
            f[k] = self.cusum_absmax[i]
            f[k + 1] = self.cusum_absmax[i] * inv_sqrt_n
            k += 2
        for w in self.windows:
            mz, lv, sk, ku, ks, ad, chi2 = w.stats(mu_h, sd_h, var_h, hcdf, self.acf1_h)
            f[k] = mz; f[k + 1] = lv; f[k + 2] = sk
            f[k + 3] = ku; f[k + 4] = ks; f[k + 5] = ad; f[k + 6] = chi2
            k += 7
        return f
