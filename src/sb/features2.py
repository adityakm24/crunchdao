"""v2 streaming feature extractor: v1 (EXP-007) features + per-series
empirical-null calibration, scan statistics, and AR(1)-prewhitened trackers.

Why v2 exists (see EXP diagnostics in experiments.md):
  Under H0 the sliding-window statistics have wildly different spreads across
  series (std of window-50 mean-z ranges 0.53..1.83 across series) because the
  DGPs have heterogeneous serial dependence. TS-AUC is a *cross-sectional*
  ranking, so a raw |z|=3 from a wandering series must not outrank |z|=3 from a
  quiet series. The break-free historical segment lets us measure each series'
  own null distribution of every window statistic and normalise online stats
  into per-series empirical z/p units.

New feature groups (on top of the unchanged v1 set):
  * w{W}_l1            : per-window mean |eCDF - hist eCDF| (CvM-flavoured).
  * w{W}_n{mz,lv,acf,ks,l1} : null-calibrated window stats. The null loc/scale
    is measured on the historical segment at dyadic scales (8..512) and
    interpolated at the window's current fill cw.
  * scan_n*            : max over window scales of the calibrated stats — a
    multiscale scan statistic (near-optimal for unknown break age).
  * rm_scan_n*         : running max of the scans (break evidence is
    cumulative; keeps old breaks ranked once spotted).
  * pw_*               : AR(1)-prewhitened trackers. e_t = (z_t - phi z_{t-1})
    / sqrt(1-phi^2) with phi from the historical acf1. CUSUM/EWMA/variance on
    e_t are correctly calibrated even for strongly autocorrelated series.
  * series-level null descriptors: acf1_h, kurt_h, null spreads at scale 64,
    long-memory slope. Constant per series (vary across series, so they carry
    cross-sectional rank information and gate the other features).

All updates remain O(1) per step; calibration is O(n_hist) vectorised numpy.
Everything is deterministic (no RNG anywhere).
"""
from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np

EWMA_ALPHAS = (0.05, 0.20)
CUSUM_KS = (0.25, 0.5, 1.0)   # mean-CUSUM allowances (sensitivity ladder)
CUSUM_VAR_K = 0.5
DIST_WINDOWS = (25, 50, 100, 200, 400)
DIFF_WINDOW = 50
N_EDGES = 19

NULL_GRID = (8, 16, 32, 64, 128, 256, 512)   # dyadic null-calibration scales
_LOG2_GRID = [math.log2(g) for g in NULL_GRID]
PW_CUSUM_KS = (0.25, 0.5)
PW_EWMA_ALPHA = 0.10

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
                   f"w{_W}_kurt", f"w{_W}_ks", f"w{_W}_acf1_diff", f"w{_W}_chi2",
                   f"w{_W}_l1",
                   f"w{_W}_nmz", f"w{_W}_nlv", f"w{_W}_nacf", f"w{_W}_nks",
                   f"w{_W}_nl1"]
_SCAN_NAMES = ["scan_nmz", "scan_nlv", "scan_nacf", "scan_nks", "scan_nl1",
               "rm_scan_nmz", "rm_scan_nlv", "rm_scan_nacf", "rm_scan_nks",
               "rm_scan_nl1"]
_PW_NAMES = ["pw_cusum_absmax_k025", "pw_cusum_absmax_k025_n",
             "pw_cusum_absmax_k05", "pw_cusum_absmax_k05_n",
             "pw_cusum_var_absmax", "pw_cusum_var_absmax_n",
             "pw_ewma_z", "pw_cum_mean_z", "pw_cum_logvar", "pw_acf1"]
_CONST_NAMES = ["acf1_h", "kurt_h", "null_smz64", "null_slv64", "null_sacf64",
                "null_mem_slope"]

FEATURE_NAMES = (_BASE_NAMES + _CUSUM_NAMES + _WIN_NAMES + _SCAN_NAMES
                 + _PW_NAMES + _CONST_NAMES)
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
        """Raw two-sample stats vs historical: returns
        (mean_z, logvar, skew, kurt, ks, acf1_diff, chi2, l1, cw)."""
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
        l1 = 0.0
        inv = 1.0 / cw
        bc = self.bin_counts
        nb = self.n_edges + 1
        exp = cw / nb            # historical bins are ~equiprobable by construction
        inv_exp = 1.0 / exp
        chi2 = 0.0
        for j in range(self.n_edges):
            run += bc[j]
            d = abs(run * inv - hist_cdf[j])
            l1 += d
            if d > ks:
                ks = d
            dd = bc[j] - exp
            chi2 += dd * dd
        # last bin (above top edge)
        dd = bc[self.n_edges] - exp
        chi2 = (chi2 + dd * dd) * inv_exp
        l1 /= self.n_edges
        if self.n_pairs > 1 and var > 1e-12:
            acf1 = (self.prod_sum / self.n_pairs - mean * mean) / var
            acf1 = max(min(acf1, 1.0), -1.0)
        else:
            acf1 = 0.0
        return mean_z, logvar, skew, kurt, ks, acf1 - acf1_h, chi2, l1, cw


def _slide_sum(c: np.ndarray, g: int) -> np.ndarray:
    """Sliding-window sums of width g from a cumsum array c (1-D)."""
    return c[g:] - c[:-g]


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
        # --- v2 ---
        "null_loc", "null_scale", "phi", "pw_denom", "z_prev",
        "pw_cusum_pos", "pw_cusum_neg", "pw_cusum_absmax",
        "pw_cusum_var_pos", "pw_cusum_var_neg", "pw_cusum_var_absmax",
        "pw_ewma", "pw_neff", "pw_s1", "pw_s2",
        "pw_ac_cross", "pw_ac_n", "pw_e_prev",
        "rm_scan", "const_feats",
    )

    def __init__(self) -> None:
        self.windows = [WindowDist(W, N_EDGES) for W in DIST_WINDOWS]
        self.reset_baseline()

    # ------------------------------------------------------------------ #
    # null-table machinery
    # ------------------------------------------------------------------ #
    def _default_nulls(self) -> None:
        # loc/scale per stat per grid scale; iid-Gaussian-flavoured defaults
        nl, ns = {}, {}
        for st in ("mz", "lv", "acf", "ks", "l1"):
            nl[st] = [0.0] * len(NULL_GRID)
            ns[st] = [1.0] * len(NULL_GRID)
        for i, g in enumerate(NULL_GRID):
            ns["mz"][i] = 1.0
            ns["lv"][i] = math.sqrt(2.0 / g)
            nl["lv"][i] = -1.0 / g
            ns["acf"][i] = 1.0 / math.sqrt(g)
            nl["ks"][i] = 0.86 / math.sqrt(g)
            ns["ks"][i] = 0.35 / math.sqrt(g)
            nl["l1"][i] = 0.4 / math.sqrt(g)
            ns["l1"][i] = 0.2 / math.sqrt(g)
        self.null_loc, self.null_scale = nl, ns

    def _interp(self, table: list, cw: int) -> float:
        """Log2-linear interpolation of a per-scale null parameter at fill cw."""
        lx = math.log2(cw) if cw > 1 else 1.0
        if lx <= _LOG2_GRID[0]:
            return table[0]
        if lx >= _LOG2_GRID[-1]:
            return table[-1]
        i = int(lx - _LOG2_GRID[0])  # grid is consecutive powers of two
        f = lx - _LOG2_GRID[i]
        return table[i] * (1.0 - f) + table[i + 1] * f

    def _calibrate_nulls(self, z: np.ndarray, bins_h: np.ndarray) -> None:
        """Measure per-series null loc/scale of window stats on the
        (break-free) historical segment at dyadic scales."""
        n = z.size
        self._default_nulls()
        if n < 64:
            return
        c1 = np.concatenate(([0.0], np.cumsum(z)))
        c2 = np.concatenate(([0.0], np.cumsum(z * z)))
        cross = z[:-1] * z[1:]
        cc = np.concatenate(([0.0], np.cumsum(cross)))
        nb = N_EDGES + 1
        onehot = bins_h[:, None] == np.arange(nb)[None, :]
        B = np.concatenate((np.zeros((1, nb)), np.cumsum(onehot, axis=0)))
        qarr = np.asarray(self.hist_cdf)

        for i, g in enumerate(NULL_GRID):
            if g > n // 2:
                # not enough data: copy previous scale (already iid default)
                for st in ("mz", "lv", "acf", "ks", "l1"):
                    self.null_loc[st][i] = self.null_loc[st][i - 1] if i else self.null_loc[st][i]
                    self.null_scale[st][i] = self.null_scale[st][i - 1] if i else self.null_scale[st][i]
                continue
            means = _slide_sum(c1, g) / g
            mz = means * math.sqrt(g)            # z-units: mu=0, sd=1
            v = np.maximum(_slide_sum(c2, g) / g - means * means, 1e-12)
            lv = np.log(v)
            # acf1 within window: pairs i..i+g-2
            cs = (cc[g - 1:] - cc[: n - g + 2 - 1])[: means.size] / (g - 1)
            acf = np.clip((cs - means * means) / v, -1.0, 1.0)
            acfd = acf - self.acf1_h
            stride = max(1, g // 8)
            sl = slice(0, None, stride)
            self.null_loc["mz"][i] = float(np.mean(mz[sl]))
            self.null_scale["mz"][i] = max(float(np.std(mz[sl])), 0.05)
            self.null_loc["lv"][i] = float(np.mean(lv[sl]))
            self.null_scale["lv"][i] = max(float(np.std(lv[sl])), 0.01)
            self.null_loc["acf"][i] = float(np.mean(acfd[sl]))
            self.null_scale["acf"][i] = max(float(np.std(acfd[sl])), 0.01)
            # KS / L1 on a strided subsample (heavier)
            stride_k = max(1, g // 4)
            pos = np.arange(0, n - g + 1, stride_k)
            counts = B[pos + g] - B[pos]                       # (P, nb)
            F = np.cumsum(counts, axis=1)[:, :N_EDGES] / g     # (P, 19)
            D = np.abs(F - qarr[None, :])
            ks = D.max(axis=1)
            l1 = D.mean(axis=1)
            ks_med = float(np.median(ks))
            l1_med = float(np.median(l1))
            self.null_loc["ks"][i] = ks_med
            self.null_scale["ks"][i] = max(float(np.quantile(ks, 0.9)) - ks_med, 0.005)
            self.null_loc["l1"][i] = l1_med
            self.null_scale["l1"][i] = max(float(np.quantile(l1, 0.9)) - l1_med, 0.003)

    # ------------------------------------------------------------------ #
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
        self._default_nulls()
        self.phi = 0.0
        self.pw_denom = 1.0
        self.z_prev = 0.0
        self.const_feats = (0.0, 0.0, 1.0, 0.1, 0.1, 0.0)
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
        # --- v2 online state ---
        self.pw_cusum_pos = [0.0 for _ in PW_CUSUM_KS]
        self.pw_cusum_neg = [0.0 for _ in PW_CUSUM_KS]
        self.pw_cusum_absmax = [0.0 for _ in PW_CUSUM_KS]
        self.pw_cusum_var_pos = self.pw_cusum_var_neg = 0.0
        self.pw_cusum_var_absmax = 0.0
        self.pw_ewma = 0.0
        self.pw_neff = 0.0
        self.pw_s1 = self.pw_s2 = 0.0
        self.pw_ac_cross = 0.0
        self.pw_ac_n = 0
        self.pw_e_prev = None
        self.rm_scan = [0.0] * 5

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

        # --- v2: per-series empirical nulls + prewhitening setup ---
        bins_h = np.searchsorted(edges, x, side="right")
        self._calibrate_nulls(z, bins_h)
        self.phi = max(min(self.acf1_h, 0.95), -0.95)
        self.pw_denom = math.sqrt(max(1.0 - self.phi * self.phi, 1e-4))
        self.z_prev = float(z[-1])
        smz64 = self.null_scale["mz"][3]
        slv64 = self.null_scale["lv"][3]
        sacf64 = self.null_scale["acf"][3]
        mem_slope = math.log(max(self.null_scale["mz"][5], 1e-6)
                             / max(self.null_scale["mz"][1], 1e-6))
        self.const_feats = (self.acf1_h, self.kurt_h, smz64, slv64, sacf64,
                            mem_slope)
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

        # --- v2: prewhitened residual trackers ---
        e = (z - self.phi * self.z_prev) / self.pw_denom
        self.z_prev = z
        for i, kk in enumerate(PW_CUSUM_KS):
            cp = self.pw_cusum_pos[i] + e - kk
            if cp < 0.0:
                cp = 0.0
            cn = self.pw_cusum_neg[i] - e - kk
            if cn < 0.0:
                cn = 0.0
            self.pw_cusum_pos[i] = cp
            self.pw_cusum_neg[i] = cn
            m = cp if cp > cn else cn
            if m > self.pw_cusum_absmax[i]:
                self.pw_cusum_absmax[i] = m
        ee1 = e * e - 1.0
        self.pw_cusum_var_pos = max(0.0, self.pw_cusum_var_pos + ee1 - CUSUM_VAR_K)
        self.pw_cusum_var_neg = max(0.0, self.pw_cusum_var_neg - ee1 - CUSUM_VAR_K)
        mv = (self.pw_cusum_var_pos if self.pw_cusum_var_pos > self.pw_cusum_var_neg
              else self.pw_cusum_var_neg)
        if mv > self.pw_cusum_var_absmax:
            self.pw_cusum_var_absmax = mv
        self.pw_ewma = ((1.0 - PW_EWMA_ALPHA) * self.pw_ewma + PW_EWMA_ALPHA * e
                        if self.pw_neff > 0 else e)
        self.pw_neff = (1.0 - PW_EWMA_ALPHA) * self.pw_neff + 1.0
        ep = self.pw_e_prev
        if ep is not None:
            self.pw_ac_cross += ep * e
            self.pw_ac_n += 1
        self.pw_e_prev = e
        self.pw_s1 += e
        self.pw_s2 += e * e

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

        # ---- windows: raw + null-calibrated stats, and scan maxima ----
        nloc, nscale = self.null_loc, self.null_scale
        scan = [0.0, 0.0, 0.0, 0.0, 0.0]
        for w in self.windows:
            mz, lv, sk, ku, ks, ad, chi2, l1, cw = w.stats(
                mu_h, sd_h, var_h, hcdf, self.acf1_h)
            if cw >= 5:
                nmz = mz / self._interp(nscale["mz"], cw)
                nlv = (lv - self._interp(nloc["lv"], cw)) / self._interp(nscale["lv"], cw)
                nacf = (ad - self._interp(nloc["acf"], cw)) / self._interp(nscale["acf"], cw)
                nks = (ks - self._interp(nloc["ks"], cw)) / self._interp(nscale["ks"], cw)
                nl1 = (l1 - self._interp(nloc["l1"], cw)) / self._interp(nscale["l1"], cw)
            else:
                nmz = nlv = nacf = nks = nl1 = 0.0
            f[k] = mz; f[k + 1] = lv; f[k + 2] = sk
            f[k + 3] = ku; f[k + 4] = ks; f[k + 5] = ad; f[k + 6] = chi2
            f[k + 7] = l1
            f[k + 8] = nmz; f[k + 9] = nlv; f[k + 10] = nacf
            f[k + 11] = nks; f[k + 12] = nl1
            k += 13
            a = -nmz if nmz < 0 else nmz
            if a > scan[0]:
                scan[0] = a
            a = -nlv if nlv < 0 else nlv
            if a > scan[1]:
                scan[1] = a
            a = -nacf if nacf < 0 else nacf
            if a > scan[2]:
                scan[2] = a
            if nks > scan[3]:
                scan[3] = nks
            if nl1 > scan[4]:
                scan[4] = nl1

        rm = self.rm_scan
        for i in range(5):
            if scan[i] > rm[i]:
                rm[i] = scan[i]
        f[k] = scan[0]; f[k + 1] = scan[1]; f[k + 2] = scan[2]
        f[k + 3] = scan[3]; f[k + 4] = scan[4]
        f[k + 5] = rm[0]; f[k + 6] = rm[1]; f[k + 7] = rm[2]
        f[k + 8] = rm[3]; f[k + 9] = rm[4]
        k += 10

        # ---- prewhitened trackers ----
        f[k] = self.pw_cusum_absmax[0]
        f[k + 1] = self.pw_cusum_absmax[0] * inv_sqrt_n
        f[k + 2] = self.pw_cusum_absmax[1]
        f[k + 3] = self.pw_cusum_absmax[1] * inv_sqrt_n
        f[k + 4] = self.pw_cusum_var_absmax
        f[k + 5] = self.pw_cusum_var_absmax * inv_sqrt_n
        se = 1.0 / math.sqrt(max(self.pw_neff, 1.0))
        f[k + 6] = self.pw_ewma / max(se, 1e-9)
        f[k + 7] = self.pw_s1 * inv_sqrt_n
        f[k + 8] = math.log(max(self.pw_s2 / n, 1e-12))
        if self.pw_ac_n > 1:
            pme = self.pw_s1 / n
            pve = max(self.pw_s2 / n - pme * pme, 1e-12)
            pac = (self.pw_ac_cross / self.pw_ac_n - pme * pme) / pve
            f[k + 9] = max(min(pac, 1.0), -1.0)
        else:
            f[k + 9] = 0.0
        k += 10

        cf = self.const_feats
        f[k] = cf[0]; f[k + 1] = cf[1]; f[k + 2] = cf[2]
        f[k + 3] = cf[3]; f[k + 4] = cf[4]; f[k + 5] = cf[5]
        return f
