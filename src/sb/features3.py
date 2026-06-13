"""v3 streaming feature extractor: v2 + null-calibrated derived-stream windows.

Adds, on top of features2 (see its docstring for the v2 rationale):

  * Derived streams with their own per-series null calibration, targeting the
    "subtle" breaks (distribution / dependence / volatility-clustering changes
    that leave mean & variance roughly alone):
      - "a"-stream: |z|        -> volatility level & clustering (acf of |z|)
      - "d"-stream: z_t-z_{t-1}-> smoothness / MA structure (mean abs + acf)
      - "x"-stream: median-crossing indicator -> oscillation-rate changes
    For each stream we keep trailing windows (50, 200) with O(1) sums and
    cross-products, normalised by the stream's own historical null loc/scale
    measured at dyadic scales, exactly like the v2 window stats.
  * scan (max over the two windows) and running-max of each calibrated stat.
  * 2 extra series constants: acf1 of |z_h| (vol persistence), historical
    median-crossing rate.

Still O(1) per step, fully deterministic.
"""
from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np

from sb.features2 import (  # noqa: F401  (re-exported machinery)
    EWMA_ALPHAS, CUSUM_KS, CUSUM_VAR_K, DIST_WINDOWS, DIFF_WINDOW, N_EDGES,
    NULL_GRID, _LOG2_GRID, PW_CUSUM_KS, PW_EWMA_ALPHA,
    WindowDist, _slide_sum,
    FEATURE_NAMES as V2_FEATURE_NAMES,
    StreamingDetector as _V2Detector,
)

DSTREAM_WINDOWS = (50, 200)

_DS_NAMES = []
for _W in DSTREAM_WINDOWS:
    _DS_NAMES += [f"a_nmz_w{_W}", f"a_nacf_w{_W}",
                  f"d_nmz_w{_W}", f"d_nacf_w{_W}",
                  f"x_nmz_w{_W}"]
_DS_SCAN = ["a_scan_mz", "a_scan_acf", "d_scan_mz", "d_scan_acf", "x_scan_mz",
            "rm_a_scan_mz", "rm_a_scan_acf", "rm_d_scan_mz", "rm_d_scan_acf",
            "rm_x_scan_mz"]
_DS_CONST = ["acf1_h_abs", "cross_rate_h"]

FEATURE_NAMES = V2_FEATURE_NAMES + _DS_NAMES + _DS_SCAN + _DS_CONST
N_FEATURES = len(FEATURE_NAMES)
_N_V2 = len(V2_FEATURE_NAMES)


class StreamWin:
    """Trailing window over a derived stream: O(1) mean + lag-1 acf."""

    __slots__ = ("W", "q", "s1", "s2", "prod_sum", "n_pairs")

    def __init__(self, W: int) -> None:
        self.W = W
        self.reset()

    def reset(self) -> None:
        self.q = deque()
        self.s1 = self.s2 = 0.0
        self.prod_sum = 0.0
        self.n_pairs = 0

    def push(self, v: float) -> None:
        q = self.q
        if q:
            self.prod_sum += q[-1] * v
            self.n_pairs += 1
        q.append(v)
        self.s1 += v
        self.s2 += v * v
        if len(q) > self.W:
            ov = q.popleft()
            self.prod_sum -= ov * q[0]
            self.n_pairs -= 1
            self.s1 -= ov
            self.s2 -= ov * ov

    def mean(self) -> float:
        return self.s1 / len(self.q)

    def acf1(self) -> float:
        cw = len(self.q)
        m = self.s1 / cw
        v = self.s2 / cw - m * m
        if self.n_pairs > 1 and v > 1e-12:
            a = (self.prod_sum / self.n_pairs - m * m) / v
            return max(min(a, 1.0), -1.0)
        return 0.0

    def size(self) -> int:
        return len(self.q)


def _null_mean_acf(stream: np.ndarray, grid=NULL_GRID):
    """Per-scale null loc/scale of sliding mean and sliding acf1 of a stream."""
    n = stream.size
    loc_m = [0.0] * len(grid)
    sc_m = [1.0] * len(grid)
    loc_a = [0.0] * len(grid)
    sc_a = [1.0] * len(grid)
    if n < 64:
        return loc_m, sc_m, loc_a, sc_a
    c1 = np.concatenate(([0.0], np.cumsum(stream)))
    c2 = np.concatenate(([0.0], np.cumsum(stream * stream)))
    cross = stream[:-1] * stream[1:]
    cc = np.concatenate(([0.0], np.cumsum(cross)))
    for i, g in enumerate(grid):
        if g > n // 2:
            loc_m[i] = loc_m[i - 1] if i else 0.0
            sc_m[i] = sc_m[i - 1] if i else 1.0
            loc_a[i] = loc_a[i - 1] if i else 0.0
            sc_a[i] = sc_a[i - 1] if i else 1.0
            continue
        means = _slide_sum(c1, g) / g
        v = np.maximum(_slide_sum(c2, g) / g - means * means, 1e-12)
        cs = (cc[g - 1:] - cc[: n - g + 1]) / (g - 1)
        acf = np.clip((cs - means * means) / v, -1.0, 1.0)
        stride = max(1, g // 8)
        sl = slice(0, None, stride)
        loc_m[i] = float(np.mean(means[sl]))
        sc_m[i] = max(float(np.std(means[sl])), 1e-4)
        loc_a[i] = float(np.mean(acf[sl]))
        sc_a[i] = max(float(np.std(acf[sl])), 1e-3)
    return loc_m, sc_m, loc_a, sc_a


class StreamingDetector(_V2Detector):
    """v2 detector + derived-stream (|z|, dz, crossing) calibrated windows."""

    __slots__ = (
        "ds_null", "median_z", "a_wins", "d_wins", "x_wins",
        "ds_rm", "ds_const", "prev_z_ds", "prev_side",
    )

    def __init__(self) -> None:
        self.ds_null = None
        self.median_z = 0.0
        self.a_wins = [StreamWin(W) for W in DSTREAM_WINDOWS]
        self.d_wins = [StreamWin(W) for W in DSTREAM_WINDOWS]
        self.x_wins = [StreamWin(W) for W in DSTREAM_WINDOWS]
        self.ds_rm = [0.0] * 5
        self.ds_const = (0.0, 0.5)
        self.prev_z_ds = 0.0
        self.prev_side = 0
        super().__init__()

    # -------------------------------------------------------------- #
    def _reset_online(self) -> None:
        super()._reset_online()
        for w in self.a_wins:
            w.reset()
        for w in self.d_wins:
            w.reset()
        for w in self.x_wins:
            w.reset()
        self.ds_rm = [0.0] * 5

    def calibrate(self, x_hist) -> None:
        super().calibrate(x_hist)
        x = np.asarray(x_hist, dtype=np.float64)
        n = x.size
        if n == 0:
            self.ds_null = None
            self.ds_const = (0.0, 0.5)
            self.prev_z_ds = 0.0
            self.prev_side = 0
            return
        z = (x - self.mu_h) / self.sd_h
        self.median_z = (self.median_h - self.mu_h) / self.sd_h
        a = np.abs(z)
        d = np.diff(z, prepend=z[0])
        side = (z > self.median_z).astype(np.float64)
        xing = np.abs(np.diff(side, prepend=side[0]))
        self.ds_null = dict(
            a=_null_mean_acf(a),
            d=_null_mean_acf(d),
            x=_null_mean_acf(xing),
        )
        # vol persistence + crossing rate as constants
        am = a - a.mean()
        den = float(np.dot(am, am))
        acf_abs = float(np.dot(am[:-1], am[1:]) / den) if den > 0 else 0.0
        self.ds_const = (acf_abs, float(xing.mean()))
        self.prev_z_ds = float(z[-1])
        self.prev_side = 1 if z[-1] > self.median_z else 0
        self._reset_online()

    # -------------------------------------------------------------- #
    def _ds_interp(self, table: list, cw: int) -> float:
        lx = math.log2(cw) if cw > 1 else 1.0
        if lx <= _LOG2_GRID[0]:
            return table[0]
        if lx >= _LOG2_GRID[-1]:
            return table[-1]
        i = int(lx - _LOG2_GRID[0])
        f = lx - _LOG2_GRID[i]
        return table[i] * (1.0 - f) + table[i + 1] * f

    def update(self, x: float) -> np.ndarray:
        f2 = super().update(x)
        z = (x - self.mu_h) / self.sd_h
        a = -z if z < 0 else z
        dz = z - self.prev_z_ds
        self.prev_z_ds = z
        side = 1 if z > self.median_z else 0
        xing = 1.0 if side != self.prev_side else 0.0
        self.prev_side = side
        for w in self.a_wins:
            w.push(a)
        for w in self.d_wins:
            w.push(dz)
        for w in self.x_wins:
            w.push(xing)

        f = np.empty(N_FEATURES, dtype=np.float64)
        f[:_N_V2] = f2
        k = _N_V2
        scan = [0.0] * 5
        if self.ds_null is None:
            f[k:] = 0.0
            return f
        nla, nsa = self.ds_null["a"][0], self.ds_null["a"][1]
        nlaa, nsaa = self.ds_null["a"][2], self.ds_null["a"][3]
        nld, nsd = self.ds_null["d"][0], self.ds_null["d"][1]
        nlda, nsda = self.ds_null["d"][2], self.ds_null["d"][3]
        nlx, nsx = self.ds_null["x"][0], self.ds_null["x"][1]
        for i in range(len(DSTREAM_WINDOWS)):
            aw, dw, xw = self.a_wins[i], self.d_wins[i], self.x_wins[i]
            cw = aw.size()
            if cw >= 5:
                a_nmz = (aw.mean() - self._ds_interp(nla, cw)) / self._ds_interp(nsa, cw)
                a_nacf = (aw.acf1() - self._ds_interp(nlaa, cw)) / self._ds_interp(nsaa, cw)
                d_nmz = (dw.mean() - self._ds_interp(nld, cw)) / self._ds_interp(nsd, cw)
                d_nacf = (dw.acf1() - self._ds_interp(nlda, cw)) / self._ds_interp(nsda, cw)
                x_nmz = (xw.mean() - self._ds_interp(nlx, cw)) / self._ds_interp(nsx, cw)
            else:
                a_nmz = a_nacf = d_nmz = d_nacf = x_nmz = 0.0
            f[k] = a_nmz; f[k + 1] = a_nacf
            f[k + 2] = d_nmz; f[k + 3] = d_nacf
            f[k + 4] = x_nmz
            k += 5
            for si, v in enumerate((a_nmz, a_nacf, d_nmz, d_nacf, x_nmz)):
                av = -v if v < 0 else v
                if av > scan[si]:
                    scan[si] = av
        rm = self.ds_rm
        for i in range(5):
            if scan[i] > rm[i]:
                rm[i] = scan[i]
        for i in range(5):
            f[k + i] = scan[i]
        for i in range(5):
            f[k + 5 + i] = rm[i]
        k += 10
        f[k] = self.ds_const[0]
        f[k + 1] = self.ds_const[1]
        return f
