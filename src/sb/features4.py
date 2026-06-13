"""v4 streaming feature extractor: v3 + null-calibrated higher-moment windows.

Motivation (diagnostics in experiments.md): the largest break cohort is
"subtle / distribution-only" (~68% of breaks) and the model only reaches
~0.56 AUC there. Those breaks change distribution *shape* (skew/kurt/tails)
while leaving mean & variance close to historical, so the mean/variance/acf
trackers miss them. Raw window skew/kurt exist in v2 but rank near-zero gain
because their null spread varies enormously across series (heavy-tailed DGPs
produce wild sample skew/kurt even with no break). v4 calibrates them per
series, the same way v2 calibrates mean/var/acf.

Adds on top of v3:
  * Windows (100, 200, 400, 800): null-calibrated skewness and excess
    kurtosis. Null loc/scale measured on the series' break-free historical
    segment at dyadic scales (8..512).
  * scan (max |.| over windows) + running-max of cal-skew and cal-kurt.
  * 1 constant: historical excess-kurt null scale at scale 128 (a
    tail-heaviness gate so the model knows how noisy this series' kurt is).

All updates O(1) per step (running power sums; no per-step rescans).
Deterministic.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from sb.features3 import (  # noqa: F401
    StreamingDetector as _V3Detector,
    FEATURE_NAMES as V3_FEATURE_NAMES,
    NULL_GRID, _LOG2_GRID, _slide_sum,
)

HM_WINDOWS = (100, 200, 400, 800)

_HM_NAMES = []
for _W in HM_WINDOWS:
    _HM_NAMES += [f"hm{_W}_nskew", f"hm{_W}_nkurt"]
_HM_SCAN = ["hm_scan_skew", "hm_scan_kurt",
            "rm_hm_scan_skew", "rm_hm_scan_kurt"]
_HM_CONST = ["null_skurt128"]

FEATURE_NAMES = V3_FEATURE_NAMES + _HM_NAMES + _HM_SCAN + _HM_CONST
N_FEATURES = len(FEATURE_NAMES)
_N_V3 = len(V3_FEATURE_NAMES)


class MomentWin:
    """Trailing window with O(1) skew / excess-kurt via running power sums."""

    __slots__ = ("W", "q", "s1", "s2", "s3", "s4")

    def __init__(self, W: int) -> None:
        self.W = W
        self.reset()

    def reset(self) -> None:
        self.q = deque()
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0

    def push(self, x: float) -> None:
        q = self.q
        q.append(x)
        xx = x * x
        self.s1 += x
        self.s2 += xx
        self.s3 += xx * x
        self.s4 += xx * xx
        if len(q) > self.W:
            ox = q.popleft()
            oxx = ox * ox
            self.s1 -= ox
            self.s2 -= oxx
            self.s3 -= oxx * ox
            self.s4 -= oxx * oxx

    def skew_kurt(self):
        cw = len(self.q)
        if cw < 3:
            return 0.0, 0.0, cw
        mean = self.s1 / cw
        var = self.s2 / cw - mean * mean
        if var <= 1e-12:
            return 0.0, 0.0, cw
        sd = math.sqrt(var)
        m3 = self.s3 / cw - 3 * mean * (self.s2 / cw) + 2 * mean ** 3
        m4 = (self.s4 / cw - 4 * mean * (self.s3 / cw)
              + 6 * mean ** 2 * (self.s2 / cw) - 3 * mean ** 4)
        return m3 / (sd ** 3), m4 / (var * var) - 3.0, cw

    def size(self) -> int:
        return len(self.q)


def _null_skew_kurt(z: np.ndarray, grid=NULL_GRID):
    """Per-scale null loc/scale of sliding-window skew and excess kurt on the
    break-free historical segment."""
    n = z.size
    loc_s = [0.0] * len(grid)
    sc_s = [math.sqrt(6.0 / g) for g in grid]
    loc_k = [0.0] * len(grid)
    sc_k = [math.sqrt(24.0 / g) for g in grid]
    if n < 64:
        return loc_s, sc_s, loc_k, sc_k
    c1 = np.concatenate(([0.0], np.cumsum(z)))
    c2 = np.concatenate(([0.0], np.cumsum(z * z)))
    c3 = np.concatenate(([0.0], np.cumsum(z ** 3)))
    c4 = np.concatenate(([0.0], np.cumsum(z ** 4)))
    for i, g in enumerate(grid):
        if g > n // 2:
            if i:
                loc_s[i], sc_s[i] = loc_s[i - 1], sc_s[i - 1]
                loc_k[i], sc_k[i] = loc_k[i - 1], sc_k[i - 1]
            continue
        m1 = _slide_sum(c1, g) / g
        m2 = _slide_sum(c2, g) / g
        m3raw = _slide_sum(c3, g) / g
        m4raw = _slide_sum(c4, g) / g
        var = np.maximum(m2 - m1 * m1, 1e-12)
        sd = np.sqrt(var)
        mu3 = m3raw - 3 * m1 * m2 + 2 * m1 ** 3
        mu4 = m4raw - 4 * m1 * m3raw + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        skew = mu3 / (sd ** 3)
        kurt = mu4 / (var * var) - 3.0
        stride = max(1, g // 8)
        sl = slice(0, None, stride)
        loc_s[i] = float(np.mean(skew[sl]))
        sc_s[i] = max(float(np.std(skew[sl])), 1e-3)
        loc_k[i] = float(np.mean(kurt[sl]))
        sc_k[i] = max(float(np.std(kurt[sl])), 1e-3)
    return loc_s, sc_s, loc_k, sc_k


class StreamingDetector(_V3Detector):
    """v3 detector + null-calibrated higher-moment window stats."""

    __slots__ = ("hm_wins", "hm_null_sk", "hm_rm", "hm_const")

    def __init__(self) -> None:
        self.hm_wins = [MomentWin(W) for W in HM_WINDOWS]
        self.hm_null_sk = None
        self.hm_rm = [0.0, 0.0]
        self.hm_const = (1.0,)
        super().__init__()

    def _reset_online(self) -> None:
        super()._reset_online()
        for w in self.hm_wins:
            w.reset()
        self.hm_rm = [0.0, 0.0]

    def calibrate(self, x_hist) -> None:
        super().calibrate(x_hist)
        x = np.asarray(x_hist, dtype=np.float64)
        n = x.size
        if n == 0:
            self.hm_null_sk = None
            self.hm_const = (1.0,)
            return
        z = (x - self.mu_h) / self.sd_h
        self.hm_null_sk = _null_skew_kurt(z)
        self.hm_const = (self.hm_null_sk[3][4],)  # kurt null scale at scale 128
        self._reset_online()

    def _hm_interp(self, table, cw):
        lx = math.log2(cw) if cw > 1 else 1.0
        if lx <= _LOG2_GRID[0]:
            return table[0]
        if lx >= _LOG2_GRID[-1]:
            return table[-1]
        i = int(lx - _LOG2_GRID[0])
        f = lx - _LOG2_GRID[i]
        return table[i] * (1.0 - f) + table[i + 1] * f

    def update(self, x: float) -> np.ndarray:
        f3 = super().update(x)
        z = (x - self.mu_h) / self.sd_h
        for w in self.hm_wins:
            w.push(z)

        f = np.empty(N_FEATURES, dtype=np.float64)
        f[:_N_V3] = f3
        k = _N_V3
        if self.hm_null_sk is None:
            f[k:] = 0.0
            return f
        loc_s, sc_s, loc_k, sc_k = self.hm_null_sk
        scan = [0.0, 0.0]
        for w in self.hm_wins:
            sk, ku, cw = w.skew_kurt()
            if cw >= 8:
                nsk = (sk - self._hm_interp(loc_s, cw)) / self._hm_interp(sc_s, cw)
                nku = (ku - self._hm_interp(loc_k, cw)) / self._hm_interp(sc_k, cw)
            else:
                nsk = nku = 0.0
            f[k] = nsk
            f[k + 1] = nku
            k += 2
            a = -nsk if nsk < 0 else nsk
            if a > scan[0]:
                scan[0] = a
            a = -nku if nku < 0 else nku
            if a > scan[1]:
                scan[1] = a
        rm = self.hm_rm
        if scan[0] > rm[0]:
            rm[0] = scan[0]
        if scan[1] > rm[1]:
            rm[1] = scan[1]
        f[k] = scan[0]; f[k + 1] = scan[1]
        f[k + 2] = rm[0]; f[k + 3] = rm[1]
        k += 4
        f[k] = self.hm_const[0]
        return f
