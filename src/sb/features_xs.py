"""features_xs: an INDEPENDENT feature block for a decorrelated GBT member.

Motivation (round 12, feature-block deep-dive). The v2/v3/v4 lineage is
saturated and its members correlate 0.97-0.99 because they share the same
Gaussian-z + eCDF-KS + CUSUM calibrated representation. The 2025 challenge
winners' edge was *epistemic diversity*: independent feature blocks built on
genuinely different views of the series. This block deliberately uses the two
views v2 under-represents, to maximise the chance of NEW decorrelated signal on
the 68% subtle / 20% acf cohorts:

  (A) Distribution-free PIT view.  p_t = F_hist(x_t) is the probability-integral
      transform through the *historical* empirical CDF.  Under H0 (no break) the
      online p_t ~ Uniform(0,1): mean 0.5, var 1/12, symmetric.  PIT moments are
      robust to the heavy non-Gaussian tails that make Gaussian-z mean/var noisy
      on the subtle majority -- a mean/var/shape shift the marginal barely shows
      in KS can still move the PIT moments.

  (B) Multi-lag dependence view.  v2 only tracks acf1/acf2.  Here we compare the
      online trailing-window autocorrelations at lags 1..L against the series'
      OWN historical acf at the same lags -- a richer dependence fingerprint for
      "distribution-only" breaks that are really dynamics shifts.

Everything is O(1) per step (incremental window sums + ring buffers); calibration
is O(n_hist) vectorised.  Deterministic, no RNG.  Per-series calibration keeps it
cross-sectionally comparable (the lesson that made v2 work).
"""
from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np

XS_WINDOWS = (120, 480)
XS_LAGS = (1, 2, 3, 4, 5)
PIT_GRID = 256          # historical quantile knots for the PIT lookup
UNIF_VAR = 1.0 / 12.0   # Var(Uniform(0,1))

# null spread of window PIT-mean ~ sqrt(Var/cw); used to z-scale per window fill
_PIT_MEAN_NULL_SD = math.sqrt(UNIF_VAR)


def _names() -> list[str]:
    names = []
    for W in XS_WINDOWS:
        names += [f"xw{W}_pit_mean_z", f"xw{W}_pit_logvar", f"xw{W}_pit_skew",
                  f"xw{W}_pit_tail_lo", f"xw{W}_pit_tail_hi", f"xw{W}_pit_below"]
        for L in XS_LAGS:
            names.append(f"xw{W}_acfd{L}")
        names.append(f"xw{W}_zcr_d")          # oscillation-rate diff vs hist
        names.append(f"xw{W}_cumslope")       # cumsum drift slope (z residuals)
    # cumulative PIT view
    names += ["xc_pit_mean_z", "xc_pit_logvar", "xc_pit_skew", "xc_pit_below"]
    # per-series dependence constants (cross-sectional rank info)
    for L in XS_LAGS:
        names.append(f"acf{L}_h")
    names += ["zcr_h", "log_t", "t_over_nhist"]
    return names


FEATURE_NAMES = _names()
N_FEATURES = len(FEATURE_NAMES)


class _PitWindow:
    """Trailing window over PIT values p in (0,1) and z values, O(1) maintenance.

    Maintains PIT moment sums, tail-mass counts, below-0.5 count, multi-lag z
    autocorrelation cross-sums, and a sign-change (turning) counter on first
    differences of z.
    """

    __slots__ = ("W", "L", "pq", "zq", "p1", "p2", "p3", "lo", "hi", "below",
                 "zc1", "zc2", "cross", "npair", "dprev", "sign_prev", "turns",
                 "nturn")

    def __init__(self, W: int, lags) -> None:
        self.W = W
        self.L = tuple(lags)
        self.reset()

    def reset(self) -> None:
        self.pq = deque()
        self.zq = deque()
        self.p1 = self.p2 = self.p3 = 0.0
        self.lo = self.hi = self.below = 0
        self.zc1 = self.zc2 = 0.0           # sum z, sum z^2 (for acf normaliser)
        self.cross = [0.0] * len(self.L)    # lagged cross sums
        self.npair = [0] * len(self.L)
        self.dprev = None                   # previous z (for sign of diff)
        self.sign_prev = 0
        self.turns = 0                      # # sign changes of dz in window
        self.nturn = 0

    def push(self, p: float, z: float) -> None:
        pq, zq = self.pq, self.zq
        # incremental lagged cross-sums BEFORE appending (use current tail)
        for i, L in enumerate(self.L):
            if len(zq) >= L:
                self.cross[i] += zq[-L] * z
                self.npair[i] += 1
        # turning points on dz sign
        if self.dprev is not None:
            dz = z - self.dprev
            s = 1 if dz > 0 else (-1 if dz < 0 else 0)
            if s != 0 and self.sign_prev != 0 and s != self.sign_prev:
                self.turns += 1
            if s != 0:
                self.sign_prev = s
            self.nturn += 1
        self.dprev = z
        pq.append(p); zq.append(z)
        self.p1 += p; pp = p * p; self.p2 += pp; self.p3 += pp * p
        if p < 0.25:
            self.lo += 1
        if p > 0.75:
            self.hi += 1
        if p < 0.5:
            self.below += 1
        self.zc1 += z; self.zc2 += z * z
        if len(pq) > self.W:
            op = pq.popleft(); oz = zq.popleft()
            self.p1 -= op; opp = op * op; self.p2 -= opp; self.p3 -= opp * op
            if op < 0.25:
                self.lo -= 1
            if op > 0.75:
                self.hi -= 1
            if op < 0.5:
                self.below -= 1
            self.zc1 -= oz; self.zc2 -= oz * oz
            # drop the lagged pairs that leave the window
            for i, L in enumerate(self.L):
                if len(zq) >= L:
                    self.cross[i] -= oz * zq[L - 1]
                    self.npair[i] -= 1

    def features(self, acf_h, zcr_h):
        cw = len(self.pq)
        if cw < 8:
            n = 6 + len(self.L) + 2
            return [0.0] * n
        mp = self.p1 / cw
        vp = max(self.p2 / cw - mp * mp, 1e-12)
        # PIT mean z: deviation from 0.5 in per-window null sd units
        pit_mean_z = (mp - 0.5) / (_PIT_MEAN_NULL_SD / math.sqrt(cw))
        pit_logvar = math.log(vp / UNIF_VAR)
        sdp = math.sqrt(vp)
        m3 = self.p3 / cw - 3 * mp * (self.p2 / cw) + 2 * mp ** 3
        pit_skew = m3 / (sdp ** 3)
        tail_lo = self.lo / cw - 0.25
        tail_hi = self.hi / cw - 0.25
        pit_below = self.below / cw - 0.5
        out = [pit_mean_z, pit_logvar, pit_skew, tail_lo, tail_hi, pit_below]
        # multi-lag acf diffs vs historical
        mz = self.zc1 / cw
        vz = max(self.zc2 / cw - mz * mz, 1e-12)
        for i, L in enumerate(self.L):
            if self.npair[i] > 2:
                ac = (self.cross[i] / self.npair[i] - mz * mz) / vz
                ac = max(min(ac, 1.0), -1.0)
            else:
                ac = 0.0
            out.append(ac - acf_h[i])
        # oscillation-rate diff
        zcr = self.turns / self.nturn if self.nturn > 0 else 0.0
        out.append(zcr - zcr_h)
        # cumsum drift slope of z over the window (normalised)
        # slope ~ 12*sum(i*z)/cw^3 - but cheap proxy: corr(time, z) scaled.
        # Use a lightweight running estimate via mean of first vs second half.
        # (kept O(1)-ish via the moment sums is impossible for halves; use mz drift)
        out.append(mz)   # window mean of z (persistent location drift proxy)
        return out


class StreamingDetector:
    """features_xs streaming extractor; reset per series via calibrate()."""

    __slots__ = ("n_hist", "acf_h", "zcr_h", "pit_knots", "n", "log_t_cache",
                 "windows", "c_p1", "c_p2", "c_p3", "c_below", "med_h")

    def __init__(self) -> None:
        self.windows = [_PitWindow(W, XS_LAGS) for W in XS_WINDOWS]
        self._reset_baseline()

    def _reset_baseline(self) -> None:
        self.n_hist = 0
        self.acf_h = [0.0] * len(XS_LAGS)
        self.zcr_h = 0.5
        self.med_h = 0.0
        # default PIT knots: standard normal-ish; replaced in calibrate
        self.pit_knots = np.linspace(-3, 3, PIT_GRID)
        self._reset_online()

    def _reset_online(self) -> None:
        self.n = 0
        self.c_p1 = self.c_p2 = self.c_p3 = 0.0
        self.c_below = 0
        for w in self.windows:
            w.reset()

    def calibrate(self, x_hist) -> None:
        x = np.asarray(x_hist, dtype=np.float64)
        n = x.size
        self._reset_baseline()
        if n < 16:
            self.n_hist = int(n)
            self._reset_online()
            return
        self.n_hist = int(n)
        mu = float(x.mean())
        sd = max(float(x.std()), 1e-9)
        xc = x - mu
        denom = float(np.dot(xc, xc))
        for i, L in enumerate(XS_LAGS):
            if n > L + 2 and denom > 0:
                self.acf_h[i] = float(np.dot(xc[:-L], xc[L:]) / denom)
            else:
                self.acf_h[i] = 0.0
        # historical turning-point rate on first differences
        dz = np.diff(x)
        s = np.sign(dz)
        s = s[s != 0]
        if s.size > 1:
            self.zcr_h = float(np.mean(s[1:] != s[:-1]))
        else:
            self.zcr_h = 0.5
        self.med_h = float(np.median(x))
        # PIT knots: sorted historical sample (subsampled to PIT_GRID for speed)
        xs = np.sort(x)
        if xs.size > PIT_GRID:
            idx = np.linspace(0, xs.size - 1, PIT_GRID).astype(np.int64)
            xs = xs[idx]
        self.pit_knots = xs
        self._reset_online()

    def _pit(self, x: float) -> float:
        # fraction of historical <= x, via bisect on sorted knots, in (0,1)
        k = bisect.bisect_right(self.pit_knots, x)
        m = len(self.pit_knots)
        return (k + 0.5) / (m + 1.0)

    def update(self, x: float) -> np.ndarray:
        self.n += 1
        # z via historical knots' robust loc/scale (median + IQR-ish from knots)
        p = self._pit(x)
        # z relative to historical median scaled by knot spread (robust)
        m = len(self.pit_knots)
        lo = self.pit_knots[max(0, int(0.159 * m))]
        hi = self.pit_knots[min(m - 1, int(0.841 * m))]
        scale = max((hi - lo) / 2.0, 1e-9)
        z = (x - self.med_h) / scale
        # cumulative PIT
        self.c_p1 += p; pp = p * p; self.c_p2 += pp; self.c_p3 += pp * p
        if p < 0.5:
            self.c_below += 1
        for w in self.windows:
            w.push(p, z)
        return self._features()

    def _features(self) -> np.ndarray:
        f = []
        for w in self.windows:
            f.extend(w.features(self.acf_h, self.zcr_h))
        n = self.n
        mp = self.c_p1 / n
        vp = max(self.c_p2 / n - mp * mp, 1e-12)
        pit_mean_z = (mp - 0.5) / (_PIT_MEAN_NULL_SD / math.sqrt(n))
        pit_logvar = math.log(vp / UNIF_VAR)
        sdp = math.sqrt(vp)
        m3 = self.c_p3 / n - 3 * mp * (self.c_p2 / n) + 2 * mp ** 3
        pit_skew = m3 / (sdp ** 3)
        pit_below = self.c_below / n - 0.5
        f += [pit_mean_z, pit_logvar, pit_skew, pit_below]
        f += list(self.acf_h)
        f += [self.zcr_h,
              math.log(n),
              (n - 1) / self.n_hist if self.n_hist > 0 else 0.0]
        return np.asarray(f, dtype=np.float64)


assert len(FEATURE_NAMES) == N_FEATURES
