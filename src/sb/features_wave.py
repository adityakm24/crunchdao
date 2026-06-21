"""features_wave: causal multi-resolution (wavelet-style) band-energy block.

The last untried 2025-winners' view (they used wavelet denoising). v2's windows
are time-domain; this is a *frequency-band* representation. We run a causal
cascade of EWMA smooths on the historical-standardized stream z_t:

    s_0 = z_t ,  s_j = (1-a) s_j + a z_t  with a = 0.5  (cascaded -> scale ~2^j)
    detail_j = s_{j-1} - s_j                              (a band-pass channel)

A break that changes high-frequency content (noise/variance) lights up the FINE
detail bands; a slow mean/dependence drift lights up the COARSE bands. We compare
the trailing-online band energy E_j = <detail_j^2> against the series' OWN
historical band energy (measured on the break-free segment) as a log-ratio, so it
is cross-sectionally comparable (the per-series-calibration lesson that made v2
work). O(1)/step, deterministic, no RNG.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

N_LEVELS = 6           # detail bands j=1..6 (scales ~2..64)
WAVE_WINDOWS = (120, 480)
_EPS = 1e-12


def _names() -> list[str]:
    names = []
    for W in WAVE_WINDOWS:
        for j in range(1, N_LEVELS + 1):
            names.append(f"wv{W}_band{j}")          # log(E_win_j / E_hist_j)
        names.append(f"wv{W}_coarse_z")             # coarse smooth drift vs hist
    for j in range(1, N_LEVELS + 1):
        names.append(f"wvc_band{j}")                # cumulative log-ratio
    for j in range(1, N_LEVELS + 1):
        names.append(f"loghE{j}")                   # per-series historical band shape
    names += ["log_t", "t_over_nhist"]
    return names


FEATURE_NAMES = _names()
N_FEATURES = len(FEATURE_NAMES)


class _BandWindow:
    """Trailing window of one band's squared detail, O(1) running mean."""

    __slots__ = ("W", "q", "s")

    def __init__(self, W: int) -> None:
        self.W = W
        self.q = deque()
        self.s = 0.0

    def reset(self) -> None:
        self.q.clear()
        self.s = 0.0

    def push(self, e2: float) -> float:
        q = self.q
        q.append(e2)
        self.s += e2
        if len(q) > self.W:
            self.s -= q.popleft()
        return self.s / len(q)


class StreamingDetector:
    __slots__ = ("n_hist", "mu_h", "sd_h", "Eh", "coarse_h", "n",
                 "s_chain", "bw", "cum_e2", "cum_n")

    def __init__(self) -> None:
        self.bw = [[_BandWindow(W) for _ in range(N_LEVELS)] for W in WAVE_WINDOWS]
        self._reset_baseline()

    def _reset_baseline(self) -> None:
        self.n_hist = 0
        self.mu_h = 0.0
        self.sd_h = 1.0
        self.Eh = [1.0] * N_LEVELS              # historical band energies
        self.coarse_h = 0.0
        self._reset_online()

    def _reset_online(self) -> None:
        self.n = 0
        self.s_chain = [0.0] * (N_LEVELS + 1)   # s_0..s_N (s_0 set each step)
        self.cum_e2 = [0.0] * N_LEVELS
        self.cum_n = 0
        for bands in self.bw:
            for b in bands:
                b.reset()

    def calibrate(self, x_hist) -> None:
        x = np.asarray(x_hist, dtype=np.float64)
        n = x.size
        self._reset_baseline()
        if n < 32:
            self.n_hist = int(n)
            self._reset_online()
            return
        self.n_hist = int(n)
        mu = float(x.mean()); sd = max(float(x.std()), 1e-9)
        self.mu_h = mu; self.sd_h = sd
        z = (x - mu) / sd
        # run the same causal cascade over the historical stream to measure E_h
        s = np.zeros(N_LEVELS + 1)
        e2 = np.zeros(N_LEVELS)
        cnt = 0
        a = 0.5
        for k in range(n):
            s[0] = z[k]
            for j in range(1, N_LEVELS + 1):
                s[j] = (1 - a) * s[j] + a * s[j - 1] if k > 0 else s[j - 1]
            if k >= 8:   # let the cascade warm up
                for j in range(1, N_LEVELS + 1):
                    d = s[j - 1] - s[j]
                    e2[j - 1] += d * d
                cnt += 1
        if cnt > 0:
            self.Eh = [max(e2[j] / cnt, _EPS) for j in range(N_LEVELS)]
        self.coarse_h = float(s[N_LEVELS])
        self._reset_online()

    def update(self, x: float) -> np.ndarray:
        self.n += 1
        z = (x - self.mu_h) / self.sd_h
        s = self.s_chain
        s[0] = z
        a = 0.5
        first = self.n == 1
        for j in range(1, N_LEVELS + 1):
            s[j] = s[j - 1] if first else (1 - a) * s[j] + a * s[j - 1]
        for wi in range(len(WAVE_WINDOWS)):
            bands = self.bw[wi]
            for j in range(N_LEVELS):
                d = s[j] - s[j + 1]
                bands[j].push(d * d)
        for j in range(N_LEVELS):
            d = s[j] - s[j + 1]
            self.cum_e2[j] += d * d
        self.cum_n += 1
        return self._features()

    def _features(self) -> np.ndarray:
        f = []
        for wi, W in enumerate(WAVE_WINDOWS):
            bands = self.bw[wi]
            for j in range(N_LEVELS):
                ew = bands[j].s / max(len(bands[j].q), 1)
                f.append(math.log(max(ew, _EPS) / self.Eh[j]))
            coarse = self.s_chain[N_LEVELS]
            f.append(coarse - self.coarse_h)
        for j in range(N_LEVELS):
            ec = self.cum_e2[j] / max(self.cum_n, 1)
            f.append(math.log(max(ec, _EPS) / self.Eh[j]))
        for j in range(N_LEVELS):
            f.append(math.log(self.Eh[j]))
        f.append(math.log(self.n))
        f.append((self.n - 1) / self.n_hist if self.n_hist > 0 else 0.0)
        return np.asarray(f, dtype=np.float64)


assert len(FEATURE_NAMES) == N_FEATURES
