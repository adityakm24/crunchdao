"""v5 streaming feature extractor: v4 + long-run-variance (LRV) calibration of
the CUMULATIVE detectors.

Motivation: in v2-v4 the single highest-gain feature is always `log_t`, because
the cumulative detectors (cumulative mean-z, the multi-k CUSUM bank, Page-Hinkley)
grow with elapsed time AND with the series' serial dependence, and the GBT must
spend capacity using `log_t`/dependence features to discount that growth. TS-AUC
is cross-sectional, so two series at the same step with the same true evidence but
different autocorrelation get different raw cumulative magnitudes -> mis-ranked.

Fix (same family as v2's window calibration, but for the cumulative stats): under
H0 the variance of the cumulative mean is inflated by the long-run variance factor
  LRV = 1 + 2 * sum_k w_k * acf_k     (Bartlett-weighted, from the historical seg)
so cum_mean_z / sqrt(LRV) ~ N(0,1) regardless of dependence. We divide the
cumulative mean-z, Page-Hinkley, and CUSUM-bank maxima by sqrt(LRV) to make them
cross-sectionally comparable, and expose log(LRV) as a per-series constant.

All O(1) per step (LRV is one constant computed at calibrate time). Deterministic.
"""
from __future__ import annotations

import math

import numpy as np

from sb.features4 import (  # noqa: F401
    StreamingDetector as _V4Detector,
    FEATURE_NAMES as V4_FEATURE_NAMES,
)
from sb.features2 import CUSUM_KS

_LRV_NAMES = (["cum_mean_z_lrv", "ph_mean_lrv", "log_lrv"]
              + [f"cusum_absmax_k{str(k).replace('.', '')}_lrv" for k in CUSUM_KS])

FEATURE_NAMES = V4_FEATURE_NAMES + _LRV_NAMES
N_FEATURES = len(FEATURE_NAMES)
_N_V4 = len(V4_FEATURE_NAMES)

# indices of the source features inside the v4 vector
_IX_CUM_MEAN_Z = V4_FEATURE_NAMES.index("cum_mean_z")
_IX_PH_MEAN = V4_FEATURE_NAMES.index("ph_mean")
_IX_CUSUM = [V4_FEATURE_NAMES.index(f"cusum_absmax_k{str(k).replace('.', '')}")
             for k in CUSUM_KS]
LRV_MAXLAG = 64


class StreamingDetector(_V4Detector):
    """v4 detector + LRV-calibrated cumulative statistics."""

    __slots__ = ("inv_sqrt_lrv", "log_lrv")

    def __init__(self) -> None:
        self.inv_sqrt_lrv = 1.0
        self.log_lrv = 0.0
        super().__init__()

    def calibrate(self, x_hist) -> None:
        super().calibrate(x_hist)
        x = np.asarray(x_hist, dtype=np.float64)
        n = x.size
        if n < 8:
            self.inv_sqrt_lrv = 1.0
            self.log_lrv = 0.0
            return
        xc = x - x.mean()
        denom = float(np.dot(xc, xc))
        if denom <= 0:
            self.inv_sqrt_lrv = 1.0
            self.log_lrv = 0.0
            return
        K = min(LRV_MAXLAG, n - 1)
        lrv = 1.0
        for k in range(1, K + 1):
            ak = float(np.dot(xc[:-k], xc[k:]) / denom)
            w = 1.0 - k / (K + 1.0)          # Bartlett taper -> guarantees lrv>0
            lrv += 2.0 * w * ak
        lrv = max(lrv, 0.05)                  # floor (anti-correlated series)
        self.inv_sqrt_lrv = 1.0 / math.sqrt(lrv)
        self.log_lrv = math.log(lrv)

    def update(self, x: float) -> np.ndarray:
        f4 = super().update(x)
        f = np.empty(N_FEATURES, dtype=np.float64)
        f[:_N_V4] = f4
        k = _N_V4
        s = self.inv_sqrt_lrv
        f[k] = f4[_IX_CUM_MEAN_Z] * s
        f[k + 1] = f4[_IX_PH_MEAN] * s
        f[k + 2] = self.log_lrv
        k += 3
        for ix in _IX_CUSUM:
            f[k] = f4[ix] * s
            k += 1
        return f
