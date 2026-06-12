"""Smoke test: run the streaming extractor on a few training series."""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from sb.data import iter_series
from sb.features import StreamingDetector, FEATURE_NAMES, N_FEATURES

det = StreamingDetector()
ids = list(range(6))
t0 = time.time()
npoints = 0
for s in iter_series("train", ids=ids):
    det.calibrate(s.x_hist)
    feats = None
    for x in s.x_online:
        feats = det.update(float(x))
        npoints += 1
    print(f"id={s.series_id} n_hist={s.n_hist} n_online={s.n_online} "
          f"tau={s.tau_index} last_cusum_absmax={feats[19]:.2f} "
          f"ks={feats[24]:.3f} logvar={feats[6]:.3f} acf_diff={feats[23]:.3f}")
    assert feats.shape == (N_FEATURES,)
    assert np.all(np.isfinite(feats)), "non-finite feature!"
dt = time.time() - t0
print(f"\n{N_FEATURES} features: {FEATURE_NAMES}")
print(f"processed {npoints} points in {dt:.3f}s -> {1e6*dt/npoints:.2f} us/point")
