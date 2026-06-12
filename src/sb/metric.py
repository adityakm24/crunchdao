"""Time-Stratified AUC (TS-AUC), the official competition metric.

At each online step ``t`` we compute a cross-sectional ROC AUC across all
series alive at that step, then take a weighted average where the weight is
the number of positive-negative pairs ``n_pos(t) * n_neg(t)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def ts_auc_from_frame(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> float:
    """Compute TS-AUC.

    Both frames are indexed by (id, time). ``predictions`` has a ``prediction``
    column, ``targets`` has a ``target`` column.
    """
    merged = predictions.merge(targets, how="left", left_index=True, right_index=True)
    # online step index per series (0,1,2,...)
    merged["t_online"] = merged.groupby(level="id").cumcount()
    return ts_auc_grouped(merged["t_online"].to_numpy(),
                          merged["target"].to_numpy(),
                          merged["prediction"].to_numpy())


def ts_auc_grouped(t_online: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> float:
    """TS-AUC via the Mann-Whitney rank identity (fast, tie-correct).

    Per step group, AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg) where R_pos
    is the sum of (average-tie) ranks of the positive samples. Identical to
    sklearn's roc_auc_score but ~10x faster (no per-call overhead), which keeps
    the LightGBM TS-AUC eval callback cheap.
    """
    order = np.argsort(t_online, kind="stable")
    t_online = t_online[order]
    target = target[order].astype(np.int64)
    prediction = prediction[order]

    weighted_sum = 0.0
    total_weight = 0.0
    uniq, starts = np.unique(t_online, return_index=True)
    starts = list(starts) + [len(t_online)]
    for i in range(len(uniq)):
        lo, hi = starts[i], starts[i + 1]
        labels = target[lo:hi]
        scores = prediction[lo:hi]
        n_pos = int(labels.sum())
        n_neg = labels.size - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(scores)  # average ranks handle ties
        r_pos = float(ranks[labels == 1].sum())
        auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        w = float(n_pos * n_neg)
        weighted_sum += w * auc
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.5
