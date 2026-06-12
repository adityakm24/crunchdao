"""Data access layer for the ADIA Lab Structural Break Real-Time challenge.

Local parquet layout (in ``Dataset/``):

- ``X_train.parquet``      MultiIndex (id, time) -> columns [value, period]
                           period == 1 : historical segment (break-free)
                           period == 2 : online segment
- ``y_train.parquet``      MultiIndex (id, time) -> column [target]
                           target is the per-online-step ideal label
                           (0 before break, 1 from break onward). Only covers
                           the online portion (time >= historical length).
- ``y_train_index.parquet``  index id -> [tau_index, tau]
                           tau_index : 0-based position within online segment,
                                       -1 if no break.
                           tau       : absolute ``time`` of the break, -1 if none.

The reduced test set mirrors this layout with ids starting at 10000 and
``y_test.reduced`` carrying the per-step target.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("SB_DATA_DIR", "Dataset")

PERIOD_HISTORICAL = 1
PERIOD_ONLINE = 2


@dataclass
class Series:
    """One time series split into historical and online segments."""

    series_id: int
    x_hist: np.ndarray  # historical segment values
    x_online: np.ndarray  # online segment values
    tau_index: Optional[int]  # break index within online segment (None if no break)

    @property
    def n_hist(self) -> int:
        return len(self.x_hist)

    @property
    def n_online(self) -> int:
        return len(self.x_online)

    @property
    def has_break(self) -> bool:
        return self.tau_index is not None


def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def load_index(split: str = "train") -> pd.DataFrame:
    """Return per-id break index (tau_index, tau). -1 normalised to NaN."""
    fname = "y_train_index.parquet" if split == "train" else "y_test_index.reduced.parquet"
    idx = pd.read_parquet(_path(fname))
    return idx


def iter_series(split: str = "train", ids: Optional[list[int]] = None) -> Iterator[Series]:
    """Stream series one at a time without loading everything into memory.

    Reads the X parquet in id order (it is sorted) and yields ``Series``.
    ``ids`` optionally restricts to a subset (still streamed in file order).
    """
    if split == "train":
        x_file, idx_file = "X_train.parquet", "y_train_index.parquet"
    else:
        x_file, idx_file = "X_test.reduced.parquet", "y_test_index.reduced.parquet"

    index = pd.read_parquet(_path(idx_file))
    tau_map = index["tau_index"].to_dict()

    want = set(ids) if ids is not None else None

    # Read the full X frame; it is large but fits in RAM (~1-2 GB for train).
    x = pd.read_parquet(_path(x_file), columns=["value", "period"])
    # x has MultiIndex (id, time); group by the id level.
    for sid, sub in x.groupby(level="id", sort=True):
        if want is not None and sid not in want:
            continue
        vals = sub["value"].to_numpy()
        period = sub["period"].to_numpy()
        x_hist = vals[period == PERIOD_HISTORICAL]
        x_online = vals[period == PERIOD_ONLINE]
        ti = tau_map.get(sid, -1)
        tau_index = None if ti is None or ti < 0 else int(ti)
        yield Series(int(sid), x_hist, x_online, tau_index)


def load_test_targets() -> pd.DataFrame:
    """Per-step targets for the reduced local test set (id, time) -> target."""
    return pd.read_parquet(_path("y_test.reduced.parquet"))
