"""Deterministic pure-numpy GRU sequence member for streaming inference.

The GRU is trained offline (scripts/train_seq.py, PyTorch) and its weights are
exported to numpy. At competition inference time we re-implement the exact
single-layer GRU recurrence in numpy so the scored output is fully deterministic
(no torch, no RNG, no threading nondeterminism) and O(H) per step.

Matches torch.nn.GRU(input, hidden, batch_first=True) exactly:
    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    h = (1 - z) * n + z * h
with weight_ih = [W_ir; W_iz; W_in], weight_hh = [W_hr; W_hz; W_hn].
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class StreamingGRU:
    """O(H) per-step GRU + linear head producing a per-step logit.

    Input features are z-scored with the frozen (train-split) mean/scale, NaN
    sanitised and clipped to +-CLIP before being fed to the GRU, exactly as in
    training. Call ``reset()`` at the start of each series, then ``step(x)`` per
    observation with the raw (un-standardised) feature vector ``x``.
    """

    CLIP = 8.0

    __slots__ = ("Wih", "Whh", "bih", "Wo", "bo", "mean", "scale", "H", "h", "_H2")

    def __init__(self, p: dict) -> None:
        # combine input/hidden biases (GRU adds them) for fewer ops
        self.Wih = np.ascontiguousarray(p["Wih"], dtype=np.float64)   # (3H, F)
        self.Whh = np.ascontiguousarray(p["Whh"], dtype=np.float64)   # (3H, H)
        self.bih = np.asarray(p["bih"], dtype=np.float64) + np.asarray(p["bhh"], dtype=np.float64)
        self.Wo = np.asarray(p["Wo"], dtype=np.float64)               # (H,)
        self.bo = float(p["bo"])
        self.mean = np.asarray(p["mean"], dtype=np.float64)
        self.scale = np.asarray(p["scale"], dtype=np.float64)
        self.H = int(self.Whh.shape[1])
        self._H2 = 2 * self.H
        self.h = np.zeros(self.H, dtype=np.float64)

    def reset(self) -> None:
        self.h = np.zeros(self.H, dtype=np.float64)

    def step(self, x: np.ndarray) -> float:
        H, H2 = self.H, self._H2
        xs = (x - self.mean) / self.scale
        np.clip(xs, -self.CLIP, self.CLIP, out=xs)
        np.nan_to_num(xs, copy=False)
        h = self.h
        g = self.Wih @ xs + self.bih          # (3H,) input contribution + bias
        gh = self.Whh @ h                      # (3H,) hidden contribution
        r = _sigmoid(g[:H] + gh[:H])
        z = _sigmoid(g[H:H2] + gh[H:H2])
        n = np.tanh(g[H2:] + r * gh[H2:])
        h = (1.0 - z) * n + z * h
        self.h = h
        return float(self.Wo @ h + self.bo)


def forward_sequence(X: np.ndarray, p: dict) -> np.ndarray:
    """Vectorised-per-step forward over a (T, F) sequence -> (T,) logits.

    Used for offline parity checks against the torch reference.
    """
    gru = StreamingGRU(p)
    out = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        out[i] = gru.step(X[i])
    return out
