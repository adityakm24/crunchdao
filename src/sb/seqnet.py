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

    __slots__ = ("Wih", "Whh", "bih", "bhh", "Wo", "bo", "mean", "scale", "H", "h", "_H2")

    def __init__(self, p: dict) -> None:
        # keep input/hidden biases SEPARATE: the GRU n-gate applies the reset
        # gate r to the hidden bias (b_hn), so b_hn must not be folded into the
        # input side (doing so is only exact when r==1 -> a serve/train mismatch).
        self.Wih = np.ascontiguousarray(p["Wih"], dtype=np.float64)   # (3H, F)
        self.Whh = np.ascontiguousarray(p["Whh"], dtype=np.float64)   # (3H, H)
        self.bih = np.asarray(p["bih"], dtype=np.float64)
        self.bhh = np.asarray(p["bhh"], dtype=np.float64)
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
        gi = self.Wih @ xs + self.bih           # (3H,) input contribution + b_ih
        gh = self.Whh @ h + self.bhh            # (3H,) hidden contribution + b_hh
        r = _sigmoid(gi[:H] + gh[:H])
        z = _sigmoid(gi[H:H2] + gh[H:H2])
        n = np.tanh(gi[H2:] + r * gh[H2:])
        h = (1.0 - z) * n + z * h
        self.h = h
        return float(self.Wo @ h + self.bo)


class StreamingLSTM:
    """O(H) per-step single-layer LSTM + linear head producing a per-step logit.

    Matches torch.nn.LSTM(input, hidden, batch_first=True) exactly (gate order
    i, f, g, o):
        i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
        f = sigmoid(W_if x + b_if + W_hf h + b_hf)
        g = tanh   (W_ig x + b_ig + W_hg h + b_hg)
        o = sigmoid(W_io x + b_io + W_ho h + b_ho)
        c = f * c + i * g ;  h = o * tanh(c)
    with weight_ih = [W_ii; W_if; W_ig; W_io]. Same standardisation/clip as GRU.
    """

    CLIP = 8.0

    __slots__ = ("Wih", "Whh", "bih", "Wo", "bo", "mean", "scale",
                 "H", "_H2", "_H3", "h", "c")

    def __init__(self, p: dict) -> None:
        self.Wih = np.ascontiguousarray(p["Wih"], dtype=np.float64)   # (4H, F)
        self.Whh = np.ascontiguousarray(p["Whh"], dtype=np.float64)   # (4H, H)
        self.bih = np.asarray(p["bih"], dtype=np.float64) + np.asarray(p["bhh"], dtype=np.float64)
        self.Wo = np.asarray(p["Wo"], dtype=np.float64)               # (H,)
        self.bo = float(p["bo"])
        self.mean = np.asarray(p["mean"], dtype=np.float64)
        self.scale = np.asarray(p["scale"], dtype=np.float64)
        self.H = int(self.Whh.shape[1])
        self._H2 = 2 * self.H
        self._H3 = 3 * self.H
        self.h = np.zeros(self.H, dtype=np.float64)
        self.c = np.zeros(self.H, dtype=np.float64)

    def reset(self) -> None:
        self.h = np.zeros(self.H, dtype=np.float64)
        self.c = np.zeros(self.H, dtype=np.float64)

    def step(self, x: np.ndarray) -> float:
        H, H2, H3 = self.H, self._H2, self._H3
        xs = (x - self.mean) / self.scale
        np.clip(xs, -self.CLIP, self.CLIP, out=xs)
        np.nan_to_num(xs, copy=False)
        g = self.Wih @ xs + self.Whh @ self.h + self.bih
        i = _sigmoid(g[:H])
        f = _sigmoid(g[H:H2])
        gg = np.tanh(g[H2:H3])
        o = _sigmoid(g[H3:])
        c = f * self.c + i * gg
        h = o * np.tanh(c)
        self.c = c
        self.h = h
        return float(self.Wo @ h + self.bo)


def make_seqnet(p: dict):
    """Build the right streaming net for an exported weights dict (kind-aware)."""
    kind = str(p["kind"]) if "kind" in p else "gru"
    return StreamingLSTM(p) if kind == "lstm" else StreamingGRU(p)


def forward_sequence(X: np.ndarray, p: dict) -> np.ndarray:
    """Vectorised-per-step forward over a (T, F) sequence -> (T,) logits.

    Used for offline parity checks against the torch reference.
    """
    net = make_seqnet(p)
    out = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        out[i] = net.step(X[i])
    return out
