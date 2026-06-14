"""Fast standalone parity test for the numpy LSTM/GRU implementations.

Builds a tiny torch nn.LSTM / nn.GRU with random weights on CPU, exports the
weights in the same layout scripts/train_seq.py uses, then checks that both
(a) scripts.train_seq.numpy_lstm_forward / numpy_gru_forward and
(b) sb.seqnet.StreamingLSTM / StreamingGRU
reproduce the torch per-step output to < 1e-10. Validates the gate-order mapping
the submission relies on BEFORE waiting for a full training run.

Run: uv run python scripts/test_seq_parity.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import torch
import torch.nn as nn

from sb.seqnet import StreamingLSTM, StreamingGRU
from train_seq import numpy_lstm_forward, numpy_gru_forward


def export(net, head, mean, scale, kind):
    sd = net.state_dict()
    return dict(
        Wih=sd["weight_ih_l0"].numpy().astype(np.float64),
        Whh=sd["weight_hh_l0"].numpy().astype(np.float64),
        bih=sd["bias_ih_l0"].numpy().astype(np.float64),
        bhh=sd["bias_hh_l0"].numpy().astype(np.float64),
        Wo=head.weight.detach().numpy()[0].astype(np.float64),
        bo=float(head.bias.detach().numpy()[0]),
        mean=mean, scale=scale, kind=kind,
        feature_names=np.array([f"f{i}" for i in range(len(mean))]),
    )


def run(kind, F=7, H=5, T=40, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rnn = nn.LSTM(F, H, batch_first=True) if kind == "lstm" else nn.GRU(F, H, batch_first=True)
    head = nn.Linear(H, 1)
    rnn.double()
    head.double()
    mean = rng.normal(size=F)
    scale = np.abs(rng.normal(size=F)) + 0.5

    Xraw = rng.normal(size=(T, F)) * 2.0
    # standardise + clip exactly as serving does
    Xs = np.clip((Xraw - mean) / scale, -8.0, 8.0)
    with torch.no_grad():
        out, _ = rnn(torch.from_numpy(Xs.astype(np.float64))[None])
        tch = head(out)[0, :, 0].numpy().astype(np.float64)

    p = export(rnn, head, mean, scale, kind)
    fwd = numpy_lstm_forward if kind == "lstm" else numpy_gru_forward
    npy = fwd(Xraw, p)  # forward re-standardises internally

    name_to_col = {f"f{i}": i for i in range(F)}
    cls = StreamingLSTM if kind == "lstm" else StreamingGRU
    net = cls(p, name_to_col) if cls.__init__.__code__.co_argcount == 3 else cls(p)
    stream = np.array([net.step(Xraw[i]) for i in range(T)])

    d_fwd = float(np.abs(npy - tch).max())
    d_str = float(np.abs(stream - tch).max())
    print(f"{kind.upper():4s}  numpy_forward max|diff|={d_fwd:.2e}   "
          f"Streaming{kind.upper()} max|diff|={d_str:.2e}")
    assert d_fwd < 1e-10 and d_str < 1e-10, f"{kind} parity FAILED"


def main():
    # seqnet.StreamingGRU takes only p; submission variant takes (p, name_to_col).
    # The reference seqnet classes take (p,) for GRU and (p,) for LSTM, so adapt:
    for kind in ("gru", "lstm"):
        run(kind)
    print("ALL PARITY CHECKS PASSED (< 1e-10)")


if __name__ == "__main__":
    main()
