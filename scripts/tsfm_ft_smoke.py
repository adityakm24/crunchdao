"""De-risk probe: confirm we can fine-tune the Chronos-T5-mini ENCODER end-to-end
for break detection (tokenize -> encoder -> pooled head -> BCE -> grads flow back
into encoder weights). Tiny + fast; just validates the API before the real build.

Run: KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 uv run python scripts/tsfm_ft_smoke.py
"""
from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")

MODEL = "models/chronos-t5-mini"


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev}")

    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(MODEL, device_map=dev, torch_dtype=torch.float32)
    tok = pipe.tokenizer
    t5 = pipe.model.model               # T5ForConditionalGeneration
    enc = t5.encoder
    d_model = t5.config.d_model
    print(f"encoder loaded, d_model={d_model}, "
          f"enc_params={sum(p.numel() for p in enc.parameters()):,}")

    # two toy windows: one stationary, one with a level shift midway
    rng = np.random.default_rng(0)
    w0 = rng.standard_normal(120).astype(np.float32)
    w1 = rng.standard_normal(120).astype(np.float32); w1[60:] += 2.0
    ctx = [torch.tensor(w0), torch.tensor(w1)]
    ctx_pad = torch.nn.utils.rnn.pad_sequence(ctx, batch_first=True)  # (2, L)

    # chronos tokenizer: context -> (token_ids, attn_mask, scale)
    token_ids, attn_mask, scale = tok.context_input_transform(ctx_pad)
    print(f"token_ids {tuple(token_ids.shape)} {token_ids.dtype}  "
          f"attn_mask {tuple(attn_mask.shape)}  scale {tuple(scale.shape)}")

    token_ids = token_ids.to(dev)
    attn_mask = attn_mask.to(dev)

    head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1)).to(dev)

    enc.train(); head.train()
    out = enc(input_ids=token_ids, attention_mask=attn_mask)
    hidden = out.last_hidden_state                       # (B, L, d)
    m = attn_mask.unsqueeze(-1).float()
    pooled = (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)
    logit = head(pooled).squeeze(-1)                     # (B,)
    print(f"hidden {tuple(hidden.shape)}  pooled {tuple(pooled.shape)}  logit {logit.detach().cpu().numpy()}")

    target = torch.tensor([0.0, 1.0], device=dev)
    loss = nn.functional.binary_cross_entropy_with_logits(logit, target)
    loss.backward()

    # confirm grads flow into the ENCODER (end-to-end), not just the head
    g_enc = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0)
                for p in enc.parameters())
    g_head = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0)
                 for p in head.parameters())
    print(f"loss={loss.item():.4f}  grad_sum(encoder)={g_enc:.4e}  grad_sum(head)={g_head:.4e}")
    assert g_enc > 0, "no gradient reached the encoder -> end-to-end FT not wired"
    print("OK: end-to-end fine-tuning is viable (grads reach encoder).")


if __name__ == "__main__":
    main()
