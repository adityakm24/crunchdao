"""De-risk: validate StreamingAttn's numpy forward against torch BEFORE the real
weights land. Builds a random-init AttnNet (the exact train_attn architecture),
saves it in the serve weight format, so serve_attn.py parity can compare the
float64 numpy streaming forward to the torch full-sequence forward. Random weights
exercise the same math as trained weights -> a clean reimpl test. CPU only.
"""
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")

F, d, heads, layers, ff = 151, 128, 8, 3, 512
DROP = {"t", "log_n_hist", "chi2_cum", "online_acf2", "acf2_diff",
        "w25_chi2", "w50_chi2", "w100_chi2", "w200_chi2", "w400_chi2", "log_t"}


class AttnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(F, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ff, dropout=0.1,
                                         batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(d, 1)


torch.manual_seed(0)
net = AttnNet()
d4 = np.load("features/train_features_v4.npz", allow_pickle=True)
names = [str(n) for n in d4["feature_names"]]
keep = [i for i, n in enumerate(names) if n not in DROP]
assert len(keep) == F, f"keep={len(keep)} != F={F}"
feat_names = [names[i] for i in keep]

meta = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()}
# random non-trivial mean/scale to exercise the normalization path too
rng = np.random.default_rng(1)
meta["__mean"] = rng.normal(0, 0.5, len(keep))
meta["__scale"] = np.abs(rng.normal(1, 0.2, len(keep))) + 0.5
meta["__feature_names"] = np.array(feat_names, dtype="U64")
meta["__config"] = np.array([F, d, heads, layers, ff], dtype=np.int64)
np.savez("/tmp/attn_rand.npz", **meta)
print(f"saved /tmp/attn_rand.npz  (F={F} keep={len(keep)})")
