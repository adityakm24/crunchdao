"""Fast standalone + rank-corr probe for attention context-length variants.

Prints, for each cached VAL attention logit file, the standalone TS-AUC and the
Spearman rank-corr to the 3-GRU mean (the decorrelation axis). NO grid search —
this is the quick read on whether longer context raises the single-model member.

Run:
  uv run python scripts/attn_probe.py \
      ml320=features/val_attn_s1.npz ml512=features/val_attn_s1_ml512.npz \
      ml224=features/val_attn_s1_ml224.npz
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
from sb.metric import ts_auc_grouped

SPLIT_SEED = 42


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt(ra @ ra) * np.sqrt(rb @ rb) + 1e-12))


def main():
    pairs = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        pairs.append((name, path))
    if not pairs:
        pairs = [("ml320", "features/val_attn_s1.npz")]

    base = np.load("features/val_base_logits.npz")
    sv, tv = base["series_id"], base["t_online"]

    d = np.load("features/train_features_v4.npz", allow_pickle=True)
    sid, t, y = d["series_id"], d["t_online"], d["y"]
    uniq = np.unique(sid); rng = np.random.default_rng(SPLIT_SEED); rng.shuffle(uniq)
    va = np.isin(sid, list(set(uniq[: int(0.2 * len(uniq))].tolist())))
    yv_lk = {(int(a), int(b)): int(c) for a, b, c in zip(sid[va], t[va], y[va])}
    yv = np.array([yv_lk[(int(a), int(b))] for a, b in zip(sv, tv)])

    grus = [np.load(f"features/val_seq_logits_{m}_nolog.npz")["val_logit"]
            for m in ("020", "021", "022")]
    gru_mean = np.mean(grus, axis=0)

    print(f"gru_mean standalone VAL TS-AUC = {ts_auc_grouped(tv, yv, gru_mean):.4f}\n")
    print(f"{'member':8s} {'standalone':>10s} {'rankcorr_gru':>13s}")
    logits = {"gru": gru_mean}
    for name, path in pairs:
        a = np.load(path)
        al = a["val_logit"]
        if "series_id" in a.files and not (
            np.array_equal(a["series_id"], sv) and np.array_equal(a["t_online"], tv)
        ):
            key = {(int(s), int(tt)): float(v)
                   for s, tt, v in zip(a["series_id"], a["t_online"], al)}
            al = np.array([key[(int(s), int(tt))] for s, tt in zip(sv, tv)])
        sa = ts_auc_grouped(tv, yv, al)
        rc = _spearman(al, gru_mean)
        logits[name] = al
        print(f"{name:8s} {sa:10.4f} {rc:13.3f}")

    names = list(logits.keys())
    print("\nrank-corr matrix:")
    print("        " + "  ".join(f"{n:>6s}" for n in names))
    for ni in names:
        cells = "  ".join(f"{_spearman(logits[ni], logits[nj]):6.3f}" for nj in names)
        print(f"  {ni:6s} {cells}")


if __name__ == "__main__":
    main()
