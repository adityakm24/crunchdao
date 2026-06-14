#!/usr/bin/env bash
# Train extra round-5 neural members back-to-back (one MPS job at a time).
# Launched via run_in_terminal async mode (the terminal tool manages stdin, so no
# nohup/`< /dev/null` dance is needed). Each saves a VAL logit cache for blending.
set -e
cd "$(dirname "$0")/.."

echo "=== [1/3] GRU h112 s3 -> model_026_gru ==="
uv run python scripts/train_seq.py --arch gru --hidden 112 --seed 3 --epochs 14 \
  --out artifacts/models/model_026_gru --val-out features/val_seq_logits_s3.npz

echo "=== [2/3] GRU h144 s4 -> model_027_gru ==="
uv run python scripts/train_seq.py --arch gru --hidden 144 --seed 4 --epochs 14 \
  --out artifacts/models/model_027_gru --val-out features/val_seq_logits_s4.npz

echo "=== [3/3] LSTM h96 s1 -> model_025_lstm ==="
uv run python scripts/train_seq.py --arch lstm --hidden 96 --seed 1 --epochs 12 \
  --out artifacts/models/model_025_lstm --val-out features/val_seq_logits_lstm1.npz

echo "=== SEQ QUEUE DONE ==="
