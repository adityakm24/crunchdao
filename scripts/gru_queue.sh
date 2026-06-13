#!/bin/bash
# Train additional GRU members sequentially AFTER the primary (seed0) finishes.
# Each writes its own model dir + VAL logit cache. MPS runs one at a time.
cd /Users/adityam2/Desktop/Projects/Personal/crunchdao

# wait for the primary GRU run to finish
while pgrep -f "train_seq.py" > /dev/null; do sleep 20; done

echo "=== primary done; training GRU seed1 (hidden 96) ===" >> /tmp/gru_queue.log
uv run python -u scripts/train_seq.py --hidden 96 --epochs 40 --dropout 0.1 \
  --lr 2e-3 --seed 1 --out artifacts/models/model_021_gru \
  --val-out features/val_seq_logits_s1.npz >> /tmp/gru_queue.log 2>&1

echo "=== training GRU seed2 (hidden 160) ===" >> /tmp/gru_queue.log
uv run python -u scripts/train_seq.py --hidden 160 --epochs 40 --dropout 0.2 \
  --lr 1.5e-3 --seed 2 --out artifacts/models/model_022_gru \
  --val-out features/val_seq_logits_s2.npz >> /tmp/gru_queue.log 2>&1

echo "=== queue complete ===" >> /tmp/gru_queue.log
touch /tmp/gru_queue.done
