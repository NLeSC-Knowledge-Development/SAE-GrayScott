#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=../../../.venv/bin/python

echo "=== START REST $(date) ==="
$PY train_sae.py --data data/gray_scott_mixed.npz --model-dir model \
  --figures figures --layer blocks.6 --n-latent 512 --k 16 --epochs 120 --lr 5e-4 \
  --train-states 300 --val-states 60 --stick-share 0.5 --seed 0 --threads 6
echo "=== SAE DONE $(date) ==="

$PY interpret.py --data data/gray_scott_mixed.npz --model-dir model \
  --figures figures --regime 0 --states 120 --tag main --threads 6
$PY interpret.py --data data/gray_scott_mixed.npz --model-dir model \
  --figures figures --regime 1 --states 54 --tag sticks --threads 6
echo "=== INTERPRET DONE $(date) ==="

$PY component_study.py --data data/gray_scott_mixed.npz --model-dir model \
  --figures figures --permutations 1000 --bootstrap 1000 \
  --probe-layers blocks.4,blocks.5,blocks.6 --threads 6
echo "=== COMPONENT DONE $(date) ==="

$PY push_study.py --data data/gray_scott_mixed.npz --model-dir model \
  --figures figures --pairs 120 --sites 3 --calls 20 --seed 0
$PY control_study.py --data data/gray_scott_mixed.npz --model-dir model \
  --figures figures --pairs 120 --sites 3 --calls 20 --seed 0
echo "=== STICK DONE $(date) ==="

$PY animate.py --data data/gray_scott_mixed.npz --model-dir model \
  --figures figures --only rollout,feature,injection,v-injection --fps 10
echo "=== ANIMATE DONE $(date) ==="
