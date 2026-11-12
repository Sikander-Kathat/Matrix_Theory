#!/usr/bin/env bash
# EE5609 Project 10 - reproduce every result so far (Week 1 + Week 2).
# Usage (from anywhere):  bash ~/ee5609/run_all.sh        (~1-2 min total)
set -e
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || echo "(venv not found - using system python)"

echo "=== Week 1: Monte Carlo vs tail bounds (2M trials) ==="
python montecarlo/week1_montecarlo.py --out results/week1 | tee results/week1/week1_log.txt

echo "=== Week 2a: train + quantize MNIST MLP ==="
python model/week2_train_quantize.py | tee results/week2/train_log.txt

echo "=== Week 2b: real accumulators, w^T Sigma w, bound ladder ==="
python model/week2_accumulators.py | tee results/week2/analysis_log.txt

echo "Done. Figures and tables are in results/week1 and results/week2."
