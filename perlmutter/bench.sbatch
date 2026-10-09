#!/bin/bash
# Full benchmark sweep as a batch job. Submit from the repo root:
#   sbatch -N 1 perlmutter/bench.sbatch        # 4 GPUs, one node
#   sbatch -N 2 perlmutter/bench.sbatch        # 8 GPUs, two nodes
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:45:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -J efa-bench
#SBATCH -o logs/bench-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source perlmutter/env.sh
mkdir -p results logs
RANKS=$((SLURM_JOB_NUM_NODES * 4))
python3 bench_sweep.py --suite all --nodes "$SLURM_JOB_NUM_NODES" --ranks "$RANKS" \
    --out "results/n${SLURM_JOB_NUM_NODES}_r${RANKS}_${SLURM_JOB_ID}.csv"
