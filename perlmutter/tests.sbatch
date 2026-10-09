#!/bin/bash
# Correctness tests on 1 or 2 nodes:  sbatch -N 1 perlmutter/tests.sbatch
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -J efa-tests
#SBATCH -o logs/tests-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source perlmutter/env.sh
mkdir -p logs
srun --gpu-bind=none ./build/attn_tests
