#!/usr/bin/env bash
set -euo pipefail

# Run from an interactive Perlmutter GPU allocation or submit with sbatch after
# adding your project account, e.g. "#SBATCH -A <account>".
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:30:00

cd "$(dirname "$0")"

export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
# Disable Cray MPICH CUDA IPC by default to avoid cuIpcOpenMemHandle failures
# observed on full 4-GPU Perlmutter node runs with strict Slurm GPU binding.
export MPICH_GPU_IPC_ENABLED="${MPICH_GPU_IPC_ENABLED:-0}"

if [[ -z "${CUDA_HOME:-}" && -n "${CUDATOOLKIT_HOME:-}" ]]; then
    export CUDA_HOME="${CUDATOOLKIT_HOME}"
fi

make

S_VALUES=(${S_VALUES:-2048 4096 8192})
RANK_VALUES=(${RANK_VALUES:-1 2 4})
MODES=(${MODES:-1 2 3})

D_H="${D_H:-64}"
H_LOCAL="${H_LOCAL:-4}"
B_M="${B_M:-32}"
B_N="${B_N:-128}"
REPEATS="${REPEATS:-3}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
VALIDATION_S="${VALIDATION_S:-128}"
VALIDATION_RANK_VALUES=(${VALIDATION_RANK_VALUES:-${RANK_VALUES[*]}})

mkdir -p results
rm -f results/results_mode*.log results/validation.log results/metrics.csv results/summary.csv

{
    echo "date=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "pwd=$(pwd)"
    echo "CUDA_HOME=${CUDA_HOME:-}"
    echo "CUDATOOLKIT_HOME=${CUDATOOLKIT_HOME:-}"
    echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-}"
    echo "MPICH_GPU_IPC_ENABLED=${MPICH_GPU_IPC_ENABLED:-}"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-}"
    echo "SLURM_GPUS=${SLURM_GPUS:-}"
    echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
    echo
    echo "## module list"
    module list 2>&1 || true
    echo
    echo "## nvidia-smi -L"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L || true
    fi
    echo
    echo "## srun nvidia-smi -L"
    srun -N 1 -n 1 --gpus-per-task=1 nvidia-smi -L || true
} > results/env.txt

run_case() {
    local log_file="$1"
    local ranks="$2"
    local S="$3"
    local mode="$4"
    local repeat="$5"
    local validate_arg="${6:-}"

    echo "RUN mode=${mode} ranks=${ranks} S=${S} d_h=${D_H} H_local=${H_LOCAL} B_M=${B_M} B_N=${B_N} repeat=${repeat} validate=${validate_arg:-0}" | tee -a "${log_file}"
    srun -N 1 -n "${ranks}" --gpus-per-task=1 --gpu-bind=closest \
        ./pipeline_dist_attn "${S}" "${D_H}" "${H_LOCAL}" "${B_M}" "${B_N}" "${mode}" ${validate_arg} \
        2>&1 | tee -a "${log_file}"
    echo | tee -a "${log_file}"
}

if [[ "${RUN_VALIDATION}" == "1" ]]; then
    validation_log="results/validation.log"
    {
        echo "# validation preflight"
        echo "# VALIDATION_S=${VALIDATION_S}"
        echo "# VALIDATION_RANK_VALUES=${VALIDATION_RANK_VALUES[*]}"
        echo "# D_H=${D_H} H_LOCAL=${H_LOCAL} B_M=${B_M} B_N=${B_N}"
        echo
    } > "${validation_log}"

    for ranks in "${VALIDATION_RANK_VALUES[@]}"; do
        for mode in "${MODES[@]}"; do
            run_case "${validation_log}" "${ranks}" "${VALIDATION_S}" "${mode}" "validation" "--validate"
        done
    done
fi

for mode in "${MODES[@]}"; do
    log_file="results/results_mode${mode}.log"
    {
        echo "# mode=${mode}"
        echo "# S_VALUES=${S_VALUES[*]}"
        echo "# RANK_VALUES=${RANK_VALUES[*]}"
        echo "# D_H=${D_H} H_LOCAL=${H_LOCAL} B_M=${B_M} B_N=${B_N} REPEATS=${REPEATS}"
        echo
    } > "${log_file}"

    for ranks in "${RANK_VALUES[@]}"; do
        for S in "${S_VALUES[@]}"; do
            for repeat in $(seq 1 "${REPEATS}"); do
                run_case "${log_file}" "${ranks}" "${S}" "${mode}" "${repeat}"
            done
        done
    done
done

python3 parse_metrics.py results/results_mode*.log > results/metrics.csv
python3 summarize_metrics.py results/metrics.csv > results/summary.csv
echo "Wrote results/metrics.csv"
echo "Wrote results/summary.csv"
