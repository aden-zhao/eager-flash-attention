#!/bin/bash
set -euo pipefail

# Perlmutter / Cray MPICH CUDA-aware MPI settings.
# GPU-aware MPI must be enabled because tile_comm passes device pointers to MPI.
# GPU IPC is disabled because Cray MPICH CUDA IPC was failing on this environment.
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0

mkdir -p build results

cd build
CC=cc CXX=CC cmake ..
cmake --build . -j
cd ..

# CLI reference:
#   --mode             serialized_full | serialized | tiled | overlap
#   --num_tiles        Number of sequence tiles (default 8)
#   --tile_elems       Pre-projection tile size (default 1048576)
#   --num_buffers      Pipeline depth, overlap mode only (default 2)
#   --max_inflight     Max concurrent MPI collectives (default 1)
#   --aggregate_tiles  Tiles per MPI message in overlap mode (default 1)
#   --fma_iters        Synthetic compute delay per elem (default 200)
#   --proj_m/k/n       Projection dimensions

TILES=8
ELEMS=1048576
FMA=200
BIN=./build/tile_comm

# Common srun options:
# one MPI rank per GPU, with Slurm binding each rank to one GPU.
SRUN_GPU_FLAGS="--gpus-per-task=1 --gpu-bind=single:1"

# Experiment 1: Does overlap help?
#   Compare serialized_full, serialized, tiled, overlap
#   Strong scaling across 1, 2, 4 GPUs

echo "###############################################"
echo "# Experiment 1: Mode comparison + strong scaling"
echo "###############################################"

for NGPUS in 1 2 4; do
    for MODE in serialized_full serialized tiled; do
        echo ""
        echo "===== ${MODE}, ${NGPUS} GPU(s) ====="
        srun -n ${NGPUS} ${SRUN_GPU_FLAGS} ${BIN} \
            --mode ${MODE} \
            --num_tiles ${TILES} --tile_elems ${ELEMS} --fma_iters ${FMA}
    done

    echo ""
    echo "===== overlap, ${NGPUS} GPU(s) ====="
    srun -n ${NGPUS} ${SRUN_GPU_FLAGS} ${BIN} \
        --mode overlap \
        --num_tiles ${TILES} --tile_elems ${ELEMS} --fma_iters ${FMA} \
        --num_buffers 2 --max_inflight 1 --aggregate_tiles 1
done

# Experiment 2: Does aggregation help?
#   Fix num_buffers=4, max_inflight=1, vary aggregate_tiles

echo ""
echo "###############################################"
echo "# Experiment 2: Aggregation sweep (4 GPUs)"
echo "###############################################"

for AGG in 1 2 4; do
    if (( TILES % AGG != 0 )); then
        echo ""
        echo "===== skipping aggregate_tiles=${AGG}: TILES=${TILES} not divisible by AGG ====="
        continue
    fi
    echo ""
    echo "===== overlap, aggregate_tiles=${AGG} ====="
    srun -n 4 ${SRUN_GPU_FLAGS} ${BIN} \
        --mode overlap \
        --num_tiles ${TILES} --tile_elems ${ELEMS} --fma_iters ${FMA} \
        --num_buffers 4 --max_inflight 1 --aggregate_tiles ${AGG}
done

# Experiment 3: Does MPI concurrency help?
#   Fix aggregate_tiles=1, vary max_inflight (= num_buffers)

echo ""
echo "###############################################"
echo "# Experiment 3: In-flight concurrency sweep (4 GPUs)"
echo "###############################################"

for INF in 1 2 4; do
    echo ""
    echo "===== overlap, max_inflight=${INF} ====="
    srun -n 4 ${SRUN_GPU_FLAGS} ${BIN} \
        --mode overlap \
        --num_tiles ${TILES} --tile_elems ${ELEMS} --fma_iters ${FMA} \
        --num_buffers ${INF} --max_inflight ${INF} --aggregate_tiles 1
done

echo ""
echo "Done. CSV timelines: results/tile_timeline_*.csv"
