#!/bin/bash
set -euo pipefail

mkdir -p build
cd build
cmake ..
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
        srun -n ${NGPUS} ${BIN} \
            --mode ${MODE} \
            --num_tiles ${TILES} --tile_elems ${ELEMS} --fma_iters ${FMA}
    done

    echo ""
    echo "===== overlap, ${NGPUS} GPU(s) ====="
    srun -n ${NGPUS} ${BIN} \
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
    echo ""
    echo "===== overlap, aggregate_tiles=${AGG} ====="
    srun -n 4 ${BIN} \
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
    srun -n 4 ${BIN} \
        --mode overlap \
        --num_tiles ${TILES} --tile_elems ${ELEMS} --fma_iters ${FMA} \
        --num_buffers ${INF} --max_inflight ${INF} --aggregate_tiles 1
done

echo ""
echo "Done. CSV timelines: results/tile_timeline_*.csv"
