# Overlapping Tensor-Parallel Attention with Host-Driven Tile Scheduling

This repository implements an experimental distributed attention framework designed to hide the collective communication latency of tensor-parallel self-attention. By restructuring the standard FlashAttention-2 grid layout into a host-driven, tile-fused pipeline, we interleave token-projection computation with non-blocking MPI collectives to achieve compute-communication overlap.

## Project Architecture

In standard Megatron-LM-style tensor parallelism, collective communication (`MPI_Allreduce`) is fully exposed: the attention kernel runs to completion across the entire sequence length, the output projection linear layer ($W_O$) is applied, and only then is the global synchronization barrier hit.

Our project introduces a co-designed attention and communication pipeline:

1. **Host-Driven Query Tiling:** The outer loop over the sequence dimension is managed by the host CPU, which launches discrete attention kernels for individual query row blocks of size $B_M$.
2. **Exact Online Softmax:** Within each query tile kernel, the GPU thread block executes a sequential sweep over Key/Value tiles of size $B_N$, computing exact attention scores via an online softmax reduction.
3. **Pipelining & Overlap:** Once a query tile $i$ finishes compute, it is immediately forwarded to a cuBLAS projection stream ($P_i = O_i \cdot W_O$). As soon as the projection completes, a non-blocking `MPI_Iallreduce` is dispatched on that tile. The communication of tile $i$ runs concurrently over inter-GPU interconnects while the compute stream moves on to calculate attention for tile $i+1$.

```
[Standard Tensor Parallelism]
Compute Entire Attention Block --> Compute Full Projection --> Exposed MPI_Allreduce (Blocking)

[Our Host-Driven Pipelined Architecture]
Tile 0: [FA Kernel] -> [cuBLAS GEMM] -> [MPI_Iallreduce (Async)]
Tile 1:                 [FA Kernel]  -> [cuBLAS GEMM]           -> [MPI_Iallreduce (Async)]
Tile 2:                                  [FA Kernel]            -> [cuBLAS GEMM] ---> ...
```

## Repository Structure

| File | Description |
|------|-------------|
| `flash_attn.cu` / `flash_attn.cuh` | CUDA device functions implementing the tiled online-softmax attention computation |
| `main.cpp` | Orchestration driver managing MPI rank configurations, device streams, cuBLAS execution handles, and the three comparative profiling pathways |
| `Makefile` | Compilation blueprint configured for NERSC Perlmutter environments using the Cray compiler wrappers |

## Requirements

- CUDA Toolkit 11.x+
- CUDA-Aware MPI Implementation (e.g., Cray-MPICH or OpenMPI compiled with CUDA support)
- cuBLAS Library

## Building and Running

To compile the executable:

```bash
make
```

The compiled binary accepts configuration parameters along with an execution mode flag:

```bash
./pipeline_dist_attn <S> <d_h> <H_local> <B_M> <B_N> <mode>
```

For the current prototype kernel, `d_h` and `B_N` must be multiples of 32, and `B_M <= 32` because each query row is assigned one warp.

| Mode | Description |
|------|-------------|
| `1` | Serialized Baseline |
| `2` | Tiled Un-overlapped |
| `3` | Pipelined Overlapped |

Append `--validate` to compare the selected mode against a small naive CPU attention reference on the same input and print a parseable `VALIDATION ...` line. This is intended for small validation shapes, not full benchmark sizes.

### Execution Example

To run a sequence length of 4096 across 4 local GPUs using the overlapped pipelined configuration:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
srun -N 1 -n 4 --gpus-per-task=1 --gpu-bind=closest ./pipeline_dist_attn 4096 64 4 32 128 3
```

### Evaluation Sweep

To collect the core timing data for the report on a Perlmutter GPU allocation:

```bash
bash run_eval_perlmutter.sh
```

The script first captures `results/env.txt`, runs a small validation preflight, then sweeps modes 1-3, sequence lengths 2048/4096/8192, and 1/2/4 ranks on a single GPU node. It writes raw logs to `results/results_mode*.log`, parsed per-run metrics to `results/metrics.csv`, and repeated-run aggregates to `results/summary.csv`.

The intended Perlmutter workflow is:

```bash
bash run_eval_perlmutter.sh
python3 parse_metrics.py results/results_mode*.log > results/metrics.csv
python3 summarize_metrics.py results/metrics.csv > results/summary.csv
```

Override defaults with environment variables, for example:

```bash
REPEATS=5 S_VALUES="4096 8192" bash run_eval_perlmutter.sh
```

Plot generation is intentionally left to the report or notebook layer; `metrics.csv` and `summary.csv` contain the readable data needed for manual plotting.