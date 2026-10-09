# eager-flash-attention

A CUDA/NCCL prototype that overlaps tensor-parallel communication with FlashAttention compute by streaming completed query tiles through the output projection and AllReduce while the rest of attention is still running.

## The idea

In standard tensor-parallel attention, each rank computes its local attention output, projects it through `W_O`, and then participates in an AllReduce to sum across ranks. These three steps run in sequence, so the AllReduce latency sits fully exposed on the critical path:

```
[----- attention -----][- proj -][--- AllReduce ---]
```

FlashAttention computes the output one block of query rows at a time, and each block produces a disjoint slice of output rows. A tile of rows is *finalized* the moment its K/V sweep completes. Because `W_O` is linear and the cross-rank reduction is a sum, projecting and reducing per tile gives the same result as doing it once at the end, but the communication now overlaps with compute:

```
[tile 0 attn][tile 1 attn][tile 2 attn]...
             [proj 0     ][proj 1     ]...
                          [AllReduce 0][AllReduce 1]...
```

## What's in the repo

| File | What it is |
|---|---|
| `flash_attn_tc.cu/.cuh` | Tensor-core FlashAttention forward: fp16 `mma.sync.m16n8k16`, `ldmatrix`, `cp.async` double-buffered K/V, online softmax in the accumulator register layout, optional split-KV with a combine kernel, per-tile completion counters |
| `flash_attn.cu/.cuh` | The original fp32 scalar kernel (one warp per query row), kept as `--kernel v1` for reference |
| `run_drivers.cu/.cuh` | The execution modes compared below |
| `attn_main.cu/.cuh` | MPI/NCCL/cuBLAS setup, synthetic weights, Q/K/V projection, stream/communicator context |
| `attn_tests.cu` | Every (kernel, mode) pair checked against a cuBLAS + materialized-softmax reference on relative error, on 1 or many ranks |
| `attn_bench.cu` | Benchmark driver, prints CSV |
| `bench_sweep.py`, `perlmutter/` | Sweep script, environment/build scripts and sbatch files for NERSC Perlmutter |

### Execution modes

| Mode | What happens |
|---|---|
| `attn` | Attention only, one launch over the whole sequence (the compute floor) |
| `attn_tiled` | Attention only, one launch per tile (what tiling alone costs) |
| `megatron` | One attention launch, one projection GEMM, one AllReduce, chained by events (the standard baseline) |
| `sync` | Per tile: attention, projection, AllReduce, each followed by a host sync (tiling with no overlap) |
| `overlap` | Per tile launches, event-chained across a compute, a projection and `ncomm` communication streams, `nbuf` tiles in flight, no host syncs |
| `stream` | **One** attention launch. Each block bumps a per-tile counter after its rows are globally visible; the projection stream waits on the counter with `cuStreamWaitValue32`, so projection and AllReduce of tile *t* start while later tiles are still computing at full-GPU efficiency |

`stream` is the mode the project is about. The tile boundary is no longer a kernel launch boundary; it is a device-side signal inside a single launch, which is what lets the attention kernel keep the whole GPU busy while the pipeline drains behind it.

## Building and running (Perlmutter)

```
source perlmutter/env.sh          # cudatoolkit/13.2, nccl, cmake, MPICH_GPU_SUPPORT_ENABLED=1
bash perlmutter/build.sh          # cmake + build into ./build
sbatch -N 1 perlmutter/tests.sbatch
sbatch -N 2 perlmutter/bench.sbatch
```

Or interactively:

```
salloc -A m4999_g -C gpu -q interactive -N 1 -G 4 -t 1:00:00
srun -n 4 --gpus-per-node=4 --gpu-bind=none ./build/attn_tests
srun -n 4 --gpus-per-node=4 --gpu-bind=none ./build/attn_bench --S 16384 --tile 2048 --split 4 --ncomm 4 --modes attn,megatron,stream
```

`perlmutter/env.sh` also sets `NCCL_PROTO=Simple`: with NCCL 2.29.2 and the libfabric plugin, inter-node AllReduces of 512 KB or less hung under the LL and LL128 protocols (larger messages and single-node runs were fine). The multi-node tables below were measured before that setting was added; only the small test configurations needed it. With it, `attn_tests` passes 108 configurations on 8 GPUs across 2 nodes.

`attn_bench --help` lists the knobs: `--S --H --d_h` (model shape), `--tile` (rows per pipeline tile), `--split` (split-KV factor), `--ncomm` (NCCL communicators, so up to that many tile AllReduces run concurrently), `--nbuf`, `--kernel tc|v1`, `--causal`, `--proj fp32|tf32|bf16` (projection GEMM precision; bf16 also stores the attention output in bf16), `--gctx_sms N` (reserve N SMs for communication with CUDA green contexts), `--no_proj/--no_comm` (drop a stage to isolate costs).

Status of the newer options (added 2026-09-15 after this project's GPU allocation ran out): `--causal`, `--proj bf16` and `--gctx_sms` pass `attn_tests` (372 configurations) on a single GPU and have single-GPU smoke numbers below, but no multi-GPU measurements yet. `bench_sweep.py` suites `causal`, `bf16`, `comm` and `model` cover them; run `sbatch -N 2 perlmutter/bench.sbatch` when hours are available and `python3 cost_model.py results/<csv>` to fit the AllReduce cost model from the `comm` suite.

## Results

Measured 2026-09-14 on NERSC Perlmutter (A100-SXM4-40GB, 4 per node with NVLink, Slingshot-11 between nodes; CUDA 13.2, NCCL 2.29, cray-mpich 9.1). Model shape unless stated: `H=16`, `d_h=64` (`d=1024`), heads split across ranks. Times are wall-clock milliseconds for the full attention + projection + AllReduce, max over ranks, median of 5 trials. Raw CSVs are in `results/`; regenerate the tables with `python3 bench_tables.py results/*.csv`. The original-code column comes from the pre-rewrite kernel run on the same node type before any changes.

<!-- results:begin (generated by bench_tables.py) -->
### Attention kernel (attention only, 4 GPUs on one node, H=16, d_h=64, 4 local heads per GPU)

| S | original code, one-shot pass incl. proj+AllReduce (May 2026) | v1 kernel, fixed grid | tensor-core kernel | tc speedup vs v1 |
|---|---|---|---|---|
| 1024 | - | 1.442 | 0.031 | 46x |
| 4096 | 383 | 14.0 | 0.176 | 80x |
| 16384 | 6262 | 211.4 | 1.950 | 108x |
| 65536 | - | 3366.9 | 30.6 | 110x |

### Execution modes, 4 GPUs on one node (NVLink)

| S | attn only | megatron | sync | overlap (best tile) | stream (best tile) | stream vs megatron |
|---|---|---|---|---|---|---|
| 4096 | 0.165 | 0.510 | 1.693 | 0.635 (tile 512) | 0.574 (tile 512) | 0.89x |
| 16384 | 1.899 | 2.969 | 5.176 | 3.190 (tile 2048) | 2.947 (tile 2048) | 1.01x |
| 65536 | 30.6 | 33.2 | 38.5 | 35.5 (tile 8192) | 35.5 (tile 8192) | 0.94x |

### Execution modes, 8 GPUs on two nodes (Slingshot between nodes)

| S | attn only | megatron | sync | overlap (best tile) | stream (best tile) | stream vs megatron |
|---|---|---|---|---|---|---|
| 4096 | 0.136 | 0.724 | 3.859 | 1.312 (tile 512) | 0.690 (tile 512) | 1.05x |
| 16384 | 1.073 | 3.136 | 9.261 | 2.881 (tile 2048) | 2.224 (tile 2048) | 1.41x |
| 65536 | 15.1 | 22.0 | 26.4 | 18.4 (tile 8192) | 17.5 (tile 4096) | 1.26x |

### Rank scaling, S=16384, one node

| ranks | attn only | megatron | overlap | stream |
|---|---|---|---|---|
| 1 | 7.114 | 9.575 | 10.9 | 11.2 |
| 2 | 3.611 | 5.610 | 5.718 | 5.023 |
| 4 | 1.899 | 2.969 | 3.190 | 2.947 |

### Llama-7B-shaped layer (H=32, d_h=128, d=4096), S=16384

| ranks | projection | attn only | megatron | overlap (best) | stream (best) | stream vs megatron |
|---|---|---|---|---|---|---|
| 4 | fp32 | 7.966 | 17.2 | 17.3 | 16.8 | 1.03x |
| 4 | tf32 | 7.960 | 11.0 | 11.3 | 10.5 | 1.05x |
| 8 | fp32 | 4.136 | 14.7 | 10.1 | 9.127 | 1.61x |
| 8 | tf32 | 4.135 | 11.4 | 8.638 | 8.583 | 1.32x |

<!-- results:end -->

Reading the tables:

- The kernel rewrite is the large win: two orders of magnitude per launch, and the attention floor at 16k tokens on 4 GPUs is now under 2 ms.
- On one NVLink node the whole projection + AllReduce costs about 1 ms of a 3 ms pass, so there is little to hide and `stream` breaks even with the one-shot baseline. The earlier prototype's per-tile-launch design (`overlap`) is now within 10% of that baseline too once split-KV gives each tile enough blocks.
- Across two nodes the AllReduce is the dominant cost and `stream` recovers most of it: 1.4x faster than one-shot at 16k tokens with the small model, 1.6x for the Llama-shaped layer with an fp32 projection, 1.3x with a TF32 projection. The remaining gap to the attention floor is the communication that cannot be hidden because total AllReduce time exceeds total attention time at these shapes.
- Tile size trades per-AllReduce latency against pipeline tail; S/8 was best everywhere here. Several communicators in flight (`--ncomm 8`) matter across nodes: with one communicator the tile AllReduces serialize and `stream` is slower than one-shot.

## What limits the speedup

These came out of `nsys` traces and the `--no_proj/--no_comm` isolation knobs. They are the reason the modes above look the way they do.

1. **Per-tile launches cannot fill the GPU.** A 512-row tile with 4 local heads is 32 blocks on 108 SMs. Each block then sweeps the whole key range alone on its SM and the tile takes as long as one block. Worse, the schedule is fragile: when an NCCL kernel happened to be resident, the block scheduler packed the attention blocks two per SM and the tile took 2.2x longer. Split-KV (`--split`) fixes the parallelism (blocks per tile x split), which is why `overlap` went from 20 ms to 4.4 ms at S=16k, but a launch per tile still pays a ramp-up and tail per tile.

2. **Wave quantization.** With one big launch, 1024 blocks run four per SM in about 2.4 waves, so without split-KV tiles complete in three bursts rather than steadily, and the comm stream is left with a burst of AllReduces after attention has finished. Split-KV makes blocks ~8x shorter and completions smooth. Grid order matters here: `(head, split)` must be the fast grid index so blocks are dispatched in query-row order; with the split as the slow index the first tile only completed at 85% of the launch.

3. **NCCL blocks need a whole SM.** The ring-LL AllReduce kernel runs 22 blocks of 640 threads at 96 registers with 96 KB of shared memory: 61K of the SM's 64K registers. It cannot share an SM with even one attention block (128 threads x 128 registers, 4 per SM = the full register file), so each tile's AllReduce waits until whole SMs drain. Capping attention occupancy (`--reserve`) did not help because the freed slot is still too small; `NCCL_MAX_NCHANNELS=4` (fewer, hence fewer SMs) helped `stream` by ~10% on one node but slows the one-shot AllReduce.

4. **Small AllReduces are latency-bound.** On one node the 64 MB one-shot AllReduce takes ~0.5 ms; 32 AllReduces of 2 MB take ~55 us each when alone (1.8 ms total) and more when contending. So on NVLink there is at most ~1 ms of exposed communication to recover and streaming roughly breaks even. Across two nodes the per-op cost is ~5x higher, the one-shot AllReduce is ~1.8 ms, and streaming with several communicators in flight (`--ncomm 4..8`) is where the design pays off.

## Next experiments

Implemented, tested for correctness, and smoke-timed on a single A100-PCIe login-node GPU (shared with other users, so treat the numbers as indicative only; multi-GPU sweeps are pending GPU hours):

- **Causal masking** (`--causal`). Attention only, 16k tokens, 16 heads on one GPU: 10.3 ms full vs 4.4 ms causal. Tile *t* now costs work proportional to *t*, so early tiles finish early and the projection/AllReduce pipeline gets a head start; expect `stream` to gain relative to one-shot, which the `causal` sweep suite measures. Split-KV hurts causal on one GPU (7.0 ms with `--split 4`): high splits have idle blocks for early rows and the combine still reads every split, so the split should become row-dependent.
- **bf16 projection** (`--proj bf16`). The attention kernel and the split-KV combine write bf16 directly and the projection is a bf16 tensor-core GEMM with fp32 accumulation. Llama-shaped layer (H=32, d_h=128), 4k tokens on one GPU, one-shot pass: 16.8 ms with the fp32 projection, 3.8 ms with TF32, 2.7 ms with bf16, against a 2.0 ms attention floor. Relative error against the fp32 reference stays under 4e-3.
- **Green-context SM partitioning** (`--gctx_sms N`, optionally `--gctx_proj`). Directly targets finding 3 above: the communication streams get their own SMs so NCCL blocks never wait for the attention kernel to drain an SM, at the price of `N` fewer SMs for attention. The partition is granted at the requested size (8 and 16 SMs verified) and all modes run on it. Sweep `N` in {8, 16, 24} with `--ncomm` and `NCCL_MAX_NCHANNELS <= N` on 2 nodes.
- **Cost model** (`cost_model.py`, `comm` suite). Fits `t_allreduce(bytes) = alpha + bytes / bandwidth` from AllReduce-only timings and predicts `stream` time as the larger of attention, `n * alpha / ncomm + bytes / bandwidth`, and projection, plus the last tile's tail. Compare its predictions with the measured `stream` rows to see how far the pipeline is from the model's floor.

Not yet implemented: 4-node scaling with the same sweep script (`sbatch -N 4`), a copy-engine AllReduce for the single-node case, row-dependent split-KV for causal, and fusing the combine into the projection GEMM prologue.

## Related work

The pattern of overlapping tensor-parallel collectives with compute is implemented in production by NVIDIA Transformer Engine's `tp_comm_overlap` (userbuffers + persistent kernels), academically by CoCoNet and Flux, and in the FlashAttention repo's own `RowParallelLinear`. This prototype targets one step earlier in the pipeline: the seam between FlashAttention tile completion and the start of projection, exposed as a device-side completion counter that the projection stream waits on rather than as a fused kernel.
