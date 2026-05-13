# Project Handoff

## Current State

This repository implements a CS 5220 prototype for host-driven tiled attention with tensor-parallel output projection and MPI allreduce. The main experiment compares three modes:

- Mode 1: serialized baseline, computing all attention tiles, then full projection, then blocking `MPI_Allreduce`.
- Mode 2: tiled but un-overlapped, computing/projection per tile and allreducing after the tile loop.
- Mode 3: pipelined overlapped design, using per-tile buffers, CUDA events, cuBLAS projection, and non-blocking `MPI_Iallreduce`.

The implementation is in `main.cpp`, `flash_attn.cu`, and `flash_attn.cuh`. The Perlmutter experiment harness is `run_eval_perlmutter.sh`. Data parsing and aggregation are handled by `parse_metrics.py` and `summarize_metrics.py`.

## Validation Status

The latest validation path compares GPU output against an independent naive CPU attention reference for small shapes. It computes standard attention as:

`softmax(QK^T / sqrt(d_h)) V`, followed by output projection with `W_O`, then scales by `num_ranks` to match the allreduce result.

Observed validation output:

```text
VALIDATION status=pass mode=3 max_abs_err=0.000001 max_rel_err=0.000001
```

Validation command used on Perlmutter:

```bash
srun -N 1 -n 4 --gpus-per-task=1 --gpu-bind=closest \
  ./pipeline_dist_attn 128 64 4 32 128 3 --validate
```

Report phrasing: the pipelined implementation was validated against an independent CPU reference for a four-rank small case with max absolute and relative error around `1e-6`.

## Perlmutter Environment

Main experiment data was collected on one Perlmutter GPU node with 4 A100-SXM4-40GB GPUs.

Important environment details from `results/env.txt`:

- CUDA: `cudatoolkit/12.9`
- Cray MPICH: `cray-mpich/9.0.1`
- Accelerator target: `craype-accel-nvidia80`
- `MPICH_GPU_SUPPORT_ENABLED=1`
- `MPICH_GPU_IPC_ENABLED=0`

`MPICH_GPU_IPC_ENABLED=0` was used to avoid observed Cray MPICH CUDA IPC failures at 4 ranks. Mention this as a methodology caveat because it may affect intra-node communication performance.

## Data Artifacts

Main sweep:

- `results/metrics.csv`: per-run rows for modes 1-3, ranks 1/2/4, and `S = 2048, 4096, 8192`.
- `results/summary.csv`: aggregated mean/min/max/stddev over 3 repeats.
- `results/env.txt`: Perlmutter environment capture.

Tile sweep:

- `results_tiles/tile_metrics.csv`: per-run rows for `S=4096`, `ranks=4`, `B_M = 8, 16, 32`.
- `results_tiles/tile_summary.csv`: aggregated tile sweep statistics.

Large-parameter sweep:

- `results_large/large_metrics.csv`: per-run rows for larger model-like settings, including `d_h=128`, `H_local=8`, and `H_local=16`.
- `results_large/large_summary.csv`: aggregated large-run statistics.

Raw logs are intentionally not needed for plotting/reporting. The CSV files are sufficient.

## Headline Results

At `S=4096`, `B_M=32`, `B_N=128`, ranks 1/2/4:

- Mode 1 wall time: `410.23 ms`, `445.15 ms`, `459.30 ms`.
- Mode 2 wall time: `419.39 ms`, `427.93 ms`, `489.66 ms`.
- Mode 3 wall time: `412.95 ms`, `500.38 ms`, `663.14 ms`.

Interpretation: the overlapped mode did not outperform the serialized or tiled baselines in this prototype. This is still useful: it shows the intended overlap was dominated by host scheduling, stream synchronization, MPI progress behavior, and launch overheads.

Tile sweep at `S=4096`, `ranks=4`:

- Mode 1: `B_M=8` to `B_M=32` improves from `1102.01 ms` to `465.38 ms`.
- Mode 2: `B_M=8` to `B_M=32` improves from `1047.95 ms` to `491.24 ms`.
- Mode 3: `B_M=8` to `B_M=32` improves from `1825.74 ms` to `629.79 ms`.

Interpretation: smaller host-driven tiles substantially increase overhead. This strongly supports the report section on launch overhead and command scheduling bottlenecks.

Large-parameter sweep:

- At `d_h=128`, `H_local=8`, `ranks=4`, `d_model=4096`:
  - `S=2048`: Mode 1 `419.20 ms`, Mode 2 `380.60 ms`, Mode 3 `415.63 ms`.
  - `S=4096`: Mode 1 `1077.09 ms`, Mode 2 `1096.86 ms`, Mode 3 `1234.54 ms`.
  - `S=8192`: Mode 1 `3830.61 ms`, Mode 2 `3914.81 ms`, Mode 3 `4205.28 ms`.
- At `S=4096`, `d_h=128`, `H_local=16`, `ranks=4`, `d_model=8192`:
  - Mode 1 `1201.92 ms`, Mode 2 `1242.20 ms`, Mode 3 `1366.82 ms`.

Interpretation: increasing the model dimensions makes the experiment more credible, but it does not rescue the overlapped mode. Mode 3 remains slower than Mode 1/2, which suggests the bottleneck is the host/runtime orchestration pattern rather than only insufficient model size.

Estimated Mode 3 overhead relative to Mode 1:

- `S=4096`, `H_local=8`: `(1234.54 - 1077.09) / 128 ~= 1.23 ms/tile`.
- `S=4096`, `H_local=16`: `(1366.82 - 1201.92) / 128 ~= 1.29 ms/tile`.
- `S=8192`, `H_local=8`: `(4205.28 - 3830.61) / 256 ~= 1.46 ms/tile`.

This consistent `1.2-1.5 ms/tile` penalty is a key conclusion. Phrase it carefully: increasing per-launch work did not materially reduce Mode 3's relative overhead. Avoid claiming exact collective latency unless separately measured; the current `exposed_ms` metric does not isolate communication accurately.

## Report Framing

Recommended thesis:

The host-driven pipelined layout is conceptually valid and numerically correct, but the prototype shows that fine-grained host scheduling can be too expensive before communication overlap pays off. In the measured Perlmutter runs, the overlapped design was slower than simpler baselines, and the tile sweep showed that smaller `B_M` values greatly increased runtime. Larger runs at `d_h=128` and `d_model=4096/8192` still showed an approximately constant per-tile Mode 3 overhead, suggesting the key issue is architectural host/runtime orchestration rather than just GPU underutilization.

Useful report points:

- Explain the online softmax variables `m` and `ell` and why they allow exact tiled attention.
- Compare modes 1, 2, and 3 as progressively more tiled/asynchronous designs.
- Use `results/summary.csv` for strong scaling and sequence-length plots.
- Use `results_tiles/tile_summary.csv` for host launch overhead discussion.
- Use `results_large/large_summary.csv` as the main evidence that larger model-like dimensions still do not make the current host-driven overlap scheme win.
- Use validation output to establish correctness against an independent CPU reference.
- Be explicit that `MPICH_GPU_IPC_ENABLED=0` was used as a Perlmutter stability workaround.

## Remaining Work

- Generate report plots manually from the CSV files.
- Compute any final percentage comparisons needed for captions.
- Write the analytical model section. At minimum, use `message_bytes` and `total_reduced_bytes` columns as the communication-volume basis for an alpha-beta comparison.
- Discuss why CUDA event time tracks wall time closely in the current instrumentation, making exposed communication time appear very small.
- Emphasize that the measured penalty is roughly `1.2-1.5 ms/tile`, so a better design would need fewer host-visible pipeline stages or GPU-resident scheduling.
- Write the persistent-thread future-work section: use GPU-resident scheduling and system memory fences to reduce host launch overhead.

