# eager-flash-attention

A CUDA/NCCL prototype that overlaps tensor-parallel communication with FlashAttention compute by pipelining completed query tiles through the output projection and AllReduce before the full attention output is materialized.

## The idea

In standard tensor-parallel attention, each rank computes its local attention output, projects it through `W_O`, and then participates in an AllReduce to sum across ranks. These three steps run in sequence, so the AllReduce latency sits fully exposed on the critical path:

```
[----- attention -----][- proj -][--- AllReduce ---]
```

Because FlashAttention computes the output one query tile at a time and each tile produces a disjoint slice of output rows, a tile is *finalized* the moment its K/V sweep completes. There's no need to wait for the rest of attention before projecting and reducing it. Eager-flash-attention exploits this by streaming each tile through projection and AllReduce as soon as it's ready, while later tiles are still computing:

```
[tile 0 attn][tile 1 attn][tile 2 attn]...
             [proj 0     ][proj 1     ]...
                          [AllReduce 0][AllReduce 1]...
```

The mathematical justification is that `W_O` is linear and the cross-rank reduction is a sum, so doing them per tile gives the same result as doing them once at the end — but the communication now overlaps with compute instead of blocking it.

## What's in the repo

- `flash_attn.cu` / `flash_attn.cuh` — exact attention forward kernel with online softmax, on-chip K/V tiling, register-resident accumulators, and warp-shuffle reductions
- `run_drivers.cu` — three execution modes for comparison:
  - **one-shot** — compute all attention, then project, then one AllReduce (Megatron-style baseline)
  - **tiled-sync** — per-tile projection and blocking AllReduce, no overlap (isolates tiling cost)
  - **overlap** — per-tile pipeline with double-buffered NCCL AllReduce on a separate stream
- `attn_defs.cuh` — shared type and constant definitions

## Results

Benchmarked across 1k–64k sequence lengths on up to 4 A100 GPUs at NERSC Perlmutter: overlap hid AllReduce latency relative to tiled-sync across all settings and beat the one-shot baseline at long sequences, with per-tile launch overhead and a `B_M=32` warp-per-row cap as the binding constraints on the speedup window.

## Related work

The pattern of overlapping tensor-parallel collectives with compute is implemented in production by NVIDIA Transformer Engine's `tp_comm_overlap` (userbuffers + persistent kernels), academically by CoCoNet and Flux, and in the FlashAttention repo's own `RowParallelLinear`. This prototype targets one step earlier in the pipeline: the seam between FlashAttention tile completion and the start of projection, exposed as a host-side scheduling boundary rather than inside a fused kernel.
