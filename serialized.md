# Serialized Baseline — Attention → W_out → All-Reduce

## Quick Start (Perlmutter)

First, get an interactive allocation:
```bash
salloc --nodes=1 --ntasks=4 --gpus=4 --cpus-per-task=4 --account=m4341 --qos=interactive --constraint=gpu --time=30:00
module load pytorch
```

### Single GPU or CPU debugging (no MPI needed)
```bash
python serialized_baseline.py --seq_len 512
```

### Multi-GPU with mpi4py (CPU all-reduce, for debugging)
```bash
srun -n 4 --gpus=4 --gpu-bind=none python serialized_baseline.py --seq_len 1024
```

### Multi-GPU with NCCL (GPU all-reduce, for benchmarking)
```bash
srun -n 4 --gpus=4 --gpu-bind=none python serialized_baseline.py --seq_len 1024 --use_nccl
```

> **Note:** Always use `--use_nccl` for benchmark numbers. The mpi4py path
> round-trips through CPU, so its all-reduce time is dominated by PCIe
> transfer rather than actual collective cost.

> **GPU binding:** On Perlmutter, use `srun -n <N> --gpus=<N> --gpu-bind=none`
> rather than `--gpus-per-task=1`. The latter sets `CUDA_VISIBLE_DEVICES=0` for
> all ranks, causing NCCL to fail with `invalid device ordinal`.

> **Port collisions:** NCCL defaults to `MASTER_PORT=29500`. If multiple jobs
> share a node, set a different port: `MASTER_PORT=29501 srun -n 4 ...`

## What It Does

Four stages run **fully sequentially** — no overlap of any kind:

1. **QKV projection\*** — full `(B, S, D) @ (D, D)` matmuls, then slice heads for this rank
2. **Standard attention** — naive scaled dot-product (no FlashAttention), serves as unambiguous reference
3. **Output projection** — each rank multiplies by its slice of `W_out`, producing a partial `(B, S, D)` result
4. **All-reduce** — `MPI_Allreduce` (or NCCL) sums partials across ranks

\* QKV is replicated across ranks (not tensor-parallel). The timing output
reports a separate **TP core total** (attention + output proj + all-reduce)
which is the apples-to-apples metric for comparing against tiled/overlapped
variants.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--seq_len` | 1024 | Sequence length |
| `--d_model` | 4096 | Model dimension |
| `--n_heads` | 32 | Total attention heads (split across ranks) |
| `--batch_size` | 1 | Batch size |
| `--warmup` | 5 | Warmup iterations before timing |
| `--iters` | 20 | Timed benchmark iterations |
| `--verify` / `--no-verify` | verify | Check output against fp32 reference |
| `--use_nccl` | off | Use NCCL via torch.distributed instead of mpi4py |

## Output

Per-rank timing breakdown (QKV proj, attention, output proj, all-reduce, TP
core total) plus a benchmark summary with median/min/max/std across iterations.

Correctness output reports both absolute and relative error against an fp32
reference, using `torch.allclose` with configurable `atol` and `rtol`.

## Dependencies

- PyTorch (works on both CUDA and CPU; dtype auto-selects fp16 on GPU, fp32 on CPU)
- `mpi4py` (for multi-GPU mpi4py path)
- `torch.distributed` + NCCL (for `--use_nccl` path)

On Perlmutter:
```bash
module load pytorch
# mpi4py should already be available
```

## Using as Correctness Oracle

Both the tiled and overlapped variants should be validated against the fp32
reference via `verify_correctness()` or `_compute_reference()`
directly. The reference runs the full attention layer in float32, making
it insensitive to accumulation order — so variants that use online softmax
with different reduction patterns can be compared without false positives.

```python
from serialized_baseline import (
    ModelConfig, serialized_attention_layer,
    _compute_reference,
    _make_qkv_weights, _make_out_weight,
)
```