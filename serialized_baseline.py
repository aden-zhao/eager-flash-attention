import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# helper
def _sync(device: torch.device):
    """Synchronize only when running on CUDA, so CPU doesn't crash."""
    if device.type == "cuda":
        torch.cuda.synchronize()


# perlmutter is 4 gpus per nodes
# 8 ranks across 2 nodes <=> world_size = 8, device_count = 4
def _init_mpi():
    """Initialise MPI, set the CUDA device, and return (comm, rank, world_size)."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    except ImportError:
        comm = None
        rank = 0
        world_size = 1

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return comm, rank, world_size


@dataclass
class ModelConfig:
    """Single-layer transformer configuration."""
    d_model: int = 4096
    n_heads: int = 32
    seq_len: int = 1024
    dtype: torch.dtype = torch.float16

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


# we use deterministic random weights so every rank has the same reference output.
# returns W_q, W_k, W_v
def _make_qkv_weights(cfg: ModelConfig, device: torch.device):
    """Create Q, K, V projection matrices (d_model -> d_model each)."""
    gen = torch.Generator(device="cpu").manual_seed(42)
    def _w():
        return torch.randn(
            cfg.d_model, cfg.d_model, generator=gen, dtype=cfg.dtype
        ).to(device) / math.sqrt(cfg.d_model)
    return _w(), _w(), _w()


def _make_out_weight(cfg: ModelConfig, device: torch.device):
    """Create output projection W_out (d_model -> d_model)."""
    gen = torch.Generator(device="cpu").manual_seed(67)
    return torch.randn(
        cfg.d_model, cfg.d_model, generator=gen, dtype=cfg.dtype
    ).to(device) / math.sqrt(cfg.d_model)


# actual algorithms
def standard_attention(
    Q: torch.Tensor,  # (B, n_local_heads, S, head_dim)
    K: torch.Tensor,
    V: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Standard scaled dot-product attention.

    Returns: (B, n_local_heads, S, head_dim)
    """
    scale = 1.0 / math.sqrt(head_dim)

    # (B, h, S, S)
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # causal mask
    # S = Q.size(-2)
    # mask = torch.triu(torch.ones(S, S, device=Q.device, dtype=torch.bool), diagonal=1)
    # attn_weights.masked_fill_(mask, float('-inf'))

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
    return torch.matmul(attn_weights, V)


def _record_timings(timings: Optional[dict], t0, t1, t2, t3, t4):
    """Store per-stage timings plus a tp_core_total that excludes replicated QKV.

    tp_core_total = attention + out_proj + allreduce
    """
    if timings is None:
        return
    timings["qkv_proj"] = t1 - t0
    timings["attention"] = t2 - t1
    timings["out_proj"] = t3 - t2
    timings["allreduce"] = t4 - t3
    timings["tp_core_total"] = (t2 - t1) + (t3 - t2) + (t4 - t3)
    timings["total"] = t4 - t0


def serialized_attention_layer(
    x: torch.Tensor,         # (B, S, d_model)
    W_q: torch.Tensor,       # (d_model, d_model)
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_out: torch.Tensor,     # (d_model, d_model)
    cfg: ModelConfig,
    comm,
    rank: int,
    world_size: int,
    timings: Optional[dict] = None,
) -> torch.Tensor:
    """Fully serialized: attention -> W_out -> all-reduce (mpi4py path)."""
    device = x.device
    B, S, D = x.shape
    h = cfg.n_heads
    d_h = cfg.head_dim

    assert h % world_size == 0, (
        f"n_heads ({h}) must be divisible by world_size ({world_size})"
    )
    heads_per_rank = h // world_size
    head_start = rank * heads_per_rank
    head_end = head_start + heads_per_rank

    _sync(device)
    t0 = time.perf_counter()

    Q = x @ W_q  # (B, S, D)
    K = x @ W_k
    V = x @ W_v

    # reshape to (B, h, S, d_h) then take this rank's heads.
    Q = Q.view(B, S, h, d_h).transpose(1, 2)[:, head_start:head_end, :, :]
    K = K.view(B, S, h, d_h).transpose(1, 2)[:, head_start:head_end, :, :]
    V = V.view(B, S, h, d_h).transpose(1, 2)[:, head_start:head_end, :, :]

    _sync(device)
    t1 = time.perf_counter()

    # attention
    attn_out = standard_attention(Q, K, V, d_h)  # (B, heads_per_rank, S, d_h)

    _sync(device)
    t2 = time.perf_counter()

    # output projection
    attn_flat = attn_out.transpose(1, 2).contiguous().view(B, S, heads_per_rank * d_h)
    # slice the input dimension of W_out for this rank's heads.
    W_out_local = W_out[head_start * d_h : head_end * d_h, :]  # (local_dim, d_model)
    partial = attn_flat @ W_out_local  # (B, S, d_model)

    _sync(device)
    t3 = time.perf_counter()

    # all-reduce
    if world_size > 1 and comm is not None:
        from mpi4py import MPI
        # move to CPU for MPI, then back.
        partial_cpu = partial.float().cpu().numpy()
        result_cpu = partial_cpu.copy()
        comm.Allreduce(partial_cpu, result_cpu, op=MPI.SUM)
        output = torch.from_numpy(result_cpu).to(device=device, dtype=cfg.dtype)
    else:
        output = partial

    _sync(device)
    t4 = time.perf_counter()

    _record_timings(timings, t0, t1, t2, t3, t4)
    return output


# NCCL variant (for benchmarking on Perlmutter)
def serialized_attention_layer_cuda_aware(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_out: torch.Tensor,
    cfg: ModelConfig,
    comm,
    rank: int,
    world_size: int,
    timings: Optional[dict] = None,
) -> torch.Tensor:
    """Same as serialized_attention_layer but uses NCCL via torch.distributed
    for the all-reduce, avoiding the CPU round-trip.

    Requires torch.distributed to be initialised before calling.
    """
    device = x.device
    B, S, D = x.shape
    h = cfg.n_heads
    d_h = cfg.head_dim

    assert h % world_size == 0, (
        f"n_heads ({h}) must be divisible by world_size ({world_size})"
    )
    heads_per_rank = h // world_size
    head_start = rank * heads_per_rank
    head_end = head_start + heads_per_rank

    _sync(device)
    t0 = time.perf_counter()

    Q = (x @ W_q).view(B, S, h, d_h).transpose(1, 2)[:, head_start:head_end, :, :]
    K = (x @ W_k).view(B, S, h, d_h).transpose(1, 2)[:, head_start:head_end, :, :]
    V = (x @ W_v).view(B, S, h, d_h).transpose(1, 2)[:, head_start:head_end, :, :]

    _sync(device)
    t1 = time.perf_counter()

    attn_out = standard_attention(Q, K, V, d_h)

    _sync(device)
    t2 = time.perf_counter()

    attn_flat = attn_out.transpose(1, 2).contiguous().view(B, S, heads_per_rank * d_h)
    W_out_local = W_out[head_start * d_h : head_end * d_h, :]
    partial = attn_flat @ W_out_local

    _sync(device)
    t3 = time.perf_counter()

    # NCCL all-reduce; stays on GPU, in-place
    if world_size > 1:
        torch.distributed.all_reduce(partial, op=torch.distributed.ReduceOp.SUM)

    _sync(device)
    t4 = time.perf_counter()

    _record_timings(timings, t0, t1, t2, t3, t4)
    return partial


# correctness verification
def _compute_reference(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_out: torch.Tensor,
    cfg: ModelConfig,
) -> torch.Tensor:
    """Compute the full attention layer in float32 on a single GPU.

    Returns: (B, S, d_model) in float32.
    """
    device = x.device
    B, S, D = x.shape
    h, d_h = cfg.n_heads, cfg.head_dim

    # upcast everything to fp32.
    x_f = x.float()
    Wq_f, Wk_f, Wv_f, Wo_f = W_q.float(), W_k.float(), W_v.float(), W_out.float()

    Q = (x_f @ Wq_f).view(B, S, h, d_h).transpose(1, 2)
    K = (x_f @ Wk_f).view(B, S, h, d_h).transpose(1, 2)
    V = (x_f @ Wv_f).view(B, S, h, d_h).transpose(1, 2)

    scale = 1.0 / math.sqrt(d_h)
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_weights, V)

    attn_flat = attn_out.transpose(1, 2).contiguous().view(B, S, D)
    return attn_flat @ Wo_f  # (B, S, D) fp32


def verify_correctness(
    output: torch.Tensor,
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_out: torch.Tensor,
    cfg: ModelConfig,
    rank: int,
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    """Compare distributed output against the fp32 reference. Only runs on rank 0."""
    if rank != 0:
        return

    reference = _compute_reference(x, W_q, W_k, W_v, W_out, cfg)

    err = output.float() - reference
    max_err = err.abs().max().item()
    mean_err = err.abs().mean().item()
    rel_err = err.abs() / reference.abs().clamp_min(1e-6)
    max_rel_err = rel_err.max().item()
    mean_rel_err = rel_err.mean().item()

    passed = torch.allclose(output.float(), reference, atol=atol, rtol=rtol)

    print(f"[Correctness] vs fp32 reference")
    print(f"  abs  — max={max_err:.6f}  mean={mean_err:.6f}")
    print(f"  rel  — max={max_rel_err:.6f}  mean={mean_rel_err:.6f}")
    print(f"  {'PASS' if passed else 'FAIL'} (atol={atol}, rtol={rtol})")

    if not passed:
        print("  WARNING: output does not match fp32 reference")


# profiling helpers
def print_timings(timings: dict, rank: int, world_size: int, cfg: ModelConfig):
    """Pretty-print timing breakdown."""
    total = timings["total"]
    tp_core = timings["tp_core_total"]
    print(f"\n{'='*60}")
    print(f"  Rank {rank}/{world_size}  |  seq_len={cfg.seq_len}  "
          f"d_model={cfg.d_model}  n_heads={cfg.n_heads}")
    print(f"{'='*60}")
    print(f"  QKV projection*: {timings['qkv_proj']*1e3:8.2f} ms  "
          f"({timings['qkv_proj']/total*100:5.1f}%)")
    print(f"  Attention       : {timings['attention']*1e3:8.2f} ms  "
          f"({timings['attention']/total*100:5.1f}%)")
    print(f"  Output proj     : {timings['out_proj']*1e3:8.2f} ms  "
          f"({timings['out_proj']/total*100:5.1f}%)")
    print(f"  All-reduce      : {timings['allreduce']*1e3:8.2f} ms  "
          f"({timings['allreduce']/total*100:5.1f}%)")
    print(f"  ──────────────────────────────────────")
    print(f"  TP core total   : {tp_core*1e3:8.2f} ms  "
          f"(attn + proj + AR)")
    print(f"  TOTAL           : {total*1e3:8.2f} ms")
    print(f"{'='*60}")
    print(f"  * QKV is replicated (not tensor-parallel); use TP core")
    print(f"    total for comparisons against tiled/overlapped variants.\n")


def benchmark(fn, warmup: int = 5, iters: int = 20):
    """Run *fn* with warmup, then time *iters* iterations.

    Returns a list of per-iteration timing dicts.
    """
    for _ in range(warmup):
        fn()

    all_timings = []
    for _ in range(iters):
        t = {}
        fn(timings=t)
        all_timings.append(t)
    return all_timings


def summarise_timings(all_timings: list[dict], rank: int, world_size: int,
                      cfg: ModelConfig):
    """Aggregate and print benchmark statistics."""
    import statistics
    keys = ["qkv_proj", "attention", "out_proj", "allreduce",
            "tp_core_total", "total"]
    print(f"\n{'='*65}")
    print(f"  BENCHMARK  Rank {rank}/{world_size}  |  {len(all_timings)} iters  "
          f"|  seq_len={cfg.seq_len}")
    print(f"{'='*65}")
    for k in keys:
        vals = [t[k] * 1e3 for t in all_timings]  # ms
        med = statistics.median(vals)
        mn = min(vals)
        mx = max(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        label = k.replace("_", " ").title().ljust(16)
        print(f"  {label}: median={med:7.2f} ms  "
              f"min={mn:7.2f}  max={mx:7.2f}  std={std:5.2f}")
    print(f"{'='*65}\n")


# main
def main():
    parser = argparse.ArgumentParser(
        description="Serialized baseline: attention -> W_out -> all-reduce"
    )
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=4096)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Run correctness check against fp32 reference")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--use_nccl", action="store_true", default=False,
                        help="Use NCCL (torch.distributed) instead of mpi4py")
    args = parser.parse_args()

    comm, rank, world_size = _init_mpi()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        dtype=dtype,
    )

    if rank == 0:
        print(f"Config: d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
              f"seq_len={cfg.seq_len}, batch={args.batch_size}, "
              f"world_size={world_size}, device={device}, dtype={cfg.dtype}")

    # deterministic input
    torch.manual_seed(0)
    x = torch.randn(args.batch_size, cfg.seq_len, cfg.d_model,
                     dtype=cfg.dtype, device=device)

    W_q, W_k, W_v = _make_qkv_weights(cfg, device)
    W_out = _make_out_weight(cfg, device)

    if args.use_nccl and world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world_size
        )

    # choose the layer function
    if args.use_nccl and world_size > 1:
        layer_fn = serialized_attention_layer_cuda_aware
    else:
        layer_fn = serialized_attention_layer

    def run(timings=None):
        return layer_fn(
            x, W_q, W_k, W_v, W_out, cfg,
            comm, rank, world_size,
            timings=timings,
        )

    # single run with timing
    timings: dict = {}
    output = layer_fn(x, W_q, W_k, W_v, W_out, cfg,
                      comm, rank, world_size, timings=timings)
    print_timings(timings, rank, world_size, cfg)

    # correctness
    if args.verify:
        verify_correctness(output, x, W_q, W_k, W_v, W_out, cfg, rank)

    # benchmark
    all_t = benchmark(run, warmup=args.warmup, iters=args.iters)
    summarise_timings(all_t, rank, world_size, cfg)

    # cleanup
    if args.use_nccl and world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
