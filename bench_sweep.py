#!/usr/bin/env python3
"""Benchmark sweeps for attn_bench. Appends CSV rows to --out (header written once).

Run inside a Slurm allocation (or pass --jobid for a background `salloc --no-shell` job):

    python3 bench_sweep.py --suite all --nodes 1 --ranks 4 --out results/n1_r4.csv
    python3 bench_sweep.py --suite modes --nodes 2 --ranks 8 --jobid 12345 --out results/n2_r8.csv

Suites:
  core     megatron vs stream, causal and not, S=16k and 64k (the headline result)
  kernel   v1 vs tensor-core kernel, attention only
  modes    megatron / sync / overlap / stream across S and tile size
  scaling  megatron vs stream while varying the number of ranks (1..ranks)
  model    a Llama-7B-shaped layer (H=32, d_h=128) with fp32, TF32 and bf16 projection
  causal   modes with causal masking (tile completion is front-loaded)
  bf16     modes with the bf16 projection at the default shape
  comm     AllReduce-only timings across tile sizes (input for the cost model)
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="all")
ap.add_argument("--nodes", type=int, default=1)
ap.add_argument("--ranks", type=int, default=4)
ap.add_argument("--jobid", default=None)
ap.add_argument("--bench", default="./build/attn_bench")
ap.add_argument("--out", required=True)
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--ncomm", type=int, default=None, help="default: 4 on one node, 8 across nodes")
ap.add_argument("--split", type=int, default=4)
args = ap.parse_args()

NCOMM = args.ncomm if args.ncomm is not None else (8 if args.nodes > 1 else 4)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
wrote_header = os.path.exists(args.out) and os.path.getsize(args.out) > 0
out = open(args.out, "a")


def run(ranks, extra, trials=None, warmup=None):
    nodes = min(args.nodes, ranks)
    cmd = ["srun"]
    if args.jobid:
        cmd += [f"--jobid={args.jobid}"]
    cmd += [f"-N{nodes}", f"-n{ranks}", f"--ntasks-per-node={ranks // nodes}",
            f"--gpus-per-node={ranks // nodes}", "--gpu-bind=none",
            args.bench, "--trials", str(trials or args.trials),
            "--warmup", str(warmup or args.warmup)] + [str(x) for x in extra]
    global wrote_header
    if wrote_header:
        cmd.append("--no-header")
    print("#", " ".join(cmd[cmd.index(args.bench) + 1:]), flush=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        print(f"# FAILED (exit {res.returncode})", flush=True)
        return
    out.write(res.stdout)
    out.flush()
    sys.stdout.write(res.stdout)
    wrote_header = True


def tiles_for(S):
    return sorted({max(256, S // 32), max(256, S // 16), max(256, S // 8)})


suites = args.suite.split(",") if args.suite != "all" else ["kernel", "modes", "scaling", "model", "causal", "bf16"]
R = args.ranks

if "core" in suites:
    for causal in [[], ["--causal"]]:
        for S in [16384, 65536]:
            run(R, ["--S", S, "--kernel", "tc", "--modes", "attn,megatron"] + causal)
            for tile in [S // 16, S // 8]:
                for split in ([1, args.split] if causal else [args.split]):
                    run(R, ["--S", S, "--kernel", "tc", "--tile", tile, "--split", split,
                            "--ncomm", NCOMM, "--modes", "stream"] + causal)

if "kernel" in suites:
    for S in [1024, 4096, 16384, 65536]:
        run(R, ["--S", S, "--kernel", "v1", "--modes", "attn"], trials=3, warmup=1)
        run(R, ["--S", S, "--kernel", "tc", "--modes", "attn"])
        run(R, ["--S", S, "--kernel", "tc", "--split", args.split, "--modes", "attn"])

if "modes" in suites:
    for S in [4096, 16384, 65536]:
        run(R, ["--S", S, "--kernel", "tc", "--modes", "attn,megatron"])
        for tile in tiles_for(S):
            modes = "overlap,stream" + (",sync" if tile == S // 16 else "")
            run(R, ["--S", S, "--kernel", "tc", "--tile", tile, "--split", args.split,
                    "--ncomm", NCOMM, "--modes", modes])

if "scaling" in suites:
    r = 1
    while r <= R:
        S = 16384
        run(r, ["--S", S, "--kernel", "tc", "--modes", "attn,megatron"])
        run(r, ["--S", S, "--kernel", "tc", "--tile", S // 8, "--split", args.split,
                "--ncomm", NCOMM, "--modes", "overlap,stream"])
        r *= 2

if "model" in suites:
    S = 16384
    for proj in ["fp32", "tf32", "bf16"]:
        run(R, ["--S", S, "--H", 32, "--d_h", 128, "--kernel", "tc", "--proj", proj, "--modes", "attn,megatron"])
        for tile in [1024, 2048]:
            run(R, ["--S", S, "--H", 32, "--d_h", 128, "--kernel", "tc", "--proj", proj, "--tile", tile,
                    "--split", args.split, "--ncomm", NCOMM, "--modes", "overlap,stream"])

if "causal" in suites:
    for S in [16384, 65536]:
        run(R, ["--S", S, "--kernel", "tc", "--causal", "--modes", "attn,megatron"])
        for tile in tiles_for(S):
            run(R, ["--S", S, "--kernel", "tc", "--causal", "--tile", tile, "--split", args.split,
                    "--ncomm", NCOMM, "--modes", "overlap,stream"])

if "comm" in suites:
    S = 16384
    for tile in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        run(R, ["--S", S, "--kernel", "tc", "--tile", tile, "--modes", "comm"])

if "bf16" in suites:
    for S in [16384, 65536]:
        run(R, ["--S", S, "--kernel", "tc", "--proj", "bf16", "--modes", "attn,megatron"])
        for tile in tiles_for(S):
            run(R, ["--S", S, "--kernel", "tc", "--proj", "bf16", "--tile", tile, "--split", args.split,
                    "--ncomm", NCOMM, "--modes", "overlap,stream"])

out.close()
print("# done ->", args.out)
