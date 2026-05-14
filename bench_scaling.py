#!/usr/bin/env python3
"""Scaling study: vary S and D at fixed B_M across device counts."""
import subprocess
import sys

BENCH = "./attn_bench"
B_N = 64
B_M = int(sys.argv[1]) if len(sys.argv) > 1 else 16

S_VALUES = [1024, 4096, 16384]
D_VALUES = [1, 2, 4]

for S in S_VALUES:
    for D in D_VALUES:
        print(f"# scaling  S={S} D={D} B_M={B_M} B_N={B_N}", flush=True)
        subprocess.run([
            "srun", f"-n{D}",
            f"--ntasks-per-node={D}",
            f"--gpus-per-node={D}",
            "--gpu-bind=none",
            BENCH,
            "--S",   str(S),
            "--B_M", str(B_M),
            "--B_N", str(B_N),
        ], check=True)
        sys.stdout.flush()
