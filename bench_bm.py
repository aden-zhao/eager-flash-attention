#!/usr/bin/env python3
"""B_M sensitivity table: vary B_M at fixed D=4 across sequence lengths."""
import subprocess
import sys

BENCH = "./attn_bench"
D = 4
B_N = 64

S_VALUES  = [1024, 4096, 16384]
BM_VALUES = [8, 16, 32]

for S in S_VALUES:
    for B_M in BM_VALUES:
        print(f"# bm_table  S={S} D={D} B_M={B_M} B_N={B_N}", flush=True)
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
