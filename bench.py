#!/usr/bin/env python3
import subprocess
import sys

BENCH = "./attn_bench"

# B_M sensitivity: find best tile size at D=4
BM_TABLE = [
    (S, BM, 4)
    for S  in [1024, 4096, 16384, 65536]
    for BM in [8, 16, 32]
]

# Strong/weak scaling: fixed B_M=16
SCALING = [
    (S, 16, D)
    for S in [1024, 4096, 16384, 65536]
    for D in [1, 2, 4]
]

def run(S, BM, D, tag):
    print(f"# {tag}  S={S} B_M={BM} D={D}", flush=True)
    subprocess.run([
        "srun", f"-n{D}",
        f"--ntasks-per-node={D}",
        f"--gpus-per-node={D}",
        "--gpu-bind=none",
        BENCH, "--S", str(S), "--B_M", str(BM)
    ], check=True)
    sys.stdout.flush()

print("# === B_M SENSITIVITY TABLE ===", flush=True)
for S, BM, D in BM_TABLE:
    run(S, BM, D, "bm_table")

print("# === SCALING ===", flush=True)
for S, BM, D in SCALING:
    run(S, BM, D, "scaling")
