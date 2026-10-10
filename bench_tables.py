#!/usr/bin/env python3
"""Render the README result tables (markdown) from bench_sweep.py CSVs.

    python3 bench_tables.py results/n1_r4.csv results/n2_r8.csv
    python3 bench_tables.py --headline results/core_n1_r4.csv results/core_n2_r8.csv
"""
import csv
import sys
from collections import OrderedDict

headline = "--headline" in sys.argv
rows = []
for path in [a for a in sys.argv[1:] if a != "--headline"]:
    with open(path) as f:
        for r in csv.DictReader(f):
            r["_src"] = path
            r.setdefault("causal", "0")   # older CSVs predate the column
            rows.append(r)
all_rows = rows
rows = [r for r in all_rows if r["causal"] == "0"]

def med(r): return float(r["median_ms"])
def key(r, *fields): return tuple(r[f] for f in fields)

def find(**kw):
    out = [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]
    return out

def fmt(ms):
    return f"{ms:.3f}" if ms < 10 else f"{ms:.1f}"

# ---- headline: one-shot vs stream, 1 vs 2 nodes, causal or not ------------
if headline:
    print("| setup | S | mask | attention only | one-shot (megatron) | stream (best tile) | speedup |")
    print("|---|---|---|---|---|---|---|")
    for R, setup in [("4", "1 node, 4 GPUs (NVLink)"), ("8", "2 nodes, 8 GPUs (Slingshot)")]:
        for causal in ["0", "1"]:
            for S in ["16384", "65536"]:
                s = [r for r in all_rows if r["R"] == R and r["causal"] == causal and r["S"] == S]
                def best(mode):
                    c = [r for r in s if r["mode"] == mode]
                    return min(c, key=med) if c else None
                a, m, st = best("attn"), best("megatron"), best("stream")
                if not (a and m and st): continue
                mask = "causal" if causal == "1" else "full"
                print(f"| {setup} | {int(S)//1024}k | {mask} | {fmt(med(a))} | {fmt(med(m))} | "
                      f"{fmt(med(st))} (tile {st['tile']}) | **{med(m)/med(st):.2f}x** |")
    print()
    sys.exit(0)

# ---- kernel table -------------------------------------------------------
print("### Attention kernel (attention only, 4 GPUs on one node, H=16, d_h=64, 4 local heads per GPU)\n")
print("| S | original code, one-shot pass incl. proj+AllReduce (May 2026) | v1 kernel, fixed grid | tensor-core kernel | tc speedup vs v1 |")
print("|---|---|---|---|---|")
orig = {4096: "383", 16384: "6262", 65536: "-"}   # measured on the pre-rewrite code (B_M=32, B_N=64), same node type
for S in [1024, 4096, 16384, 65536]:
    v1 = find(S=S, R=4, kernel="v1", mode="attn", H=16)
    tc = find(S=S, R=4, kernel="tc", mode="attn", split=1, H=16, proj="fp32")
    if not v1 or not tc: continue
    a, b = med(v1[0]), med(tc[0])
    print(f"| {S} | {orig.get(S, '-')} | {fmt(a)} | {fmt(b)} | {a/b:.0f}x |")
print()

# ---- modes tables -------------------------------------------------------
def modes_table(R, title):
    sub = [r for r in rows if r["R"] == str(R) and r["H"] == "16" and r["proj"] == "fp32"]
    if not sub: return
    print(f"### {title}\n")
    print("| S | attn only | megatron | sync | overlap (best tile) | stream (best tile) | stream vs megatron |")
    print("|---|---|---|---|---|---|---|")
    for S in ["4096", "16384", "65536"]:
        s = [r for r in sub if r["S"] == S]
        if not s: continue
        def best(mode):
            c = [r for r in s if r["mode"] == mode]
            if not c: return None
            b = min(c, key=med)
            return b
        attn = best("attn"); meg = best("megatron"); sync = best("sync"); ov = best("overlap"); st = best("stream")
        if not (attn and meg and st): continue
        cell = lambda r: "-" if r is None else fmt(med(r))
        tcell = lambda r: "-" if r is None else f"{fmt(med(r))} (tile {r['tile']})"
        ratio = med(meg) / med(st)
        print(f"| {S} | {cell(attn)} | {cell(meg)} | {cell(sync)} | {tcell(ov)} | {tcell(st)} | {ratio:.2f}x |")
    print()

modes_table(4, "Execution modes, 4 GPUs on one node (NVLink)")
modes_table(8, "Execution modes, 8 GPUs on two nodes (Slingshot between nodes)")

# ---- scaling ------------------------------------------------------------
sc = [r for r in rows if r["S"] == "16384" and r["H"] == "16" and r["proj"] == "fp32" and r["_src"].find("n1") >= 0]
if sc:
    print("### Rank scaling, S=16384, one node\n")
    print("| ranks | attn only | megatron | overlap | stream |")
    print("|---|---|---|---|---|")
    for R in ["1", "2", "4"]:
        s = [r for r in sc if r["R"] == R]
        if not s: continue
        def g(mode):
            c = [r for r in s if r["mode"] == mode]
            return "-" if not c else fmt(min(med(r) for r in c))
        print(f"| {R} | {g('attn')} | {g('megatron')} | {g('overlap')} | {g('stream')} |")
    print()

# ---- causal variant (default shape) -----------------------------------
def variant_table(sel, title):
    sub = [r for r in all_rows if sel(r) and r["H"] == "16"]
    if not sub: return
    print(f"### {title}\n")
    print("| ranks | S | attn only | megatron | overlap (best) | stream (best) | stream vs megatron |")
    print("|---|---|---|---|---|---|---|")
    for R in sorted({r["R"] for r in sub}, key=int):
        for S in sorted({r["S"] for r in sub}, key=int):
            s = [r for r in sub if r["R"] == R and r["S"] == S]
            def g(mode):
                c = [r for r in s if r["mode"] == mode]
                return None if not c else min(med(r) for r in c)
            a, m, o, st = g("attn"), g("megatron"), g("overlap"), g("stream")
            if m is None or st is None: continue
            cell = lambda x: "-" if x is None else fmt(x)
            print(f"| {R} | {S} | {cell(a)} | {fmt(m)} | {cell(o)} | {fmt(st)} | {m/st:.2f}x |")
    print()

variant_table(lambda r: r["causal"] == "1" and r["proj"] == "fp32", "Causal attention (H=16, d_h=64)")

# ---- model --------------------------------------------------------------
md = [r for r in rows if r["H"] == "32"]
if md:
    print("### Llama-7B-shaped layer (H=32, d_h=128, d=4096), S=16384\n")
    print("| ranks | projection | attn only | megatron | overlap (best) | stream (best) | stream vs megatron |")
    print("|---|---|---|---|---|---|---|")
    for R in sorted({r["R"] for r in md}, key=int):
        for proj in ["fp32", "tf32"]:
            s = [r for r in md if r["R"] == R and r["proj"] == proj]
            if not s: continue
            def g(mode):
                c = [r for r in s if r["mode"] == mode]
                return None if not c else min(med(r) for r in c)
            a, m, o, st = g("attn"), g("megatron"), g("overlap"), g("stream")
            if m is None or st is None: continue
            print(f"| {R} | {proj} | {fmt(a)} | {fmt(m)} | {fmt(o)} | {fmt(st)} | {m/st:.2f}x |")
    print()
