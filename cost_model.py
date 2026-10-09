#!/usr/bin/env python3
"""Latency + bandwidth model for the streamed pipeline, fitted from bench CSVs.

    python3 cost_model.py results/n2_r8.csv [more.csv ...]

Needs rows from the `comm` sweep suite (mode=comm at several --tile values) to
fit the per-AllReduce cost  t_ar(bytes) = alpha + bytes / beta,  plus the usual
attn / megatron / stream rows to compare predictions against measurements.

Model (per rank, one pass over S rows, n = S / tile tiles, B = S * d * 4 bytes):
    T_attn          measured (mode=attn)
    t_proj(S)       = T_megatron - T_attn - t_ar(B)      (one-shot projection)
    T_megatron_pred = T_attn + t_proj(S) + t_ar(B)
    T_stream_pred   = max(T_attn, n * alpha / ncomm + B / beta, t_proj(S))
                      + t_proj(S) * tile / S + t_ar(B / n)   (last tile's tail)
The ncomm term says concurrent communicators overlap latency but share bandwidth.
"""
import csv
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                r.setdefault("causal", "0")
                r.setdefault("gctx", "0")
                rows.append(r)
    return rows


def med(r):
    return float(r["median_ms"])


def fit_line(xs, ys):
    """Least squares y = a + x / beta -> returns (a, beta)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    a = my - slope * mx
    beta = 1.0 / slope if slope else float("inf")
    return a, beta


def main():
    rows = load(sys.argv[1:])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["R"], r["H"], r["d_h"], r["causal"], r["proj"])].append(r)

    for key, g in sorted(groups.items()):
        R, H, d_h, causal, proj = key
        comm = [r for r in g if r["mode"] == "comm"]
        if not comm:
            continue
        d = int(H) * int(d_h)
        xs, ys = [], []
        for r in comm:
            S, tile = int(r["S"]), int(r["tile"])
            n = S // tile
            xs.append(tile * d * 4)          # bytes per AllReduce
            ys.append(med(r) / n)            # ms per AllReduce
        alpha, beta = fit_line(xs, ys)       # ms, bytes/ms
        print(f"## R={R} H={H} d_h={d_h} causal={causal} proj={proj}")
        print(f"AllReduce fit: t(bytes) = {alpha*1e3:.1f} us + bytes / {beta/1e6:.1f} GB/s   "
              f"({len(comm)} sizes, {xs[0]/1e6:.2f}..{xs[-1]/1e6:.1f} MB)")

        def t_ar(bytes_):
            return alpha + bytes_ / beta

        for S in sorted({int(r["S"]) for r in g}):
            attn = [r for r in g if r["mode"] == "attn" and int(r["S"]) == S and r["split"] == "1"]
            meg = [r for r in g if r["mode"] == "megatron" and int(r["S"]) == S]
            if not attn or not meg:
                continue
            T_attn = min(med(r) for r in attn)
            T_meg = min(med(r) for r in meg)
            B = S * d * 4
            t_proj = max(T_meg - T_attn - t_ar(B), 0.0)
            print(f"\nS={S}: attn {T_attn:.3f} ms, one-shot AllReduce {t_ar(B):.3f} ms, "
                  f"projection {t_proj:.3f} ms, megatron measured {T_meg:.3f}")
            print("| tile | ncomm | stream measured | stream predicted | comm-bound? |")
            print("|---|---|---|---|---|")
            for r in sorted([r for r in g if r["mode"] == "stream" and int(r["S"]) == S],
                            key=lambda r: (int(r["tile"]), int(r["ncomm"]))):
                tile, k = int(r["tile"]), int(r["ncomm"])
                n = S // tile
                T_comm = n * alpha / k + B / beta
                body = max(T_attn, T_comm, t_proj)
                tail = t_proj * tile / S + t_ar(B / n)
                pred = body + tail
                bound = "comm" if T_comm >= T_attn else "attn"
                print(f"| {tile} | {k} | {med(r):.3f} | {pred:.3f} | {bound} |")
        print()


if __name__ == "__main__":
    main()
