# Draws the README figures as SVG. Standard library only.
#   python3 docs/make_figures.py results/core_n1_r4.csv results/core_n2_r8.csv
# docs/figures/schedule.svg  : schematic one-shot vs stream timeline
# docs/figures/headline.svg  : headline results, times relative to the one-shot baseline
# (docs/figures/tile_sweep.png is cropped from the course poster.)
import csv, os, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
ATTN, ATTN_E = "#f2dcdc", "#9c4a4a"     # poster palette: attention pink, projection blue, comm green
PROJ, PROJ_E = "#d8e1ef", "#4a6a9c"
COMM, COMM_E = "#d6eadb", "#3f8a55"
INK, MUTED, RED = "#1f2328", "#57606a", "#b02a2a"
FONT = "font-family='Helvetica, Arial, sans-serif'"

def text(x, y, s, size=13, anchor="middle", color=INK, weight="normal"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' text-anchor='{anchor}' "
            f"fill='{color}' font-weight='{weight}'>{s}</text>")

def rect(x, y, w, h, fill, edge, dash=False):
    d = " stroke-dasharray='4 3'" if dash else ""
    return (f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='4' "
            f"fill='{fill}' stroke='{edge}' stroke-width='1.5'{d}/>")

def svg(w, h, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<rect width='{w}' height='{h}' fill='white'/>" + "".join(body) + "</svg>\n")

# ---- schedule: one-shot vs stream (illustrative units, not measured) --------------------
def schedule():
    T, ATT, PROJ_ALL, AR_ALL = 8, 8.0, 2.0, 6.0    # tiles; one-shot attention/projection/AllReduce
    p, ar, ncomm = PROJ_ALL / T, 1.6, 2             # per-tile projection; per-tile AllReduce; communicators
    x0, scale, rh = 150, 46, 30
    b = [text(x0, 26, "One-shot (Megatron): attention, then projection, then one AllReduce",
              14, "start", weight="bold")]
    y = 44
    b += [text(x0 - 12, y + 20, "compute", 12, "end", MUTED),
          rect(x0, y, ATT * scale, rh, ATTN, ATTN_E), text(x0 + ATT * scale / 2, y + 20, "attention"),
          rect(x0 + ATT * scale, y, PROJ_ALL * scale, rh, PROJ, PROJ_E),
          text(x0 + (ATT + PROJ_ALL / 2) * scale, y + 20, "proj")]
    y += rh + 8
    t_one = ATT + PROJ_ALL + AR_ALL
    b += [text(x0 - 12, y + 20, "comm", 12, "end", MUTED),
          rect(x0 + (ATT + PROJ_ALL) * scale, y, AR_ALL * scale, rh, COMM, COMM_E),
          text(x0 + (ATT + PROJ_ALL + AR_ALL / 2) * scale, y + 20, "AllReduce (fully exposed)")]

    y += rh + 52
    b += [text(x0, y - 14, "Stream (this project): one attention launch, tiles signal when done",
               14, "start", weight="bold")]
    b += [text(x0 - 12, y + 20, "attention", 12, "end", MUTED),
          rect(x0, y, ATT * scale, rh, ATTN, ATTN_E)]
    for t in range(T):
        cx = x0 + (t + 0.5) * scale
        b.append(text(cx, y + 20, f"T{t}", 12))
        if t: b.append(f"<line x1='{x0 + t * scale}' y1='{y}' x2='{x0 + t * scale}' y2='{y + rh}' "
                       f"stroke='{ATTN_E}' stroke-dasharray='3 3'/>")
    y_proj = y + rh + 8
    b.append(text(x0 - 12, y_proj + 20, "projection", 12, "end", MUTED))
    comm_free, end = [0.0] * ncomm, 0.0
    for t in range(T):
        ps = t + 1.0                                   # tile t's counter reaches its target
        b += [rect(x0 + ps * scale, y_proj, p * scale, rh, PROJ, PROJ_E),
              f"<line x1='{x0 + ps * scale}' y1='{y + rh}' x2='{x0 + ps * scale}' y2='{y_proj}' "
              f"stroke='{MUTED}' stroke-width='1'/>"]
        c = t % ncomm
        s = max(ps + p, comm_free[c])
        comm_free[c] = end = s + ar
        yc = y_proj + rh + 8 + c * (rh + 6)
        b += [rect(x0 + s * scale, yc, ar * scale, rh, COMM, COMM_E),
              text(x0 + (s + ar / 2) * scale, yc + 20, f"AR {t}", 12)]
    for c in range(ncomm):
        b.append(text(x0 - 12, y_proj + rh + 8 + c * (rh + 6) + 20, f"comm {'AB'[c]}", 12, "end", MUTED))
    y_end = y_proj + rh + 8 + ncomm * (rh + 6) + 18
    b += [f"<line x1='{x0 + end * scale}' y1='44' x2='{x0 + end * scale}' y2='{y_end}' stroke='{RED}' stroke-dasharray='5 4'/>",
          f"<line x1='{x0 + t_one * scale}' y1='44' x2='{x0 + t_one * scale}' y2='{y_end}' stroke='{MUTED}' stroke-dasharray='5 4'/>",
          text(x0 + end * scale, y_end + 16, "stream done", 12, color=RED, weight="bold"),
          text(x0 + t_one * scale, y_end + 16, "one-shot done", 12, color=MUTED),
          text(x0, y_end + 40, "Schematic, not to scale. Tile t's projection waits on its counter "
               "(cuStreamWaitValue32);", 12, "start", MUTED),
          text(x0, y_end + 56, "tile AllReduces alternate between NCCL communicators so they can overlap.",
               12, "start", MUTED)]
    return svg(int(x0 + t_one * scale + 60), int(y_end + 72), b)

# ---- headline results ------------------------------------------------------------------
def headline(paths):
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    med = lambda r: float(r["median_ms"])
    groups = []
    for R, label in [("4", "1 node, 4 GPUs (NVLink)"), ("8", "2 nodes, 8 GPUs (Slingshot)")]:
        cfgs = []
        for causal in ["0", "1"]:
            for S in ["16384", "65536"]:
                s = [r for r in rows if r["R"] == R and r["causal"] == causal and r["S"] == S]
                best = lambda m: min((r for r in s if r["mode"] == m), key=med, default=None)
                a, m, st = best("attn"), best("megatron"), best("stream")
                if a and m and st:
                    cfgs.append((f"{int(S) // 1024}k {'causal' if causal == '1' else 'full'}",
                                 med(a) / med(m), med(st) / med(m), med(m) / med(st)))
        groups.append((label, cfgs))

    x0, y0, H, bw, gap = 70, 70, 260, 26, 34      # plot origin (top), height, bar width, gap between configs
    b = [text(x0, 26, "Time relative to one-shot (lower is better)", 15, "start", weight="bold"),
         text(x0, 46, "attention + projection + AllReduce, A100-SXM4, H=16, d_h=64, best tile size",
              12, "start", MUTED)]
    ybar = lambda v: y0 + H * (1 - v / 1.1)
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        b += [f"<line x1='{x0}' y1='{ybar(v):.1f}' x2='{x0 + 40 + 8 * (3 * bw + gap) + 60}' y2='{ybar(v):.1f}' "
              f"stroke='{'#8c959f' if v == 1.0 else '#e6e8eb'}'{' stroke-dasharray=\"5 4\"' if v == 1.0 else ''}/>",
              text(x0 - 8, ybar(v) + 4, f"{v:.2f}", 11, "end", MUTED)]
    x = x0 + 20
    for gi, (label, cfgs) in enumerate(groups):
        gx = x
        for name, attn, stream, speedup in cfgs:
            for k, (v, fill, edge) in enumerate([(1.0, PROJ, PROJ_E), (stream, COMM, COMM_E), (attn, ATTN, ATTN_E)]):
                b.append(rect(x + k * bw, ybar(v), bw - 3, ybar(0) - ybar(v), fill, edge))
            b += [text(x + 1.5 * bw, ybar(max(1.0, stream)) - 8, f"{speedup:.2f}x", 12, color=RED, weight="bold"),
                  text(x + 1.5 * bw, ybar(0) + 18, name, 12)]
            x += 3 * bw + gap
        b.append(text((gx + x - gap) / 2, ybar(0) + 40, label, 13, weight="bold"))
        x += 40 * (gi == 0)
    ly = ybar(0) + 66
    for i, (name, fill, edge) in enumerate([("one-shot (baseline)", PROJ, PROJ_E),
                                            ("stream (this project)", COMM, COMM_E),
                                            ("attention alone (floor)", ATTN, ATTN_E)]):
        lx = x0 + 20 + i * 210
        b += [rect(lx, ly, 16, 16, fill, edge), text(lx + 24, ly + 13, name, 12, "start")]
    return svg(int(x + 30), int(ly + 34), b)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "schedule.svg"), "w").write(schedule())
    if len(sys.argv) > 1:
        open(os.path.join(OUT, "headline.svg"), "w").write(headline(sys.argv[1:]))
