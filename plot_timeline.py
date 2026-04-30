#!/usr/bin/env python3
"""Plot a tile_comm timeline CSV as a clearer Gantt chart.

Default behavior emphasizes communication:
- aggregated runs are plotted one row per communication group
- CPU-side launch/enqueue bars are hidden by default
- grouped communication is plotted once per group leader row

Usage:
  python3 plot_timeline.py results/tile_timeline_overlap_compute_baseline_rank0_ngpu4_tiles8_buf4_inf1_agg4.csv
  python3 plot_timeline.py input.csv --show-launches
  python3 plot_timeline.py input.csv --output timeline.png
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FILENAME_RE = re.compile(
    r"tile_timeline_(?P<body>.+?)_rank(?P<rank>\d+)_ngpu(?P<ngpu>\d+)_tiles(?P<tiles>\d+)_buf(?P<buf>\d+)_inf(?P<inf>\d+)_agg(?P<agg>\d+)"
)


def finite_interval(start, end):
    return pd.notna(start) and pd.notna(end) and end > start


def add_bar(ax, y, start, end, label=None, color=None, height=0.42, alpha=0.95):
    if not finite_interval(start, end):
        return
    ax.barh(y, end - start, left=start, height=height, label=label, color=color, alpha=alpha)


def parse_meta(csv_path, df):
    meta = {
        "mode": "unknown",
        "compute": None,
        "rank": None,
        "ngpu": None,
        "tiles": int(df["tile_id"].nunique()) if "tile_id" in df.columns else None,
        "buf": None,
        "inf": None,
        "agg": int(df.groupby("group_id").size().max()) if "group_id" in df.columns else 1,
    }
    m = FILENAME_RE.search(csv_path.stem)
    if m:
        gd = {k: v for k, v in m.groupdict().items() if v is not None}
        body = gd.pop("body", "")
        meta.update({k: (int(v) if str(v).isdigit() else v) for k, v in gd.items()})
        if "_compute_" in body:
            mode, compute = body.split("_compute_", 1)
            meta["mode"] = mode
            meta["compute"] = compute
        else:
            meta["mode"] = body
    return meta


def build_rows(df, aggregate_tiles, show_launches):
    """Return rows to plot.

    For aggregate_tiles > 1, build one row per communication group.
    Otherwise, build one row per tile.
    """
    rows = []

    if aggregate_tiles > 1 and "group_id" in df.columns:
        grouped = df.sort_values(["group_id", "tile_id"]).groupby("group_id", sort=True)
        for group_id, gdf in grouped:
            leader = gdf[gdf["is_group_leader"] == 1]
            if leader.empty:
                leader_row = gdf.iloc[0]
            else:
                leader_row = leader.iloc[0]

            first_tile = int(gdf["tile_id"].min())
            last_tile = int(gdf["tile_id"].max())
            plot_row = {
                "row_label": "group %d (tiles %d–%d)" % (group_id, first_tile, last_tile),
                "comm_start_ms": leader_row.get("comm_start_ms", math.nan),
                "comm_end_ms": leader_row.get("comm_end_ms", math.nan),
                "copy_start_ms": leader_row.get("copy_start_ms", math.nan),
                "copy_end_ms": leader_row.get("copy_end_ms", math.nan),
                "group_id": int(group_id),
            }

            if show_launches:
                plot_row["compute_launch_start_ms"] = gdf["compute_enqueue_start_ms"].min()
                plot_row["compute_launch_end_ms"] = gdf["compute_enqueue_end_ms"].max()
                plot_row["proj_launch_start_ms"] = gdf["proj_enqueue_start_ms"].min()
                plot_row["proj_launch_end_ms"] = gdf["proj_enqueue_end_ms"].max()

            rows.append(plot_row)
    else:
        for _, row in df.sort_values("tile_id").iterrows():
            plot_row = {
                "row_label": "tile %d" % int(row["tile_id"]),
                "comm_start_ms": row.get("comm_start_ms", math.nan),
                "comm_end_ms": row.get("comm_end_ms", math.nan),
                "copy_start_ms": row.get("copy_start_ms", math.nan),
                "copy_end_ms": row.get("copy_end_ms", math.nan),
            }
            if show_launches:
                plot_row["compute_launch_start_ms"] = row.get("compute_enqueue_start_ms", math.nan)
                plot_row["compute_launch_end_ms"] = row.get("compute_enqueue_end_ms", math.nan)
                plot_row["proj_launch_start_ms"] = row.get("proj_enqueue_start_ms", math.nan)
                plot_row["proj_launch_end_ms"] = row.get("proj_enqueue_end_ms", math.nan)
            rows.append(plot_row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot tile_comm timeline CSV")
    parser.add_argument("csv", type=Path, help="tile_timeline_*.csv file")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output image path; defaults to <csv stem>.png")
    parser.add_argument("--title", default=None, help="Optional title override")
    parser.add_argument("--show-launches", action="store_true",
                        help="Show CPU-side compute/projection launch windows")
    parser.add_argument("--hide-copy", action="store_true",
                        help="Hide D2D copy bars even if present")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = args.output or args.csv.with_suffix(".png")
    meta = parse_meta(args.csv, df)
    aggregate_tiles = int(meta.get("agg") or 1)

    rows = build_rows(df, aggregate_tiles, args.show_launches)
    if not rows:
        raise RuntimeError("No rows to plot from %s" % args.csv)

    fig_height = max(4.0, 0.70 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(13.5, fig_height))

    labels_seen = set()

    def label_once(name):
        if name in labels_seen:
            return None
        labels_seen.add(name)
        return name

    show_copy = (not args.hide_copy) and ("copy_start_ms" in df.columns)

    for y, row in enumerate(rows):
        if args.show_launches:
            add_bar(ax, y,
                    row.get("compute_launch_start_ms", math.nan),
                    row.get("compute_launch_end_ms", math.nan),
                    label_once("compute launch"),
                    color="C0", height=0.26, alpha=0.85)
            add_bar(ax, y,
                    row.get("proj_launch_start_ms", math.nan),
                    row.get("proj_launch_end_ms", math.nan),
                    label_once("projection launch"),
                    color="C1", height=0.26, alpha=0.85)

        add_bar(ax, y,
                row.get("comm_start_ms", math.nan),
                row.get("comm_end_ms", math.nan),
                label_once("MPI all-reduce"),
                color="C2", height=0.48, alpha=0.95)

        if show_copy:
            add_bar(ax, y,
                    row.get("copy_start_ms", math.nan),
                    row.get("copy_end_ms", math.nan),
                    label_once("D2D copy"),
                    color="C3", height=0.16, alpha=0.95)

    ax.set_xlabel("Time from run start (ms)")
    ax.set_ylabel("Communication group" if aggregate_tiles > 1 else "Tile id")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["row_label"] for row in rows])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    title = args.title
    if title is None:
        mode_name = str(meta.get("mode", "unknown")).replace("_", " ").title()
        comp = meta.get("compute")
        if comp:
            mode_name = "%s (%s compute)" % (mode_name, comp)
        title = "%s timeline (rank %s, %s GPUs)" % (
            mode_name,
            meta.get("rank", "?"),
            meta.get("ngpu", "?"),
        )

    subtitle_parts = []
    if aggregate_tiles > 1:
        subtitle_parts.append("%d groups × %d tiles/group" % (len(rows), aggregate_tiles))
    else:
        subtitle_parts.append("%d tiles" % len(rows))

    if args.show_launches:
        subtitle_parts.append("launch windows shown")
    else:
        subtitle_parts.append("launch windows hidden")

    if show_copy:
        subtitle_parts.append("D2D copy shown")

    fig.suptitle(title, fontsize=14)
    ax.set_title(" · ".join(subtitle_parts), fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200)
    print("Wrote %s" % out)


if __name__ == "__main__":
    main()
