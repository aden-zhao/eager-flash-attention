#!/usr/bin/env python3
"""Plot a tile_comm timeline CSV as a simple Gantt chart.

Usage:
  python plot_timeline.py results_large_fma200_run1/tile_timeline_overlap_rank0_ngpu4_tiles8_buf4_inf1_agg4.csv
  python plot_timeline.py input.csv --output timeline.png

For aggregated runs, communication rows are plotted only for rows where
is_group_leader == 1, so a grouped all-reduce is not drawn once per tile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def add_bar(ax, y: int, start: float, end: float, label: str) -> None:
    if pd.isna(start) or pd.isna(end) or end <= start:
        return
    ax.barh(y, end - start, left=start, height=0.24, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tile_comm timeline CSV")
    parser.add_argument("csv", type=Path, help="tile_timeline_*.csv file")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output image path; defaults to <csv stem>.png")
    parser.add_argument("--title", default=None, help="Optional plot title")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = args.output or args.csv.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(df) + 1.5)))

    labels_seen: set[str] = set()

    def label_once(name: str) -> str | None:
        if name in labels_seen:
            return None
        labels_seen.add(name)
        return name

    for _, row in df.iterrows():
        tile = int(row["tile_id"])
        add_bar(ax, tile, row["compute_enqueue_start_ms"],
                row["compute_enqueue_end_ms"], label_once("compute enqueue"))
        add_bar(ax, tile, row["proj_enqueue_start_ms"],
                row["proj_enqueue_end_ms"], label_once("projection enqueue"))

        # For aggregated runs, only draw the grouped comm interval once.
        if int(row.get("is_group_leader", 1)) == 1:
            add_bar(ax, tile, row["comm_start_ms"], row["comm_end_ms"],
                    label_once("MPI all-reduce"))

        if "copy_start_ms" in df.columns and int(row.get("is_group_leader", 1)) == 1:
            add_bar(ax, tile, row["copy_start_ms"], row["copy_end_ms"],
                    label_once("D2D copy"))

    ax.set_xlabel("Time from run start (ms)")
    ax.set_ylabel("Tile id / group leader tile id")
    ax.set_yticks(df["tile_id"].astype(int).tolist())
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(args.title or args.csv.name)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
