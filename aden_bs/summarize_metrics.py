#!/usr/bin/env python3
import argparse
import csv
import statistics
import sys
from collections import defaultdict


GROUP_FIELDS = ["mode", "ranks", "S", "d_h", "H_local", "B_M", "B_N"]
STATIC_FIELDS = ["tile_count", "d_model", "message_bytes", "total_reduced_bytes"]
METRIC_FIELDS = ["wall_ms", "cuda_event_ms", "exposed_ms"]


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(values):
    if not values:
        return {
            "mean": "",
            "min": "",
            "max": "",
            "stddev": "",
        }

    return {
        "mean": f"{(sum(values) / len(values)):.6f}",
        "min": f"{min(values):.6f}",
        "max": f"{max(values):.6f}",
        "stddev": f"{statistics.stdev(values):.6f}" if len(values) > 1 else "0.000000",
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate repeated benchmark rows.")
    parser.add_argument("metrics_csv", help="CSV produced by parse_metrics.py")
    args = parser.parse_args()

    groups = defaultdict(list)
    with open(args.metrics_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if any(row.get(field, "") == "" for field in GROUP_FIELDS):
                continue
            key = tuple(row[field] for field in GROUP_FIELDS)
            groups[key].append(row)

    fieldnames = GROUP_FIELDS + STATIC_FIELDS + ["n"]
    for metric in METRIC_FIELDS:
        fieldnames.extend([
            f"{metric}_mean",
            f"{metric}_min",
            f"{metric}_max",
            f"{metric}_stddev",
        ])

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    for key in sorted(groups.keys(), key=lambda item: tuple(int(x) for x in item)):
        rows = groups[key]
        out = {field: value for field, value in zip(GROUP_FIELDS, key)}
        for field in STATIC_FIELDS:
            out[field] = rows[0].get(field, "")
        out["n"] = str(len(rows))

        for metric in METRIC_FIELDS:
            values = [
                parsed
                for parsed in (parse_float(row.get(metric, "")) for row in rows)
                if parsed is not None
            ]
            stats = summarize(values)
            out[f"{metric}_mean"] = stats["mean"]
            out[f"{metric}_min"] = stats["min"]
            out[f"{metric}_max"] = stats["max"]
            out[f"{metric}_stddev"] = stats["stddev"]

        writer.writerow(out)


if __name__ == "__main__":
    main()
