#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path


RUN_RE = re.compile(r"^RUN\s+(?P<fields>.+)$")
METRIC_RE = re.compile(r"^METRIC\s+(?P<fields>.+)$")
VALIDATION_RE = re.compile(r"^VALIDATION\s+(?P<fields>.+)$")
FIELD_RE = re.compile(r"(\w+)=([^\s]+)")
MODE_RE = re.compile(r"Execution Mode:\s+(\d+)")
WALL_RE = re.compile(r"Wall Clock Execution Time \(MPI_Wtime\):\s+([0-9.eE+-]+)\s+ms")
CUDA_RE = re.compile(r"CUDA Event Window Duration:\s+([0-9.eE+-]+)\s+ms")


def parse_run_fields(line):
    match = RUN_RE.match(line)
    if not match:
        return None
    return dict(FIELD_RE.findall(match.group("fields")))


def parse_prefixed_fields(line, regex):
    match = regex.match(line)
    if not match:
        return None
    return dict(FIELD_RE.findall(match.group("fields")))


def as_int(row, key):
    try:
        return int(row.get(key, ""))
    except ValueError:
        return None


def add_derived_columns(row):
    S = as_int(row, "S")
    ranks = as_int(row, "ranks")
    d_h = as_int(row, "d_h")
    H_local = as_int(row, "H_local")
    B_M = as_int(row, "B_M")
    d_model = as_int(row, "d_model")
    message_bytes = as_int(row, "message_bytes")

    if d_model is None and None not in (d_h, H_local, ranks):
        d_model = d_h * H_local * ranks
        row["d_model"] = str(d_model)

    if S is not None and B_M not in (None, 0) and row.get("tile_count", "") == "":
        row["tile_count"] = str(S // B_M)

    if message_bytes is None and None not in (B_M, d_model):
        message_bytes = B_M * d_model * 4
        row["message_bytes"] = str(message_bytes)

    if message_bytes is not None:
        tile_count = as_int(row, "tile_count")
        if tile_count is not None and row.get("total_reduced_bytes", "") == "":
            row["total_reduced_bytes"] = str(message_bytes * tile_count)

    if row.get("exposed_ms", "") == "":
        try:
            wall_ms = float(row["wall_ms"])
            cuda_ms = float(row["cuda_event_ms"])
            row["exposed_ms"] = f"{max(wall_ms - cuda_ms, 0.0):.6f}"
        except (KeyError, ValueError):
            row["exposed_ms"] = ""

    return row


def parse_logs(paths):
    rows = []
    current = None

    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                run_fields = parse_run_fields(line)
                if run_fields is not None:
                    if current is not None and current.get("wall_ms") and current.get("cuda_event_ms"):
                        rows.append(add_derived_columns(current))
                    current = {
                        "source_log": str(path),
                        "mode": run_fields.get("mode", ""),
                        "ranks": run_fields.get("ranks", ""),
                        "S": run_fields.get("S", ""),
                        "d_h": run_fields.get("d_h", ""),
                        "H_local": run_fields.get("H_local", ""),
                        "B_M": run_fields.get("B_M", ""),
                        "B_N": run_fields.get("B_N", ""),
                        "repeat": run_fields.get("repeat", ""),
                        "validate": run_fields.get("validate", ""),
                        "tile_count": "",
                        "d_model": "",
                        "message_bytes": "",
                        "total_reduced_bytes": "",
                        "wall_ms": "",
                        "cuda_event_ms": "",
                        "exposed_ms": "",
                        "validation_status": "",
                        "max_abs_err": "",
                        "max_rel_err": "",
                    }
                    continue

                metric_fields = parse_prefixed_fields(line, METRIC_RE)
                if metric_fields is not None:
                    row = {"source_log": str(path)}
                    if current is not None:
                        row.update(current)
                    row.update(metric_fields)
                    rows.append(add_derived_columns(row))
                    current = None
                    continue

                if current is None:
                    continue

                validation_fields = parse_prefixed_fields(line, VALIDATION_RE)
                if validation_fields is not None:
                    current["validation_status"] = validation_fields.get("status", "")
                    current["max_abs_err"] = validation_fields.get("max_abs_err", "")
                    current["max_rel_err"] = validation_fields.get("max_rel_err", "")
                    continue

                mode_match = MODE_RE.search(line)
                if mode_match:
                    current["mode"] = mode_match.group(1)
                    continue

                wall_match = WALL_RE.search(line)
                if wall_match:
                    current["wall_ms"] = wall_match.group(1)
                    continue

                cuda_match = CUDA_RE.search(line)
                if cuda_match:
                    current["cuda_event_ms"] = cuda_match.group(1)

            if current is not None and current.get("wall_ms") and current.get("cuda_event_ms"):
                rows.append(add_derived_columns(current))
                current = None

    return rows


def main():
    parser = argparse.ArgumentParser(description="Parse benchmark logs into metrics CSV.")
    parser.add_argument("logs", nargs="+", help="Log files produced by run_eval_perlmutter.sh")
    args = parser.parse_args()

    fieldnames = [
        "source_log",
        "mode",
        "ranks",
        "S",
        "d_h",
        "H_local",
        "B_M",
        "B_N",
        "repeat",
        "validate",
        "tile_count",
        "d_model",
        "message_bytes",
        "total_reduced_bytes",
        "wall_ms",
        "cuda_event_ms",
        "exposed_ms",
        "validation_status",
        "max_abs_err",
        "max_rel_err",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(parse_logs(args.logs))


if __name__ == "__main__":
    main()
