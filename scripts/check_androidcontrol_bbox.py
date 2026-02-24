#!/usr/bin/env python3
"""Inspect gt_bbox format in AndroidControl parquet files."""

import argparse
from collections import Counter
from typing import Any

from datasets import load_dataset


def _bbox_type_name(value: Any) -> str:
    if isinstance(value, list):
        return f"list_len_{len(value)}"
    if value is None:
        return "none"
    return type(value).__name__


def _safe_max_abs(value: Any) -> float:
    if not isinstance(value, list) or not value:
        return -1.0
    try:
        return max(abs(float(v)) for v in value)
    except Exception:
        return -1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/workspace/datasets/GUI-R1/androidcontrol_high_test.parquet",
        help="Path to androidcontrol parquet file.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=5000,
        help="Max rows to scan for distribution stats.",
    )
    parser.add_argument(
        "--show_rows",
        type=int,
        default=10,
        help="How many sample rows to print.",
    )
    args = parser.parse_args()

    ds = load_dataset("parquet", data_files=args.data_path, split="train")
    n = min(len(ds), max(0, args.max_rows))

    type_counter = Counter()
    max_abs_counter = Counter()

    for i in range(n):
        row = ds[i]
        bbox = row.get("gt_bbox", None)
        tname = _bbox_type_name(bbox)
        type_counter[tname] += 1

        max_abs = _safe_max_abs(bbox)
        if max_abs >= 0:
            if max_abs <= 1.5:
                max_abs_counter["<=1.5 (normalized?)"] += 1
            elif max_abs <= 1000:
                max_abs_counter["<=1000 (relative?)"] += 1
            else:
                max_abs_counter[">1000 (pixel?)"] += 1

    print(f"Data path: {args.data_path}")
    print(f"Total rows: {len(ds)}")
    print(f"Scanned rows: {n}")
    print("\n[gt_bbox type/length distribution]")
    for k, v in sorted(type_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}: {v}")

    print("\n[gt_bbox value range hint]")
    for k, v in sorted(max_abs_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}: {v}")

    print(f"\n[First {min(args.show_rows, len(ds))} rows sample]")
    for i in range(min(args.show_rows, len(ds))):
        row = ds[i]
        print(
            {
                "idx": i,
                "gt_action": row.get("gt_action"),
                "gt_bbox": row.get("gt_bbox"),
                "image_size": row.get("image_size"),
                "gt_input_text": row.get("gt_input_text"),
            }
        )


if __name__ == "__main__":
    main()
