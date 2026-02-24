#!/usr/bin/env python3
"""Count action distribution in converted training/eval jsonl files."""

import argparse
import json
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/workspace/datasets/hm_data_converted/train.jsonl",
        help="Path to jsonl file (e.g. hm_data_converted/train.jsonl).",
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default="gt_action",
        help="Action field name in each json row.",
    )
    args = parser.parse_args()

    counter = Counter()
    total = 0

    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            action = str(row.get(args.action_key, "")).strip()
            if not action:
                action = "<EMPTY>"
            counter[action] += 1
            total += 1

    print(f"data_path: {args.data_path}")
    print(f"action_key: {args.action_key}")
    print(f"total_rows: {total}")
    print("\n[action distribution]")
    for action, cnt in counter.most_common():
        ratio = (cnt / total) if total > 0 else 0.0
        print(f"{action:20s} {cnt:8d}  {ratio:8.4%}")

    wait_cnt = counter.get("wait", 0)
    wait_ratio = (wait_cnt / total) if total > 0 else 0.0
    print(f"\nwait_count: {wait_cnt}")
    print(f"wait_ratio: {wait_ratio:.4%}")


if __name__ == "__main__":
    main()
