#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert Open-AutoGLM hm_data trajectories into GUI-R1 training jsonl files.

Output schema is aligned with `verl/utils/dataset.py`:
    instruction, history, task_type, image, gt_bbox, gt_action, gt_input_text
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from PIL import Image


POINT_RE = re.compile(r"<point>\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*</point>")


def parse_point_tag(point_text: str) -> Optional[Tuple[float, float]]:
    if not isinstance(point_text, str):
        return None
    m = POINT_RE.search(point_text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def to_normalized_xy(x: float, y: float, width: int, height: int) -> list[float]:
    x = max(0.0, min(1.0, x / width))
    y = max(0.0, min(1.0, y / height))
    return [x, y]


def to_normalized_bbox(box: list[float], width: int, height: int) -> list[float]:
    if len(box) == 2:
        return to_normalized_xy(box[0], box[1], width, height)
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(1.0, x1 / width))
    y1 = max(0.0, min(1.0, y1 / height))
    x2 = max(0.0, min(1.0, x2 / width))
    y2 = max(0.0, min(1.0, y2 / height))
    return [x1, y1, x2, y2]


def infer_swipe_direction(start_point: Optional[Tuple[float, float]], end_point: Optional[Tuple[float, float]]) -> str:
    if start_point is None or end_point is None:
        return "down"
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "down" if dy >= 0 else "up"


def map_action(
    action_parsed: Dict[str, Any],
    box: Any,
    image_size: Tuple[int, int],
) -> Tuple[str, list[float], str]:
    width, height = image_size
    action = action_parsed.get("action")
    params = action_parsed.get("params", {}) or {}

    # Default values for non-grounding actions.
    default_bbox = [-100.0, -100.0]
    default_text = "no input text"

    if action in ("finished", "call_user", "back_information"):
        content = params.get("content") or params.get("message") or "task completed"
        return "complete", default_bbox, str(content)

    if action == "open_app":
        app_name = params.get("app_name") or params.get("content") or "unknown app"
        return "open_app", default_bbox, str(app_name)

    if action == "click":
        # Prefer detection/grounding box from trace.
        if isinstance(box, dict) and isinstance(box.get("point"), list):
            p = box["point"]
            if len(p) == 4:
                return "click", to_normalized_bbox([float(v) for v in p], width, height), default_text
            if len(p) == 2:
                return "click", to_normalized_bbox([float(v) for v in p], width, height), default_text

        # Fallback to model point string (<point>x y</point>), often in 1000-grid coordinates.
        point_text = params.get("point")
        point = parse_point_tag(point_text)
        if point is not None:
            x, y = point
            if x > width or y > height:
                # Heuristic: many action points use 0-1000 normalized grid.
                return "click", [max(0.0, min(1.0, x / 1000.0)), max(0.0, min(1.0, y / 1000.0))], default_text
            return "click", to_normalized_xy(x, y, width, height), default_text

        return "click", default_bbox, default_text

    if action == "type":
        content = params.get("content") or "unknown"
        return "type", default_bbox, str(content)

    if action == "swipe":
        start_point = parse_point_tag(params.get("start_point"))
        end_point = parse_point_tag(params.get("end_point"))
        direction = infer_swipe_direction(start_point, end_point)
        return "scroll", default_bbox, direction

    if action == "press_back":
        return "press_back", default_bbox, default_text

    if action == "wait":
        wait_t = params.get("t")
        return "wait", default_bbox, str(wait_t if wait_t is not None else "1")

    # Unknown actions are mapped to complete to keep training robust.
    return "complete", default_bbox, default_text


def stringify_history(history: Any) -> str:
    if isinstance(history, list):
        return " | ".join(str(x) for x in history)
    if history is None:
        return ""
    return str(history)


def iter_episode_records(episode_dir: Path) -> Iterable[Dict[str, Any]]:
    traj_path = episode_dir / "trajectory.json"
    trace_path = episode_dir / "trace.jsonl"
    if not (traj_path.exists() and trace_path.exists()):
        return

    try:
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return

    task = traj.get("task", {}) if isinstance(traj, dict) else {}
    if isinstance(task, dict):
        instruction = str(task.get("task_description", "")).strip()
    else:
        instruction = ""

    if not instruction:
        instruction = "请完成当前界面任务。"

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue

            action_parsed = row.get("action_parsed")
            if not isinstance(action_parsed, dict):
                continue

            screenshot_path = row.get("screenshot_path")
            image_name = os.path.basename(screenshot_path) if screenshot_path else f"step_{row.get('step', 1)}.jpg"
            image_path = episode_dir / image_name
            if not image_path.exists():
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                continue

            gt_action, gt_bbox, gt_input_text = map_action(action_parsed, row.get("box"), (width, height))

            yield {
                "instruction": instruction,
                "history": stringify_history(row.get("history")),
                "task_type": "high",
                "image": str(image_path),
                "gt_bbox": gt_bbox,
                "gt_action": gt_action,
                "gt_input_text": gt_input_text,
                "source_task_id": str(traj.get("task_id", "")) if isinstance(traj, dict) else "",
                "source_step": int(row.get("step", 0) or 0),
            }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to hm_data root directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for train/test jsonl")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Episode-level test split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = [d for d in input_dir.iterdir() if d.is_dir()]
    random.Random(args.seed).shuffle(episodes)
    test_n = max(1, int(len(episodes) * args.test_ratio))
    test_set = set(episodes[:test_n])

    train_rows, test_rows = [], []
    action_counter = Counter()
    for ep in episodes:
        rows = list(iter_episode_records(ep))
        if ep in test_set:
            test_rows.extend(rows)
        else:
            train_rows.extend(rows)
        for r in rows:
            action_counter[r["gt_action"]] += 1

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    train_n = write_jsonl(train_path, train_rows)
    test_n_written = write_jsonl(test_path, test_rows)

    print(f"Converted episodes: {len(episodes)}")
    print(f"Train samples: {train_n}")
    print(f"Test samples: {test_n_written}")
    print(f"Action distribution: {dict(action_counter)}")
    print(f"Train file: {train_path}")
    print(f"Test file: {test_path}")


if __name__ == "__main__":
    main()
