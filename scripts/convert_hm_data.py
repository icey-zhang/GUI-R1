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

Output schema is aligned with `verl/utils/dataset.py` and keeps hm_data action style.
All grounding coordinates are normalized to a 0-1000 relative coordinate system.
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


POINT_TAG_RE = re.compile(r"<point>\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*</point>")
POINT_CSV_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _quote_for_action(value: str) -> str:
    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("'", "\\'")
    )
    return f"'{escaped}'"


def _sanitize_action_call(action_call: str) -> str:
    if not isinstance(action_call, str):
        return ""
    return (
        action_call.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _normalize_velocity(value: Any, default: str = "600") -> str:
    try:
        v = float(value)
    except Exception:
        return default
    if v <= 0:
        return default
    if abs(v - int(v)) < 1e-6:
        return str(int(v))
    return str(v)


def parse_point_value(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None

    if isinstance(value, str):
        s = value.strip()
        m = POINT_TAG_RE.search(value)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = POINT_CSV_RE.match(s)
        if m:
            return float(m.group(1)), float(m.group(2))
        nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", s)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    return None


def to_1000_point(point: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    x, y = point
    if x <= 1.5 and y <= 1.5:
        # [0,1] normalized -> [0,1000]
        x, y = x * 1000.0, y * 1000.0
    elif x > 1000 or y > 1000:
        # Pixel coordinates -> [0,1000].
        if width > 0 and height > 0:
            x, y = x / float(width) * 1000.0, y / float(height) * 1000.0
    # Otherwise treat as already in [0,1000].
    x = max(0.0, min(1000.0, float(x)))
    y = max(0.0, min(1000.0, float(y)))
    return x, y


def parse_point_to_1000(value: Any, width: int, height: int) -> Optional[Tuple[float, float]]:
    point = parse_point_value(value)
    if point is None:
        return None
    return to_1000_point(point, width, height)


def infer_swipe_direction(start_point: Optional[Tuple[float, float]], end_point: Optional[Tuple[float, float]]) -> str:
    if start_point is None or end_point is None:
        return "down"
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "down" if dy >= 0 else "up"


def point_to_string(point: Optional[Tuple[float, float]]) -> str:
    if point is None:
        return "-100,-100"
    return f"{int(round(point[0]))},{int(round(point[1]))}"


def _bbox_center(box: list[float]) -> Tuple[float, float]:
    if len(box) == 2:
        return box[0], box[1]
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _to_1000_from_pixel(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    if width <= 0 or height <= 0:
        return max(0.0, min(1000.0, x)), max(0.0, min(1000.0, y))
    return max(0.0, min(1000.0, x / width * 1000.0)), max(0.0, min(1000.0, y / height * 1000.0))


def parse_box_to_1000(
    box: Any,
    width: int,
    height: int,
    ref_point_1000: Optional[Tuple[float, float]] = None,
) -> Optional[list[float]]:
    if not (isinstance(box, dict) and isinstance(box.get("point"), list)):
        return None
    p = box["point"]
    try:
        if len(p) == 2:
            x, y = float(p[0]), float(p[1])
            cand_a = [max(0.0, min(1000.0, x)), max(0.0, min(1000.0, y))]
            px, py = _to_1000_from_pixel(x, y, width, height)
            cand_b = [px, py]
            if ref_point_1000 is None:
                return cand_b
            da = (cand_a[0] - ref_point_1000[0]) ** 2 + (cand_a[1] - ref_point_1000[1]) ** 2
            db = (cand_b[0] - ref_point_1000[0]) ** 2 + (cand_b[1] - ref_point_1000[1]) ** 2
            return cand_a if da <= db else cand_b
        if len(p) == 4:
            x1, y1, x2, y2 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            cand_a = [max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, x2)), max(0.0, min(1000.0, y2))]
            cand_bxy1 = _to_1000_from_pixel(x1, y1, width, height)
            cand_bxy2 = _to_1000_from_pixel(x2, y2, width, height)
            cand_b = [cand_bxy1[0], cand_bxy1[1], cand_bxy2[0], cand_bxy2[1]]
            cand_a = [min(cand_a[0], cand_a[2]), min(cand_a[1], cand_a[3]), max(cand_a[0], cand_a[2]), max(cand_a[1], cand_a[3])]
            cand_b = [min(cand_b[0], cand_b[2]), min(cand_b[1], cand_b[3]), max(cand_b[0], cand_b[2]), max(cand_b[1], cand_b[3])]
            if ref_point_1000 is None:
                return cand_b
            ca = _bbox_center(cand_a)
            cb = _bbox_center(cand_b)
            da = (ca[0] - ref_point_1000[0]) ** 2 + (ca[1] - ref_point_1000[1]) ** 2
            db = (cb[0] - ref_point_1000[0]) ** 2 + (cb[1] - ref_point_1000[1]) ** 2
            return cand_a if da <= db else cand_b
    except Exception:
        return None
    return None


def map_action(
    action_raw: str,
    action_parsed: Dict[str, Any],
    box: Any,
    image_size: Tuple[int, int],
) -> Tuple[str, list[float], str, Dict[str, str], str]:
    width, height = image_size
    action_raw = _sanitize_action_call(action_raw)
    action = str(action_parsed.get("action", "")).strip()
    params = action_parsed.get("params", {}) or {}

    # Default values for non-grounding actions.
    default_bbox = [-100.0, -100.0]
    default_text = "no input text"

    if action in ("click", "long_press"):
        point_1000 = parse_point_to_1000(params.get("point"), width, height)
        bbox_1000 = parse_box_to_1000(box, width, height, ref_point_1000=point_1000)
        if point_1000 is None and bbox_1000 is not None and len(bbox_1000) == 4:
            x = (bbox_1000[0] + bbox_1000[2]) / 2.0
            y = (bbox_1000[1] + bbox_1000[3]) / 2.0
            point_1000 = (x, y)
        if point_1000 is None and bbox_1000 is not None and len(bbox_1000) == 2:
            point_1000 = (bbox_1000[0], bbox_1000[1])
        if bbox_1000 is None and point_1000 is not None:
            bbox_1000 = [point_1000[0], point_1000[1]]

        point_str = point_to_string(point_1000)
        gt_params = {"point": point_str}
        action_call = f"{action}(point={_quote_for_action(point_str)})"
        return action, bbox_1000 or default_bbox, default_text, gt_params, action_call

    if action == "type":
        content = str(params.get("content") or "")
        gt_params = {"content": content}
        action_call = f"type(content={_quote_for_action(content)})"
        return "type", default_bbox, content, gt_params, action_call

    if action == "open_app":
        app_name = str(params.get("app_name") or "")
        gt_params = {"app_name": app_name}
        action_call = f"open_app(app_name={_quote_for_action(app_name)})"
        return "open_app", default_bbox, app_name, gt_params, action_call

    if action == "swipe":
        start_px = parse_point_to_1000(params.get("start_point"), width, height)
        end_px = parse_point_to_1000(params.get("end_point"), width, height)
        velocity = _normalize_velocity(params.get("velocity"), default="600")

        if start_px is None:
            start_px = (500.0, 800.0)
        if end_px is None:
            end_px = (500.0, 200.0)

        start_str = point_to_string(start_px)
        end_str = point_to_string(end_px)
        direction = infer_swipe_direction(start_px, end_px)
        gt_params = {"start_point": start_str, "end_point": end_str, "velocity": velocity}
        action_call = (
            "swipe("
            f"start_point={_quote_for_action(start_str)}, "
            f"end_point={_quote_for_action(end_str)}, "
            f"velocity={velocity})"
        )
        return "swipe", default_bbox, direction, gt_params, action_call

    if action == "drag":
        start_px = parse_point_to_1000(params.get("start_point"), width, height)
        end_px = parse_point_to_1000(params.get("end_point"), width, height)
        if start_px is None:
            start_px = (500.0, 800.0)
        if end_px is None:
            end_px = (500.0, 200.0)
        start_str = point_to_string(start_px)
        end_str = point_to_string(end_px)
        gt_params = {"start_point": start_str, "end_point": end_str}
        action_call = (
            f"drag(start_point={_quote_for_action(start_str)}, end_point={_quote_for_action(end_str)})"
        )
        return "drag", default_bbox, default_text, gt_params, action_call

    if action == "press_home":
        return "press_home", default_bbox, default_text, {}, "press_home()"

    if action == "press_back":
        return "press_back", default_bbox, default_text, {}, "press_back()"

    if action == "wait":
        wait_t = str(params.get("t") if params.get("t") is not None else "1")
        gt_params = {"t": wait_t}
        action_call = f"wait(t={_quote_for_action(wait_t)})"
        return "wait", default_bbox, wait_t, gt_params, action_call

    if action in ("finished", "call_user", "back_information"):
        content = str(params.get("content") or params.get("message") or "")
        gt_params = {"content": content}
        action_call = f"{action}(content={_quote_for_action(content)})"
        return action, default_bbox, content, gt_params, action_call

    # Unknown actions fallback.
    content = str(params.get("content") or "")
    gt_params = {"content": content}
    action_call = f"finished(content={_quote_for_action(content)})"
    return "finished", default_bbox, content or default_text, gt_params, action_call


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

            gt_action, gt_bbox, gt_input_text, gt_params, gt_action_call = map_action(
                str(row.get("action_raw") or ""),
                action_parsed,
                row.get("box"),
                (width, height),
            )

            yield {
                "instruction": instruction,
                "history": stringify_history(row.get("history")),
                "task_type": "high",
                "image": str(image_path),
                "gt_bbox": gt_bbox,
                "gt_action": gt_action,
                "gt_input_text": gt_input_text,
                "gt_params": gt_params,
                "gt_action_call": gt_action_call,
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
