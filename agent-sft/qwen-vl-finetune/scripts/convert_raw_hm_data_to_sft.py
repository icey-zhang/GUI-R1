#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _import_hm_utils(repo_root: Path):
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.convert_hm_data import map_action, stringify_history

    return map_action, stringify_history


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _clean_thinking(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"</?thinking>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</?answer>", "", s, flags=re.IGNORECASE)
    return s.strip()


def _build_user_prompt(instruction: str, history: Any) -> str:
    instr = str(instruction or "请完成当前界面任务。").strip()
    if not instr:
        instr = "请完成当前界面任务。"
    return f"<image>\nInstruction: {instr}\nPrevious actions: {history}"


def _resolve_image_path(input_dir: Path, episode_dir: Path, screenshot_path: str, step: Any) -> Optional[Path]:
    if screenshot_path:
        p = Path(screenshot_path)
        if p.is_absolute() and p.exists():
            return p

        normalized = screenshot_path.strip().replace("\\", "/").lstrip("./")
        if normalized.startswith("hm_data/"):
            normalized = normalized[len("hm_data/") :]
        cand = input_dir / normalized
        if cand.exists():
            return cand

        cand2 = episode_dir / Path(normalized).name
        if cand2.exists():
            return cand2

    try:
        step_int = int(step)
    except Exception:
        step_int = None
    if step_int is not None:
        cand3 = episode_dir / f"step_{step_int}.jpg"
        if cand3.exists():
            return cand3
    return None


def _build_assistant_text(thinking: str, action_call: str, response_style: str) -> str:
    if response_style == "action_only":
        return action_call
    return f"<thinking>{thinking}</thinking><answer>{action_call}</answer>"


def iter_episode_sft_rows(
    episode_dir: Path,
    input_dir: Path,
    map_action_fn,
    stringify_history_fn,
    response_style: str,
) -> Iterable[Dict[str, Any]]:
    traj_path = episode_dir / "trajectory.json"
    trace_path = episode_dir / "trace.jsonl"
    if not traj_path.exists() or not trace_path.exists():
        return

    traj = read_json(traj_path)
    if not isinstance(traj, dict):
        return

    task = traj.get("task") if isinstance(traj.get("task"), dict) else {}
    instruction = str(task.get("task_description", "")).strip() if isinstance(task, dict) else ""
    if not instruction:
        instruction = "请完成当前界面任务。"

    rows = read_jsonl(trace_path)
    for row in rows:
        action_parsed = row.get("action_parsed")
        if not isinstance(action_parsed, dict):
            continue

        img_path = _resolve_image_path(
            input_dir=input_dir,
            episode_dir=episode_dir,
            screenshot_path=str(row.get("screenshot_path") or ""),
            step=row.get("step"),
        )
        if img_path is None:
            continue

        # image size is used by map_action() for canonical 0-1000 action normalization.
        try:
            from PIL import Image

            with Image.open(img_path) as img:
                image_size = img.size
        except Exception:
            continue

        _, _, _, _, action_call = map_action_fn(
            str(row.get("action_raw") or ""),
            action_parsed,
            row.get("box"),
            image_size,
        )

        history = stringify_history_fn(row.get("history"))
        thinking_text = _clean_thinking(
            str(row.get("thinking") or row.get("explain") or row.get("summary") or "")
        )

        user_prompt = _build_user_prompt(instruction=instruction, history=history)
        assistant_text = _build_assistant_text(
            thinking=thinking_text,
            action_call=action_call,
            response_style=response_style,
        )

        yield {
            "image": str(img_path.resolve()),
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "assistant", "value": assistant_text},
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="Convert raw hm_data directory directly to SFT conversations.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/workspace/datasets/hm_data/hm_data/",
        help="Raw hm_data root directory, each subfolder contains trajectory.json + trace.jsonl + step_*.jpg",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for SFT jsonl/meta.")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Episode-level test split ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conv_style", type=str, default="chat")
    parser.add_argument("--meta_name", type=str, default="hm_data_raw_sft_train")
    parser.add_argument(
        "--response_style",
        type=str,
        choices=["answer_tag", "action_only"],
        default="answer_tag",
        help="Assistant response style.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    map_action_fn, stringify_history_fn = _import_hm_utils(repo_root=repo_root)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = [d for d in input_dir.iterdir() if d.is_dir()]
    random.Random(args.seed).shuffle(episodes)

    test_n = max(1, int(len(episodes) * args.test_ratio))
    test_set = set(episodes[:test_n])

    train_rows = []
    test_rows = []
    for ep in episodes:
        rows = list(
            iter_episode_sft_rows(
                episode_dir=ep,
                input_dir=input_dir,
                map_action_fn=map_action_fn,
                stringify_history_fn=stringify_history_fn,
                response_style=args.response_style,
            )
        )
        if ep in test_set:
            test_rows.extend(rows)
        else:
            train_rows.extend(rows)

    train_out = output_dir / "train_sft.jsonl"
    test_out = output_dir / "test_sft.jsonl"
    train_n = write_jsonl(train_out, train_rows)
    test_n_written = write_jsonl(test_out, test_rows)

    train_meta = {
        args.meta_name: {
            "root": "",
            "annotation": str(train_out),
            "conv_style": args.conv_style,
            "repeat_time": 1,
            "length": train_n,
        }
    }
    (output_dir / "meta_train.json").write_text(json.dumps(train_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    test_meta = {
        "hm_data_raw_sft_test": {
            "root": "",
            "annotation": str(test_out),
            "conv_style": args.conv_style,
            "repeat_time": 1,
            "length": test_n_written,
        }
    }
    (output_dir / "meta_test.json").write_text(json.dumps(test_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"episodes: {len(episodes)}")
    print(f"train: {train_n} -> {train_out}")
    print(f"test: {test_n_written} -> {test_out}")
    print(f"meta_train: {output_dir / 'meta_train.json'}")
    print(f"meta_test: {output_dir / 'meta_test.json'}")


if __name__ == "__main__":
    main()
