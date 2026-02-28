#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
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
    if not s:
        return ""
    # avoid breaking output tag structure
    s = re.sub(r"</?thinking>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</?think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</?answer>", "", s, flags=re.IGNORECASE)
    return s.strip()


def _stringify_history(history: Any) -> str:
    if history is None:
        return "None"
    if isinstance(history, list):
        if len(history) == 0:
            return "None"
        return " | ".join(str(x) for x in history)
    text = str(history).strip()
    return text if text else "None"


def _build_user_prompt(instruction: str, history: Any, task_type: str = "high") -> str:
    instruction = str(instruction or "请完成当前界面任务。").strip()
    if not instruction:
        instruction = "请完成当前界面任务。"
    history_text = _stringify_history(history)
    task_type = str(task_type or "high").strip().lower()
    if task_type == "high":
        action_space = [
            "click",
            "long_press",
            "type",
            "swipe",
            "open_app",
            "drag",
            "press_home",
            "press_back",
            "wait",
            "finished",
            "call_user",
            "back_information",
        ]
        return (
            f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue "
            f"executing the command '{instruction}', with the action history being '{history_text}'.\n"
            f"Please output exactly ONE action call from {action_space} using hm_data format.\n"
            "All coordinates must be in 0-1000 relative coordinate system.\n"
            "Output the thinking process in <thinking> </thinking> tags, and the final answer in <answer> </answer> tags as "
            "follows:\n"
            "<thinking> ... </thinking> <answer>action(params...)</answer>\n"
            "Available actions and signatures:\n"
            "click(point='x1,y1')\n"
            "long_press(point='x1,y1')\n"
            "type(content='')\n"
            "swipe(start_point='x1,y1', end_point='x2,y2', velocity=600)\n"
            "open_app(app_name='')\n"
            "drag(start_point='x1,y1', end_point='x2,y2')\n"
            "press_home()\n"
            "press_back()\n"
            "wait(t='t')\n"
            "finished(content='')\n"
            "call_user(content='')\n"
            "back_information(content='')"
        )

    return (
        f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue "
        f"executing the command '{instruction}', with the action history being '{history_text}'.\n"
        "Please output exactly one action call in hm_data format.\n"
        "Coordinates must be in 0-1000 relative coordinate system.\n"
        "Output the thinking process in <thinking> </thinking> tags, and the final answer in <answer> </answer> tags as "
        "follows:\n"
        "<thinking> ... </thinking> <answer>click(point='x1,y1')</answer>\n"
    )


def _safe_action_call(row: Dict[str, Any]) -> str:
    action_call = str(row.get("gt_action_call", "") or "").strip()
    if action_call:
        return action_call

    action = str(row.get("gt_action", "wait") or "wait").strip()
    if action:
        return f"{action}()"
    return "wait(t='1')"


def _build_assistant_response(row: Dict[str, Any], response_style: str, thinking_text: str) -> str:
    action_call = _safe_action_call(row)
    if response_style == "action_only":
        return action_call
    return f"<thinking>{thinking_text}</thinking><answer>{action_call}</answer>"


class ThinkingResolver:
    def __init__(self, raw_hm_data_dir: Optional[Path], thinking_fields: Sequence[str]):
        self.raw_hm_data_dir = raw_hm_data_dir
        self.thinking_fields = [f.strip() for f in thinking_fields if str(f).strip()]
        self.trace_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def _parse_step(self, image_path: Path, row: Dict[str, Any]) -> Optional[int]:
        source_step = row.get("source_step")
        if source_step is not None:
            try:
                return int(source_step)
            except Exception:
                pass
        m = re.match(r"step_(\d+)$", image_path.stem)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _locate_trace(self, image_path: Path) -> Optional[Path]:
        # Most converted samples store absolute image path under episode dir.
        cand = image_path.parent / "trace.jsonl"
        if cand.exists():
            return cand

        if self.raw_hm_data_dir is not None:
            episode_name = image_path.parent.name
            cand2 = self.raw_hm_data_dir / episode_name / "trace.jsonl"
            if cand2.exists():
                return cand2
        return None

    def _load_trace_index(self, trace_path: Path) -> Dict[int, Dict[str, Any]]:
        cache_key = str(trace_path.resolve())
        if cache_key in self.trace_cache:
            return self.trace_cache[cache_key]

        idx: Dict[int, Dict[str, Any]] = {}
        for row in read_jsonl(trace_path):
            try:
                step = int(row.get("step"))
            except Exception:
                continue
            idx[step] = row
        self.trace_cache[cache_key] = idx
        return idx

    def find_thinking(self, image: str, row: Dict[str, Any]) -> str:
        if not image:
            return ""
        image_path = Path(str(image)).expanduser()
        step = self._parse_step(image_path=image_path, row=row)
        if step is None:
            return ""

        trace_path = self._locate_trace(image_path=image_path)
        if trace_path is None:
            return ""

        trace_idx = self._load_trace_index(trace_path=trace_path)
        trace_row = trace_idx.get(step)
        if not isinstance(trace_row, dict):
            return ""

        for field in self.thinking_fields:
            value = _clean_thinking(trace_row.get(field, ""))
            if value:
                return value
        return ""


def convert_row(row: Dict[str, Any], response_style: str, thinking_resolver: ThinkingResolver) -> Optional[Dict[str, Any]]:
    image = row.get("image", "")
    if not image:
        return None
    user_text = _build_user_prompt(
        row.get("instruction", ""),
        row.get("history", "None"),
        task_type=str(row.get("task_type", "high")),
    )
    thinking_text = thinking_resolver.find_thinking(image=str(image), row=row)
    assistant_text = _build_assistant_response(row, response_style=response_style, thinking_text=thinking_text)
    return {
        "image": str(image),
        "conversations": [
            {"from": "human", "value": user_text},
            {"from": "assistant", "value": assistant_text},
        ],
    }


def convert_split(in_path: Path, out_path: Path, response_style: str, thinking_resolver: ThinkingResolver) -> tuple[int, int]:
    rows = read_jsonl(in_path)
    converted = []
    thinking_hit = 0
    for row in rows:
        out = convert_row(row, response_style=response_style, thinking_resolver=thinking_resolver)
        if out is not None:
            converted.append(out)
            try:
                assistant = out["conversations"][1]["value"]
            except Exception:
                assistant = ""
            if "<thinking></thinking>" not in assistant:
                thinking_hit += 1
    return write_jsonl(out_path, converted), thinking_hit


def main():
    parser = argparse.ArgumentParser(description="Convert GUI-R1 hm_data jsonl to agent-sft conversations format.")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Input hm_data train jsonl path.")
    parser.add_argument("--test_jsonl", type=str, default="", help="Input hm_data test jsonl path (optional).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--response_style",
        type=str,
        choices=["answer_tag", "action_only"],
        default="answer_tag",
        help="Assistant target format.",
    )
    parser.add_argument("--conv_style", type=str, default="chat", help="conv_style in generated meta.")
    parser.add_argument("--meta_name", type=str, default="hm_data_sft_train", help="Dataset key in meta json.")
    parser.add_argument(
        "--raw_hm_data_dir",
        type=str,
        default="",
        help="Optional raw hm_data dir for trace lookup fallback. Example: /root/workspace/datasets/hm_data/hm_data",
    )
    parser.add_argument(
        "--thinking_fields",
        type=str,
        default="thinking,explain,summary",
        help="Comma-separated field priority when extracting thinking from trace rows.",
    )
    args = parser.parse_args()

    train_jsonl = Path(args.train_jsonl).expanduser().resolve()
    test_jsonl = Path(args.test_jsonl).expanduser().resolve() if args.test_jsonl else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_hm_data_dir = Path(args.raw_hm_data_dir).expanduser().resolve() if args.raw_hm_data_dir else None
    thinking_fields = [x.strip() for x in args.thinking_fields.split(",") if x.strip()]
    thinking_resolver = ThinkingResolver(raw_hm_data_dir=raw_hm_data_dir, thinking_fields=thinking_fields)

    train_out = output_dir / "train_sft.jsonl"
    train_n, train_thinking_hit = convert_split(
        train_jsonl, train_out, response_style=args.response_style, thinking_resolver=thinking_resolver
    )

    test_out = None
    test_n = 0
    test_thinking_hit = 0
    if test_jsonl is not None and test_jsonl.exists():
        test_out = output_dir / "test_sft.jsonl"
        test_n, test_thinking_hit = convert_split(
            test_jsonl, test_out, response_style=args.response_style, thinking_resolver=thinking_resolver
        )

    meta = {
        args.meta_name: {
            "root": "",
            "annotation": str(train_out),
            "conv_style": args.conv_style,
            "repeat_time": 1,
            "length": train_n,
        }
    }
    meta_path = output_dir / "meta_train.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if test_out is not None:
        test_meta = {
            "hm_data_sft_test": {
                "root": "",
                "annotation": str(test_out),
                "conv_style": args.conv_style,
                "repeat_time": 1,
                "length": test_n,
            }
        }
        (output_dir / "meta_test.json").write_text(json.dumps(test_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train_in: {train_jsonl}")
    print(f"train_out: {train_out} ({train_n})")
    if args.response_style == "answer_tag":
        print(f"train_thinking_found: {train_thinking_hit}/{train_n}")
    if test_out is not None:
        print(f"test_out: {test_out} ({test_n})")
        if args.response_style == "answer_tag":
            print(f"test_thinking_found: {test_thinking_hit}/{test_n}")
    print(f"meta_train: {meta_path}")


if __name__ == "__main__":
    main()
