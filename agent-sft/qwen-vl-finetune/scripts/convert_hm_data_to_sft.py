#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


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


def _stringify_history(history: Any) -> str:
    if history is None:
        return "None"
    if isinstance(history, list):
        if len(history) == 0:
            return "None"
        return " | ".join(str(x) for x in history)
    text = str(history).strip()
    return text if text else "None"


def _build_user_prompt(instruction: str, history: Any) -> str:
    instruction = str(instruction or "请完成当前界面任务。").strip()
    if not instruction:
        instruction = "请完成当前界面任务。"
    history_text = _stringify_history(history)
    return f"<image>\nInstruction: {instruction}\nPrevious actions: {history_text}"


def _safe_action_call(row: Dict[str, Any]) -> str:
    action_call = str(row.get("gt_action_call", "") or "").strip()
    if action_call:
        return action_call

    action = str(row.get("gt_action", "wait") or "wait").strip()
    if action:
        return f"{action}()"
    return "wait(t='1')"


def _build_assistant_response(row: Dict[str, Any], response_style: str) -> str:
    action_call = _safe_action_call(row)
    if response_style == "action_only":
        return action_call
    return f"<thinking></thinking><answer>{action_call}</answer>"


def convert_row(row: Dict[str, Any], response_style: str) -> Optional[Dict[str, Any]]:
    image = row.get("image", "")
    if not image:
        return None
    user_text = _build_user_prompt(row.get("instruction", ""), row.get("history", "None"))
    assistant_text = _build_assistant_response(row, response_style=response_style)
    return {
        "image": str(image),
        "conversations": [
            {"from": "human", "value": user_text},
            {"from": "assistant", "value": assistant_text},
        ],
    }


def convert_split(in_path: Path, out_path: Path, response_style: str) -> int:
    rows = read_jsonl(in_path)
    converted = []
    for row in rows:
        out = convert_row(row, response_style=response_style)
        if out is not None:
            converted.append(out)
    return write_jsonl(out_path, converted)


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
    args = parser.parse_args()

    train_jsonl = Path(args.train_jsonl).expanduser().resolve()
    test_jsonl = Path(args.test_jsonl).expanduser().resolve() if args.test_jsonl else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train_sft.jsonl"
    train_n = convert_split(train_jsonl, train_out, response_style=args.response_style)

    test_out = None
    test_n = 0
    if test_jsonl is not None and test_jsonl.exists():
        test_out = output_dir / "test_sft.jsonl"
        test_n = convert_split(test_jsonl, test_out, response_style=args.response_style)

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
    if test_out is not None:
        print(f"test_out: {test_out} ({test_n})")
    print(f"meta_train: {meta_path}")


if __name__ == "__main__":
    main()
