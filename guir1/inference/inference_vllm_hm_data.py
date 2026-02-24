import argparse
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

try:
    from verl.utils.reward_score.r1gui import (
        extract_action,
        extract_coord,
        extract_input_text,
        r1gui_compute_score,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from verl.utils.reward_score.r1gui import (
        extract_action,
        extract_coord,
        extract_input_text,
        r1gui_compute_score,
    )


def build_prompt(instruction: str, history: str) -> str:
    return (
        f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue "
        f"executing the command '{instruction}', with the action history being '{history}'.\n"
        "Please output exactly ONE action call using hm_data format.\n"
        "All coordinates must be in 0-1000 relative coordinate system.\n"
        "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
        "<think> ... </think> <answer>action(params...)</answer>\n"
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
        "back_information(content='')\n"
        "Examples:\n"
        "<answer>click(point='123,300')</answer>\n"
        "<answer>type(content='蒜蓉小龙虾\\n')</answer>\n"
        "<answer>finished(content='任务已完成')</answer>"
    )


def _load_rows(data_path: str) -> List[Dict[str, Any]]:
    if data_path.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=data_path, split="train")
        return [ds[i] for i in range(len(ds))]
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data) if isinstance(data, list) else [data]


def _open_image(image_value: Any) -> Image.Image:
    if isinstance(image_value, dict) and "bytes" in image_value:
        return Image.open(BytesIO(image_value["bytes"]))
    if isinstance(image_value, str):
        return Image.open(image_value)
    raise ValueError("Unsupported image format. Expect {'bytes': ...} or image path.")


def _prepare_one_sample(sample: Dict[str, Any], processor, max_pixels: int) -> Dict[str, Any]:
    image = _open_image(sample["image"]).convert("RGB")
    orig_w, orig_h = image.size
    if max_pixels > 0 and orig_w * orig_h > max_pixels:
        scale = (max_pixels / float(orig_w * orig_h)) ** 0.5
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)

    instruction = str(sample.get("instruction", "请完成当前界面任务。"))
    history = str(sample.get("history", "None") or "None")
    text = "<image>\n" + build_prompt(instruction, history)
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]

    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    out_sample = dict(sample)
    out_sample["image_size"] = [orig_w, orig_h]
    return {
        "llm_input": {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        },
        "sample": out_sample,
    }


def _iter_batches(rows: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def _build_output_path(model_path: str, data_path: str, output_root: str) -> str:
    model_name = os.path.basename(model_path.rstrip("/"))
    output_dir = os.path.join(output_root, model_name)
    os.makedirs(output_dir, exist_ok=True)
    data_name = os.path.basename(data_path)
    if data_name.endswith(".jsonl"):
        out_name = data_name.replace(".jsonl", "_pred.jsonl")
    elif data_name.endswith(".parquet"):
        out_name = data_name.replace(".parquet", "_pred.jsonl")
    elif data_name.endswith(".json"):
        out_name = data_name.replace(".json", "_pred.jsonl")
    else:
        out_name = data_name + "_pred.jsonl"
    return os.path.join(output_dir, out_name)


def main(args):
    rows = _load_rows(args.data_path)
    print(f"Loaded rows: {len(rows)}")

    processor = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 1, "video": 1},
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens,
        stop_token_ids=[],
    )

    output_path = _build_output_path(args.model_path, args.data_path, args.output_path)
    print(f"Output path: {output_path}")

    all_results: List[Dict[str, Any]] = []
    debug_left = max(0, args.debug_print_n)

    for batch_rows in tqdm(_iter_batches(rows, args.micro_batch), total=math.ceil(len(rows) / args.micro_batch)):
        prepared = [_prepare_one_sample(row, processor, args.max_pixels) for row in batch_rows]
        llm_inputs = [x["llm_input"] for x in prepared]
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)

        for meta, output in zip(prepared, outputs):
            sample = meta["sample"]
            generated_text = output.outputs[0].text if output.outputs else ""
            sample["pred"] = generated_text
            pred_coord, _ = extract_coord(generated_text)
            sample["pred_coord"] = [pred_coord[0], pred_coord[1]]
            sample["pred_action"] = extract_action(generated_text)
            sample["pred_input_text"] = extract_input_text(generated_text)
            sample["image"] = ""

            if debug_left > 0:
                print(f"[DEBUG] instruction={str(sample.get('instruction', ''))[:200]}", flush=True)
                print(f"[DEBUG] pred={generated_text}", flush=True)
                print(
                    f"[DEBUG] parsed action={sample['pred_action']}, coord={sample['pred_coord']}, input={sample['pred_input_text']}",
                    flush=True,
                )
                debug_left -= 1

            all_results.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.compute_metrics:
        total = 0
        action_hit = 0
        overall_sum = 0.0
        format_sum = 0.0
        acc_sum = 0.0
        for row in all_results:
            if "gt_action" not in row:
                continue
            total += 1
            if str(row.get("pred_action", "")).strip().lower() == str(row.get("gt_action", "")).strip().lower():
                action_hit += 1
            gt_obj = {
                "action": row.get("gt_action", ""),
                "input_text": row.get("gt_input_text", ""),
                "gt_bbox": row.get("gt_bbox", [-100, -100]),
                "gt_params": row.get("gt_params", {}),
            }
            score = r1gui_compute_score(row.get("pred", ""), json.dumps(gt_obj, ensure_ascii=False))
            overall_sum += float(score["overall"])
            format_sum += float(score["format"])
            acc_sum += float(score["accuracy"])

        if total > 0:
            print(f"[METRIC] action_exact={action_hit / total:.4f} ({action_hit}/{total})")
            print(f"[METRIC] r1gui_overall={overall_sum / total:.4f}")
            print(f"[METRIC] r1gui_format={format_sum / total:.4f}")
            print(f"[METRIC] r1gui_accuracy={acc_sum / total:.4f}")
        else:
            print("[METRIC] skip: no gt_action found in prediction rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./guir1/outputs")
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=458752)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.72)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.001)
    parser.add_argument("--debug_print_n", type=int, default=20)
    parser.add_argument("--compute_metrics", action="store_true")
    args = parser.parse_args()
    main(args)
