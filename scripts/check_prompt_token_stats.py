#!/usr/bin/env python3
"""Estimate prompt token length distribution and suggest training config."""

import argparse
import math
import random
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from verl.utils.dataset import RLHFDataset
from verl.utils.tokenizer import get_processor, get_tokenizer


def _summary(arr: List[int]) -> Dict[str, int]:
    x = np.array(arr, dtype=np.int64)
    return {
        "count": int(x.size),
        "p50": int(np.percentile(x, 50)),
        "p90": int(np.percentile(x, 90)),
        "p95": int(np.percentile(x, 95)),
        "p99": int(np.percentile(x, 99)),
        "max": int(x.max()),
    }


def _ceil_to(v: int, align: int) -> int:
    return int(math.ceil(v / float(align)) * align)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/workspace/models/Qwen3-VL-4B-Instruct/")
    parser.add_argument("--data_path", type=str, default="/root/workspace/datasets/hm_data_converted/train.jsonl")
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pixels", type=int, default=1258291)
    parser.add_argument("--min_pixels", type=int, default=0)
    parser.add_argument(
        "--probe_max_prompt_length",
        type=int,
        default=32768,
        help="Large probe length to reduce truncation during measurement.",
    )
    parser.add_argument("--suggest_percentile", type=float, default=99.0)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--reserve_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_path, trust_remote_code=True)
    processor = get_processor(args.model_path, trust_remote_code=True)

    min_pixels = None if args.min_pixels <= 0 else args.min_pixels
    dataset = RLHFDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key="instruction",
        answer_key="gt_action_call",
        image_key=args.image_key,
        max_prompt_length=args.probe_max_prompt_length,
        truncation="right",
        system_prompt="",
        min_pixels=min_pixels,
        max_pixels=args.max_pixels,
    )

    total = len(dataset)
    n = min(total, max(1, args.sample_size))
    indices = list(range(total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    indices = indices[:n]

    full_prompt_lens: List[int] = []
    text_prompt_lens: List[int] = []
    skipped = 0

    for idx in tqdm(indices, desc="Collecting prompt stats"):
        try:
            item = dataset[idx]
            full_prompt_lens.append(int(item["attention_mask"].sum().item()))
            text_prompt_lens.append(int(len(item["raw_prompt_ids"])))
        except Exception:
            skipped += 1

    if not full_prompt_lens:
        raise RuntimeError("No valid samples collected. Please check data/model paths.")

    print(f"data_path: {args.data_path}")
    print(f"model_path: {args.model_path}")
    print(f"dataset_size: {total}")
    print(f"sampled: {n}")
    print(f"valid: {len(full_prompt_lens)}")
    print(f"skipped: {skipped}")
    print(f"probe_max_prompt_length: {args.probe_max_prompt_length}")
    print()

    full_stat = _summary(full_prompt_lens)
    text_stat = _summary(text_prompt_lens)
    print("[full_prompt_len stats]  # attention_mask.sum(), multimodal prompt length")
    print(full_stat)
    print("[text_prompt_len stats]  # len(raw_prompt_ids), text-side prompt length")
    print(text_stat)
    print()

    p = float(args.suggest_percentile)
    p = max(50.0, min(99.9, p))
    suggested_prompt_len = int(np.percentile(np.array(full_prompt_lens), p))
    suggested_prompt_len = _ceil_to(suggested_prompt_len, 128)
    suggested_model_len = _ceil_to(
        suggested_prompt_len + args.max_response_length + args.reserve_tokens,
        256,
    )

    print("[suggestion]")
    print(f"suggest_percentile: p{p}")
    print(f"recommended data.max_prompt_length: {suggested_prompt_len}")
    print(
        "recommended worker.rollout.max_model_len: "
        f"{suggested_model_len}  "
        f"(= prompt {suggested_prompt_len} + response {args.max_response_length} + reserve {args.reserve_tokens})"
    )
    print()

    print("[overflow ratio on common prompt limits]")
    for limit in [2048, 3072, 4096, 5120, 6144, 7168, 8192]:
        overflow = sum(1 for v in full_prompt_lens if v > limit)
        ratio = overflow / len(full_prompt_lens)
        print(f"limit={limit:5d}  overflow={overflow:5d}/{len(full_prompt_lens)}  ratio={ratio:.2%}")


if __name__ == "__main__":
    main()
