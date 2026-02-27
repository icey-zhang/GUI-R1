#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from transformers import AutoProcessor

from qwenvl.data import data_list
from qwenvl.data.data_processor import (
    _build_messages,
    preprocess_qwen_visual,
    read_json,
    read_jsonl,
    update_processor_pixels,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview prompt composition without training"
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--meta_path", default="")
    parser.add_argument("--dataset_use", default="")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--process_coords", action="store_true")
    parser.add_argument("--conv_style", default="")
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--print_messages", action="store_true")
    parser.add_argument("--print_decoded", action="store_true")
    return parser.parse_args()


def build_dataset_list(args):
    dataset_list = []

    if args.dataset_use:
        dataset = args.dataset_use.split(",")
        dataset_list.extend(data_list(dataset))

    if args.meta_path:
        meta = read_json(args.meta_path)
        meta_items = []
        if isinstance(meta, dict):
            for name, item in meta.items():
                if isinstance(item, dict):
                    item = item.copy()
                    item["_dataset_name"] = name
                meta_items.append(item)
        elif isinstance(meta, list):
            if meta and all(isinstance(item, str) for item in meta):
                for meta_item in meta:
                    sub_meta = read_json(meta_item)
                    if isinstance(sub_meta, dict):
                        for name, item in sub_meta.items():
                            if isinstance(item, dict):
                                item = item.copy()
                                item["_dataset_name"] = name
                            meta_items.append(item)
                    elif isinstance(sub_meta, list):
                        meta_items.extend(sub_meta)
                    else:
                        raise ValueError(f"Unsupported meta type in {meta_item}")
            else:
                meta_items = meta
        else:
            raise ValueError(f"Unsupported meta type in {args.meta_path}")

        for item in meta_items:
            if not isinstance(item, dict):
                raise ValueError("Meta item must be a dict")
            annotation_path = item.get("annotation") or item.get("annotation_path")
            if not annotation_path:
                raise ValueError("Meta item missing annotation path")
            entry = {
                "annotation_path": annotation_path,
                "data_path": item.get("root") or item.get("data_path") or "",
            }
            if "conv_style" in item:
                entry["conv_style"] = item["conv_style"]
            if "sampling_rate" in item:
                entry["sampling_rate"] = item["sampling_rate"]
            if "repeat_time" in item:
                entry["repeat_time"] = item["repeat_time"]
            if "_dataset_name" in item:
                entry["_dataset_name"] = item["_dataset_name"]
            dataset_list.append(entry)

    if not dataset_list:
        raise ValueError("No dataset configured. Provide dataset_use or meta_path.")

    return dataset_list


def load_samples(dataset_list, seed):
    rng = random.Random(seed)
    samples = []
    skipped_lists = 0

    for data in dataset_list:
        repeat_time = data.get("repeat_time", 1)
        if repeat_time == 0:
            continue

        file_format = data["annotation_path"].split(".")[-1]
        if file_format == "jsonl":
            annotations = read_jsonl(data["annotation_path"])
        else:
            annotations = json.load(open(data["annotation_path"], "r"))

        sampling_rate = data.get("sampling_rate", 1.0)
        if 0 < repeat_time < 1:
            sampling_rate = min(sampling_rate, repeat_time)
        if sampling_rate < 1.0:
            annotations = random.sample(
                annotations, int(len(annotations) * sampling_rate)
            )

        if repeat_time > 1 and annotations:
            full_repeats = int(repeat_time)
            fractional = repeat_time - full_repeats
            extra = round(fractional * len(annotations))
            annotations = annotations * full_repeats + annotations[:extra]

        for ann in annotations:
            if isinstance(ann, list):
                skipped_lists += 1
                continue
            ann["data_path"] = data.get("data_path", "")
            if "conv_style" in data and "conv_style" not in ann:
                ann["conv_style"] = data["conv_style"]
            if "_dataset_name" in data:
                ann["_dataset_name"] = data["_dataset_name"]
            samples.append(ann)

    rng.shuffle(samples)
    return samples, skipped_lists


def parse_indices(text):
    if not text:
        return []
    return [int(part) for part in text.split(",") if part.strip()]


def main():
    args = parse_args()

    dataset_list = build_dataset_list(args)
    samples, skipped_lists = load_samples(dataset_list, args.seed)

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if args.min_pixels is not None or args.max_pixels is not None:
        data_args = type("Args", (), {})()
        data_args.min_pixels = args.min_pixels
        data_args.max_pixels = args.max_pixels
        data_args.video_min_pixels = getattr(args, "video_min_pixels", 0)
        data_args.video_max_pixels = getattr(args, "video_max_pixels", 0)
        data_args.video_min_frames = getattr(args, "video_min_frames", 0)
        data_args.video_max_frames = getattr(args, "video_max_frames", 0)
        data_args.video_fps = getattr(args, "video_fps", 0)
        processor = update_processor_pixels(processor, data_args)

    indices = parse_indices(args.sample_indices)
    if indices:
        chosen = [samples[i] for i in indices if i < len(samples)]
    else:
        chosen = samples[: args.num_samples]

    if skipped_lists:
        print(f"Skipped {skipped_lists} packed list entries")

    for idx, sample in enumerate(chosen):
        dataset_name = sample.get("_dataset_name", "")
        header = f"Sample {idx}"
        if dataset_name:
            header += f" | dataset={dataset_name}"
        print("=" * 80)
        print(header)
        print("image:", sample.get("image"))

        conv_style = sample.get("conv_style") or args.conv_style or None
        messages = _build_messages(
            sample,
            Path(sample.get("data_path", "")),
            process_coords=args.process_coords,
            conv_style=conv_style,
        )

        if args.print_messages:
            print("\n[Messages]")
            for m in messages:
                print(m)

        if args.print_decoded:
            data_dict = preprocess_qwen_visual(
                [sample],
                processor,
                process_coords=args.process_coords,
                conv_style=conv_style,
            )
            decoded = processor.tokenizer.decode(
                data_dict["input_ids"][0], skip_special_tokens=False
            )
            print("\n[Decoded Prompt]")
            print(decoded)

    print("=" * 80)
    print(f"Total samples available: {len(samples)}")


if __name__ == "__main__":
    main()
