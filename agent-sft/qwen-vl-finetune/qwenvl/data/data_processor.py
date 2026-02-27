import json
import random
import logging
import re
import time
import itertools
import ast
import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

try:
    import sys

    sys.path.append("../internvl_chat/internvl")
    from conversation import get_conv_template
except ImportError:
    get_conv_template = None

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def read_json(path):
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    with open(path, "r") as f:
        return json.load(f)


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def find_bbox(ref_matches, message, image_size):
    """
    Extract and convert bounding box coordinates to normalized [0,1000].
    If input is already normalized, keep it unchanged.
    If input is absolute pixel coordinates, convert to normalized [0,1000].
    """
    image_width, image_height = image_size
    box_matches = re.findall(r"<box>(.*?)</box>", message)

    if not box_matches:
        box_matches = re.findall(r"\[\[.*?\]\]", message)

    if not box_matches:
        return None

    assert len(ref_matches) == len(
        box_matches
    ), f"ref_matches: {ref_matches}, box_matches: {box_matches}, message: {message}"

    formatted_values = []
    for ref, boxes in zip(ref_matches, box_matches):
        if type(boxes) == str:
            boxes = ast.literal_eval(boxes)

        if not isinstance(boxes[0], list):
            boxes = [boxes]

        if len(boxes[0]) != 4:
            return None

        new_boxes = []
        for bbox in boxes:
            # Check if coordinates are already normalized (in [0,1000] range)
            # If all coordinates are <= 1000, assume they are already normalized
            if all(0 <= coord <= 1000 for coord in bbox):
                # Already normalized, keep unchanged
                x1, y1, x2, y2 = bbox
            else:
                # Absolute pixel coordinates, convert to normalized [0,1000]
                x1, y1, x2, y2 = (
                    round(bbox[0] / image_width * 1000),
                    round(bbox[1] / image_height * 1000),
                    round(bbox[2] / image_width * 1000),
                    round(bbox[3] / image_height * 1000),
                )
            item_str = json.dumps(
                {"bbox_2d": [x1, y1, x2, y2], "label": ref}, ensure_ascii=False
            )
            formatted_values.append(item_str)

    if len(formatted_values) > 0:
        return (
            "```json\n" + "[\n    " + ",\n    ".join(formatted_values) + "\n]" + "\n```"
        )

    return message


def find_point(ref_matches, message, image_size):
    """
    Extract and convert point coordinates to normalized [0,1000].
    If input is already normalized, keep it unchanged.
    If input is absolute pixel coordinates, convert to normalized [0,1000].
    """
    image_width, image_height = image_size
    point_matches = re.findall(r"<point>(.*?)</point>", message)

    if not point_matches:
        point_matches = re.findall(r"\[\[.*?\]\]", message)

    if not point_matches:
        return None

    assert len(ref_matches) == len(
        point_matches
    ), f"ref_matches: {ref_matches}, point_matches: {point_matches}, message: {message}"

    formatted_values = []
    for ref, points in zip(ref_matches, point_matches):
        if type(points) == str:
            points = ast.literal_eval(points)
        if not isinstance(points[0], list):
            points = [points]

        if len(points[0]) != 2:
            return None

        new_points = []
        for point in points:
            # Check if coordinates are already normalized (in [0,1000] range)
            if all(0 <= coord <= 1000 for coord in point):
                # Already normalized, keep unchanged
                point = list(point)
            else:
                # Absolute pixel coordinates, convert to normalized [0,1000]
                point = [
                    round(point[0] / image_width * 1000),
                    round(point[1] / image_height * 1000),
                ]
            new_points.append(point)
            item_str = json.dumps({"point_2d": point, "label": ref}, ensure_ascii=False)
            formatted_values.append(item_str)

    if len(formatted_values) > 0:
        return (
            "```json\n" + "[\n    " + ",\n    ".join(formatted_values) + "\n]" + "\n```"
        )

    return message


def transform_coordinates(text, new_image_size):
    """
    Convert normalized coordinates (0-1 range) to absolute pixel coordinates.
    Supports formats: x=0.123, y=0.456 and [0.123, 0.456]
    """
    # Pattern for x=0.123 or y=0.456 format
    pattern = r"(x=|y=)(0?\.\d+)"

    def replace_match_1(match):
        prefix = match.group(1)  # x= or y=
        value = float(match.group(2))
        # Check if value is already normalized (0-1 range)
        if 0 <= value <= 1:
            # Already normalized (0-1), convert to [0,1000]
            if prefix == "x=":
                return f"{prefix}{round(value * 1000)}"
            else:
                return f"{prefix}{round(value * 1000)}"
        else:
            # Absolute pixel coordinates, convert to normalized [0,1000]
            if prefix == "x=":
                return f"{prefix}{round(value / new_image_size[0] * 1000)}"
            else:
                return f"{prefix}{round(value / new_image_size[1] * 1000)}"

    # Pattern for [0.123, 0.456] or [0.123, 0.456, 0.260, 0.625] format
    point_pattern = r"\[(0?\.\d+),\s*(0?\.\d+)\]"

    def replace_match_2(match):
        bbox_or_point = list(map(float, match.groups()))
        # Check if values are already normalized (0-1 range)
        if all(0 <= coord <= 1 for coord in bbox_or_point):
            # Already normalized (0-1), convert to [0,1000]
            if len(bbox_or_point) == 4:
                return json.dumps([round(coord * 1000) for coord in bbox_or_point], ensure_ascii=False)
            elif len(bbox_or_point) == 2:
                return json.dumps([round(coord * 1000) for coord in bbox_or_point], ensure_ascii=False)
        else:
            # Absolute pixel coordinates, convert to normalized [0,1000]
            if len(bbox_or_point) == 4:
                ori_x1 = round(bbox_or_point[0] / new_image_size[0] * 1000)
                ori_y1 = round(bbox_or_point[1] / new_image_size[1] * 1000)
                ori_x2 = round(bbox_or_point[2] / new_image_size[0] * 1000)
                ori_y2 = round(bbox_or_point[3] / new_image_size[1] * 1000)
                return json.dumps([ori_x1, ori_y1, ori_x2, ori_y2], ensure_ascii=False)
            elif len(bbox_or_point) == 2:
                ori_x = round(bbox_or_point[0] / new_image_size[0] * 1000)
                ori_y = round(bbox_or_point[1] / new_image_size[1] * 1000)
                return json.dumps([ori_x, ori_y])
            else:
                raise ValueError("Invalid match length")

    # Apply replacements to each command
    transformed_text = re.sub(pattern, replace_match_1, text)
    transformed_text = re.sub(point_pattern, replace_match_2, transformed_text)
    return transformed_text


def format_grounding_internvl2qwenvl(
    message,
    role,
    raw_image_size,
    new_image_size,
) -> Dict:
    """
    Convert grounding data format from InternVL to Qwen2-VL format.
    Handles coordinate conversion to normalized [0,1000].
    If input is already normalized, keep it unchanged.
    If input is absolute pixel coordinates, convert to normalized [0,1000].
    """
    if raw_image_size is None or new_image_size is None:
        return message

    if role in ("human", "user"):
        box_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        point_pattern = r"\[(\d+),\s*(\d+)\]"

        def replace_match1(match):
            bbox_or_point = list(map(int, match.groups()))
            # Check if coordinates are already normalized (in [0,1000] range)
            if all(0 <= coord <= 1000 for coord in bbox_or_point):
                # Already normalized, keep unchanged
                if len(bbox_or_point) == 4:
                    return json.dumps(bbox_or_point, ensure_ascii=False)
                elif len(bbox_or_point) == 2:
                    return json.dumps(bbox_or_point, ensure_ascii=False)
            else:
                # Absolute pixel coordinates, convert to normalized [0,1000]
                if len(bbox_or_point) == 4:
                    ori_x1 = round(bbox_or_point[0] / new_image_size[0] * 1000)
                    ori_y1 = round(bbox_or_point[1] / new_image_size[1] * 1000)
                    ori_x2 = round(bbox_or_point[2] / new_image_size[0] * 1000)
                    ori_y2 = round(bbox_or_point[3] / new_image_size[1] * 1000)
                    return json.dumps([ori_x1, ori_y1, ori_x2, ori_y2], ensure_ascii=False)
                elif len(bbox_or_point) == 2:
                    ori_x = round(bbox_or_point[0] / new_image_size[0] * 1000)
                    ori_y = round(bbox_or_point[1] / new_image_size[1] * 1000)
                    return json.dumps([ori_x, ori_y])
                else:
                    raise ValueError("Invalid match length")

        message = re.sub(box_pattern, replace_match1, message)
        message = re.sub(point_pattern, replace_match1, message)
    elif role in ("assistant", "gpt"):
        if "<ref>" in message and "</ref>" in message:
            ref_matches = re.findall(r"<ref>(.*?)</ref>", message, re.DOTALL)
            if "<point>" in message and "</point>" in message:
                message = find_point(ref_matches, message, new_image_size)
            elif "<box>" in message and "</box>" in message:
                message = find_bbox(ref_matches, message, new_image_size)
            else:
                new_message = find_bbox(ref_matches, message, new_image_size)
                if new_message is not None:
                    message = new_message
                else:
                    new_message = find_point(ref_matches, message, new_image_size)
                    message = new_message
        else:
            box_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
            point_pattern = r"\[(\d+),\s*(\d+)\]"

            def replace_match2(match):
                bbox_or_point = list(map(int, match.groups()))
                # Check if coordinates are already normalized (in [0,1000] range)
                if all(0 <= coord <= 1000 for coord in bbox_or_point):
                    # Already normalized, keep unchanged
                    if len(bbox_or_point) == 4:
                        return f"[{bbox_or_point[0]}, {bbox_or_point[1]}, {bbox_or_point[2]}, {bbox_or_point[3]}]"
                    elif len(bbox_or_point) == 2:
                        return f"[{bbox_or_point[0]}, {bbox_or_point[1]}]"
                else:
                    # Absolute pixel coordinates, convert to normalized [0,1000]
                    if len(bbox_or_point) == 4:
                        ori_x1 = round(bbox_or_point[0] / new_image_size[0] * 1000)
                        ori_y1 = round(bbox_or_point[1] / new_image_size[1] * 1000)
                        ori_x2 = round(bbox_or_point[2] / new_image_size[0] * 1000)
                        ori_y2 = round(bbox_or_point[3] / new_image_size[1] * 1000)
                        return f"[{ori_x1}, {ori_y1}, {ori_x2}, {ori_y2}]"
                    elif len(bbox_or_point) == 2:
                        ori_x = round(bbox_or_point[0] / new_image_size[0] * 1000)
                        ori_y = round(bbox_or_point[1] / new_image_size[1] * 1000)
                        return f"[{ori_x}, {ori_y}]"
                    else:
                        raise ValueError("Invalid match length")

            message = re.sub(box_pattern, replace_match2, message)
            message = re.sub(point_pattern, replace_match2, message)

    return message


def extract_image_references(text):
    pattern = r"((?:Image-\d+: <image>\n?)+)|(<image>)"
    matches = re.findall(pattern, text)

    results = []
    for group1, group2 in matches:
        if group1:
            results.append(group1.strip())
        elif group2:
            results.append(group2.strip())

    if not results:
        return ""

    return results[0]


def format_human_input(input_text):
    image_ref = extract_image_references(input_text)
    # Extract instruction
    instruction_match = re.search(
        r"Instruction:(.*?)\n+Previous actions:", input_text, re.DOTALL
    )
    instruction = instruction_match.group(1).strip() if instruction_match else ""

    # Extract previous actions
    actions_match = re.search(r"Previous actions:(.*?)$", input_text, re.DOTALL)
    actions = actions_match.group(1).strip() if actions_match else ""

    if instruction == "":
        return input_text

    if actions == "":
        actions = None

    new_header = "Please generate the next move according to the UI screenshot, the task and previous operations.\n\n"
    target_format = (
        new_header + f"Task:\n{instruction}\n\nPrevious operations:\n{actions}"
    )
    return image_ref + "\n" + target_format


def format_cot_process(input_text):
    # Extract the observation, thought, and action from the input text
    observation_match = re.search(r"Observation:(.*?)\nThought:", input_text, re.DOTALL)
    observation = observation_match.group(1).strip() if observation_match else ""

    thought_match = re.search(r"Thought:(.*?)\nAction:", input_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    action_match = re.search(r"Action:(.*?)$", input_text, re.DOTALL)
    action = action_match.group(1).strip() if action_match else ""

    # Format the extracted content into the desired format
    think_content = f"{observation} {thought}".strip()
    operation_content = action
    if think_content == "":
        target_format = f"<operation>\n{operation_content}\n</operation>"
    else:
        target_format = f"<think>\n{think_content}\n</think>\n<operation>\n{operation_content}\n</operation>"

    return target_format


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _get_image_size(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Get image size from item. Returns (width, height) or None.
    """
    # Try to get image size from item
    if "width" in item and "height" in item:
        return (item["width"], item["height"])
    if "image_wh" in item and len(item["image_wh"]) >= 2:
        return (item["image_wh"][0], item["image_wh"][1])
    return None


def _build_messages(
    item: Dict[str, Any],
    base_path: Path,
    process_coords: bool = False,
    conv_style: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Get image size for coordinate processing
    raw_image_size = _get_image_size(item) if process_coords else None
    new_image_size = None
    if raw_image_size and process_coords:
        # Calculate new image size based on processor settings
        # This is a simplified version - you may need to adjust based on your processor settings
        from qwen_vl_utils.vision_process import MIN_PIXELS, MAX_PIXELS
        new_image_size = (
            min(max(raw_image_size[0], MIN_PIXELS), MAX_PIXELS),
            min(max(raw_image_size[1], MIN_PIXELS), MAX_PIXELS)
        )

    # Build media pools with absolute paths
    image_pool = []
    for img in images:
        image_path = _make_abs_paths(base_path, img)
        if not Path(image_path).exists():
            raise ValueError(
                f"Image not found: {image_path}. Check meta root/data_path and annotation image paths."
            )
        image_pool.append({"type": "image", "image": image_path})

    video_pool = []
    for vid in videos:
        video_path = _make_abs_paths(base_path, vid)
        if not Path(video_path).exists():
            raise ValueError(
                f"Video not found: {video_path}. Check meta root/data_path and annotation video paths."
            )
        video_pool.append({"type": "video", "video": video_path})

    conversations = item["conversations"]
    if conversations and conversations[0].get("from") != "system":
        sys_prompt = None
        if conv_style in [
            "chat",
            "internvl2_5",
            "internvl3",
            "internvl_grounding",
            "internvl_referring",
        ]:
            sys_prompt = "You are a helpful assistant."
        elif conv_style and get_conv_template is not None:
            conv_temp = get_conv_template(conv_style)
            sys_prompt = conv_temp.system_message

        if sys_prompt:
            conversations = [
                {"from": "system", "value": sys_prompt},
            ] + conversations

    messages = []
    for turn in conversations:
        if turn["from"] == "system":
            role = "system"
        else:
            role = "user" if turn["from"] in ("human", "user") else "assistant"
        text: str = turn["value"]

        if role == "user":
            if "Instruction:" in text and "Previous actions:" in text:
                text = format_human_input(text)
        elif role == "assistant":
            if "Observation:" in text and "Action:" in text:
                text = format_cot_process(text)

        # Process coordinates if enabled
        if process_coords and raw_image_size and new_image_size:
            # Check if this is InternVL style grounding data
            if "<ref>" in text or "<box>" in text or "<point>" in text:
                text = format_grounding_internvl2qwenvl(
                    text, role, raw_image_size, new_image_size
                )
            # Check if this is GUI data with normalized coordinates
            elif ("x=" in text or "y=" in text) or re.search(r"\[(0?\.\d+),\s*(0?\.\d+)\]", text):
                text = transform_coordinates(text, new_image_size)

        if role == "system":
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
        elif role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
    process_coords: bool = False,
    conv_style: Optional[str] = None,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    effective_conv_style = source.get("conv_style") or conv_style
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(
        source,
        base_path,
        process_coords=process_coords,
        conv_style=effective_conv_style,
    )

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset_list = []
        dataset_use = getattr(data_args, "dataset_use", "")
        if dataset_use:
            dataset = dataset_use.split(",")
            dataset_list.extend(data_list(dataset))

        meta_path = getattr(data_args, "meta_path", "")
        if meta_path:
            meta = read_json(meta_path)
            meta_items = []
            if isinstance(meta, dict):
                meta_items = list(meta.values())
            elif isinstance(meta, list):
                if meta and all(isinstance(item, str) for item in meta):
                    for meta_item in meta:
                        sub_meta = read_json(meta_item)
                        if isinstance(sub_meta, dict):
                            meta_items.extend(list(sub_meta.values()))
                        elif isinstance(sub_meta, list):
                            meta_items.extend(sub_meta)
                        else:
                            raise ValueError(f"Unsupported meta type in {meta_item}")
                else:
                    meta_items = meta
            else:
                raise ValueError(f"Unsupported meta type in {meta_path}")

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
                dataset_list.append(entry)

        if not dataset_list:
            raise ValueError("No dataset configured. Provide dataset_use or meta_path.")

        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        self.process_coords = getattr(data_args, "process_coords", False)
        self.conv_style = getattr(data_args, "conv_style", None)
        rank0_print(f"Coordinate processing enabled: {self.process_coords}")

        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            repeat_time = data.get("repeat_time", 1)
            if repeat_time == 0:
                continue
            file_format = data["annotation_path"].split(".")[-1]
            rank0_print(f"Reading annotations from: {data['annotation_path']}")
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
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            if repeat_time > 1 and annotations:
                full_repeats = int(repeat_time)
                fractional = repeat_time - full_repeats
                extra = round(fractional * len(annotations))
                annotations = annotations * full_repeats + annotations[:extra]
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                        if "conv_style" in data and "conv_style" not in sub_ann:
                            sub_ann["conv_style"] = data["conv_style"]
                else:
                    ann["data_path"] = data["data_path"]
                    if "conv_style" in data and "conv_style" not in ann:
                        ann["conv_style"] = data["conv_style"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            process_coords=self.process_coords,
            conv_style=self.conv_style,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
