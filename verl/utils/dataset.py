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

import json
import math
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index as get_qwen2_vl_rope_index
from ..models.transformers.qwen3_vl import get_rope_index as get_qwen3_vl_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject, str], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, str):
        image = Image.open(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def _get_vl_rope_index(
    processor: ProcessorMixin,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    processor_name = processor.__class__.__name__
    image_grid_thw = model_inputs.get("image_grid_thw")
    video_grid_thw = model_inputs.get("video_grid_thw")
    second_per_grid_ts = model_inputs.get("second_per_grid_ts")

    if processor_name == "Qwen3VLProcessor":
        return get_qwen3_vl_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    return get_qwen2_vl_rope_index(
        processor,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
    )


def _load_local_dataset(data_path: str):
    path = Path(data_path)
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            return load_dataset("parquet", data_dir=str(path), split="train")
        json_files = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
        if json_files:
            return load_dataset("json", data_files=[str(x) for x in json_files], split="train")
        raise ValueError(f"No supported dataset files found in {data_path}. Expected *.parquet or *.jsonl/*.json.")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train")
    if suffix in {".jsonl", ".json"}:
        return load_dataset("json", data_files=str(path), split="train")
    raise ValueError(f"Unsupported dataset file suffix: {suffix} ({data_path})")


def _stringify_history(history: Any) -> str:
    if isinstance(history, list):
        return " | ".join(str(x) for x in history)
    if history is None:
        return ""
    return str(history)


def _to_image_list(row_dict: Dict[str, Any], image_key: str) -> List[Any]:
    image_value = row_dict.get(image_key, row_dict.get("image", row_dict.get("images")))
    if image_value is None:
        raise ValueError(f"No image found in row. Tried keys: {image_key}, image, images")
    if isinstance(image_value, list):
        return image_value
    return [image_value]


def _convert_gt_bbox_to_absolute(gt_bbox: Any, width: int, height: int) -> List[float]:
    if not isinstance(gt_bbox, list):
        return [-100.0, -100.0]
    if len(gt_bbox) not in (2, 4):
        return [-100.0, -100.0]

    bbox = [float(v) for v in gt_bbox]
    # Heuristic: normalized bbox/point in [0, 1] -> convert to pixels.
    if max(abs(v) for v in bbox) <= 1.5:
        bbox[0] *= width
        bbox[1] *= height
        if len(bbox) == 4:
            bbox[2] *= width
            bbox[3] *= height
    return bbox


def _build_prompt(instruction: str, history: str, task_type: str) -> str:
    if task_type == "high":
        action_space = [
            "complete",
            "close/delete",
            "press_home",
            "click",
            "press_back",
            "type",
            "select",
            "scroll",
            "enter",
            "open_app",
            "wait",
        ]
        return (
            f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue "
            f"executing the command '{instruction}', with the action history being '{history}'.\n"
            f"Please provide the action to perform (enumerate from {action_space}), the point where the cursor is moved "
            f"to (integer) if a click is performed, and any input text required to complete the action.\n"
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as "
            "follows:\n"
            "<think> ... </think> <answer>[{'action': enum[action_space], 'point': [x, y], "
            "'input_text': 'no input text [default]'}]</answer>\n"
            "Note:\n specific input text (no default) is necessary for actions enum['type', 'select', 'open_app'] \n"
            "for action enum['scroll'], input_text must be enum['up', 'left', 'right', 'down'].\n"
            "Examples:\n"
            "[{'action': enum['complete', 'close/delete', 'press_home', 'press_back', 'enter', 'wait'], "
            "'point': [-100, -100], 'input_text': 'no input text'}]\n"
            "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            "[{'action': enum['type', 'select', 'open_app'], 'point': [-100, -100], 'input_text': "
            "'shanghai shopping mall'}]\n"
            "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
        )

    return (
        f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue "
        f"executing the command '{instruction}', with the action history being '{history}'.\n"
        "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to "
        "(integer) if a click is performed, and any input text required to complete the action.\n"
        "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as "
        "follows:\n"
        "<think> ... </think> <answer>[{'action': enum['click'], 'point': [x, y], 'input_text': "
        "'no input text'}]</answer>\n"
        "Example:\n"
        "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
    )


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path) or os.path.isfile(data_path):
            self.dataset = _load_local_dataset(data_path)
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = dict(self.dataset[index])
        row_dict.pop("verify_bbox", None)
        row_dict.pop("success_rate", None)
        row_dict.pop("scale", None)

        instruction = str(row_dict.get("instruction", row_dict.get("task", "")))
        history = _stringify_history(row_dict.get("history", ""))
        task_type = str(row_dict.get("task_type", "high"))

        prompt_str = _build_prompt(instruction=instruction, history=history, task_type=task_type)
        messages = [{"role": "user", "content": prompt_str}]

        images = _to_image_list(row_dict, self.image_key)
        images = [process_image(image, self.max_pixels, self.min_pixels) for image in images]

        width, height = images[0].size
        gt_bbox_abs = _convert_gt_bbox_to_absolute(row_dict.get("gt_bbox", [-100, -100]), width, height)
        gt = {
            "action": str(row_dict.get("gt_action", "click")),
            "gt_bbox": gt_bbox_abs,
            "input_text": str(row_dict.get("gt_input_text", "no input text")),
        }

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")

        row_dict["multi_modal_data"] = {"image": images}
        model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict["multi_modal_inputs"] = dict(model_inputs)

        position_ids = _get_vl_rope_index(
            self.processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            model_inputs=model_inputs,
        )  # (3, seq_length)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = json.dumps(gt, ensure_ascii=False)
        return row_dict
