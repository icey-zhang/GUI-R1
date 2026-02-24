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


def _normalize_gt_params(gt_params: Any) -> Dict[str, str]:
    if isinstance(gt_params, dict):
        return {str(k): str(v) for k, v in gt_params.items()}
    if isinstance(gt_params, str) and gt_params.strip():
        try:
            obj = json.loads(gt_params)
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        except Exception:
            return {}
    return {}


def _point_str_from_bbox(gt_bbox: List[float]) -> str:
    if len(gt_bbox) == 2:
        return f"{int(round(gt_bbox[0]))},{int(round(gt_bbox[1]))}"
    if len(gt_bbox) == 4:
        x = (gt_bbox[0] + gt_bbox[2]) / 2.0
        y = (gt_bbox[1] + gt_bbox[3]) / 2.0
        return f"{int(round(x))},{int(round(y))}"
    return "-100,-100"


def _derive_gt_params(action: str, gt_bbox: List[float], gt_input_text: str) -> Dict[str, str]:
    action = str(action).strip().lower()
    if action in {"click", "long_press"}:
        return {"point": _point_str_from_bbox(gt_bbox)}
    if action == "swipe":
        return {"direction": str(gt_input_text)}
    if action == "type":
        return {"content": str(gt_input_text)}
    if action == "open_app":
        return {"app_name": str(gt_input_text)}
    if action == "wait":
        return {"t": str(gt_input_text)}
    if action in {"finished", "call_user", "back_information", "complete"}:
        return {"content": str(gt_input_text)}
    return {}


def _escape_action_str(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("'", "\\'")
    )


def _build_action_call(action: str, params: Dict[str, str]) -> str:
    action = str(action).strip().lower()
    if action in {"click", "long_press"}:
        point = params.get("point", "-100,-100")
        return f"{action}(point='{_escape_action_str(point)}')"
    if action in {"swipe", "drag"}:
        start_point = params.get("start_point")
        end_point = params.get("end_point")
        if start_point and end_point:
            if action == "swipe":
                try:
                    velocity_val = float(params.get("velocity", "600"))
                    velocity = str(int(velocity_val)) if abs(velocity_val - int(velocity_val)) < 1e-6 else str(velocity_val)
                except Exception:
                    velocity = "600"
                return (
                    f"swipe(start_point='{_escape_action_str(start_point)}', "
                    f"end_point='{_escape_action_str(end_point)}', velocity={velocity})"
                )
            return (
                f"drag(start_point='{_escape_action_str(start_point)}', "
                f"end_point='{_escape_action_str(end_point)}')"
            )
    if action == "type":
        content = params.get("content", "")
        return f"type(content='{_escape_action_str(content)}')"
    if action == "open_app":
        app_name = params.get("app_name", "")
        return f"open_app(app_name='{_escape_action_str(app_name)}')"
    if action == "wait":
        t = params.get("t", "1")
        return f"wait(t='{_escape_action_str(t)}')"
    if action in {"finished", "call_user", "back_information", "complete"}:
        content = params.get("content", "")
        final_action = "finished" if action == "complete" else action
        return f"{final_action}(content='{_escape_action_str(content)}')"
    if action == "press_home":
        return "press_home()"
    if action == "press_back":
        return "press_back()"
    return f"{action}()"


def _build_prompt(instruction: str, history: str, task_type: str) -> str:
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
            f"executing the command '{instruction}', with the action history being '{history}'.\n"
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
        f"executing the command '{instruction}', with the action history being '{history}'.\n"
        "Please output exactly one action call in hm_data format.\n"
        "Coordinates must be in 0-1000 relative coordinate system.\n"
        "Output the thinking process in <thinking> </thinking> tags, and the final answer in <answer> </answer> tags as "
        "follows:\n"
        "<thinking> ... </thinking> <answer>click(point='x1,y1')</answer>\n"
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
        gt_action = str(row_dict.get("gt_action", "click"))
        gt_input_text = str(row_dict.get("gt_input_text", "no input text"))
        gt_params = _normalize_gt_params(row_dict.get("gt_params", {}))
        if not gt_params:
            gt_params = _derive_gt_params(gt_action, gt_bbox_abs, gt_input_text)
        gt_action_call = str(row_dict.get("gt_action_call", "")).strip()
        if not gt_action_call:
            gt_action_call = _build_action_call(gt_action, gt_params)

        gt = {
            "action": gt_action,
            "gt_bbox": gt_bbox_abs,
            "input_text": gt_input_text,
            "gt_params": gt_params,
            "action_call": gt_action_call,
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
