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
"""Utils for tokenization."""

import types
import warnings
from typing import Optional

from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin

__all__ = ["get_tokenizer", "get_processor"]


def _set_pad_token(tokenizer: PreTrainedTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Set to {tokenizer.eos_token_id}.", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Set to {tokenizer.eos_token}.", stacklevel=1)


def get_tokenizer(model_path: str, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    _set_pad_token(tokenizer)

    return tokenizer


def _bind_get_rope_index(processor: ProcessorMixin, model_path: str, **kwargs) -> None:
    """Bind model-specific get_rope_index to processor for VLM models."""
    processor_name = processor.__class__.__name__
    config = AutoConfig.from_pretrained(model_path, **kwargs)
    processor.config = config

    try:
        if processor_name == "Qwen2VLProcessor":
            from transformers.models.qwen2_vl import Qwen2VLModel

            processor.get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, processor)
        elif processor_name == "Qwen2_5_VLProcessor":
            from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

            processor.get_rope_index = types.MethodType(Qwen2_5_VLModel.get_rope_index, processor)
        elif processor_name == "Qwen3VLProcessor":
            from transformers.models.qwen3_vl import Qwen3VLModel

            processor.get_rope_index = types.MethodType(Qwen3VLModel.get_rope_index, processor)
    except Exception as e:  # pragma: no cover - depends on installed transformers version
        warnings.warn(f"Bind get_rope_index failed for {processor_name}: {e}", stacklevel=1)


def get_processor(model_path: str, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except Exception:
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    if processor is not None:
        try:
            _bind_get_rope_index(processor, model_path, **kwargs)
        except Exception as e:  # pragma: no cover - keep backward compatibility
            warnings.warn(f"Failed to prepare processor: {e}", stacklevel=1)

    return processor
