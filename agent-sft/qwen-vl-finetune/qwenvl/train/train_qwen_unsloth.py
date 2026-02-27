#!/usr/bin/env python3
"""
使用 Unsloth 加速训练 Qwen3-VL 模型
"""

import os
import logging
import pathlib
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 重要：unsloth 必须在 transformers 之前导入
from unsloth import FastVisionModel

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import Trainer
from peft import LoraConfig, get_peft_model, TaskType

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 使用 Unsloth 加载模型
    model_name = model_args.model_name_or_path

    rank0_print(f"Loading model with Unsloth: {model_name}")

    # 使用 16-bit 训练（不使用量化）
    # Unsloth 的优化仍然可以加速训练，即使不使用 4-bit 量化
    load_in_4bit = False
    load_in_16bit = True

    rank0_print(f"Training mode: 16-bit (no quantization)")

    # 使用 FastVisionModel 加载 Qwen3-VL 模型
    # 注意：unsloth 会自动检测模型类型并应用优化
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=training_args.model_max_length,
        load_in_4bit=load_in_4bit,
        load_in_16bit=load_in_16bit,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        trust_remote_code=True,
    )

    # 获取 processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 更新 processor 参数
    from qwenvl.data.data_processor import update_processor_pixels
    processor = update_processor_pixels(processor, data_args)

    # 设置模型配置
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 设置哪些模块需要训练
    if training_args.lora_enable:
        rank0_print("LoRA enabled with Unsloth")

        # 使用 Unsloth 的 get_peft_model 方法
        model = FastVisionModel.get_peft_model(
            model,
            r=training_args.lora_r or 64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            max_seq_length=training_args.model_max_length,
        )
    else:
        # 如果不使用 LoRA，设置需要训练的模块
        # 注意：unsloth 加载的模型结构可能与原始模型略有不同
        rank0_print("Full fine-tuning mode")

        # 设置所有参数为可训练
        for param in model.parameters():
            param.requires_grad = True

        # 根据参数设置冻结特定模块
        if not model_args.tune_mm_vision:
            if hasattr(model, 'visual'):
                for n, p in model.visual.named_parameters():
                    p.requires_grad = False

        if not model_args.tune_mm_mlp:
            if hasattr(model, 'visual') and hasattr(model.visual, 'merger'):
                for n, p in model.visual.merger.named_parameters():
                    p.requires_grad = False

        if not model_args.tune_mm_llm:
            # 对于 Qwen3-VL，language model 可能在 model.model 中
            if hasattr(model, 'model'):
                for n, p in model.model.named_parameters():
                    p.requires_grad = False
            if hasattr(model, 'lm_head'):
                model.lm_head.requires_grad = False

        # 打印可训练参数（仅在 rank 0）
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            rank0_print("Trainable parameters:")
            if hasattr(model, 'visual'):
                model.visual.print_trainable_parameters()
            if hasattr(model, 'model'):
                model.model.print_trainable_parameters()

    # 准备数据
    data_module = make_supervised_data_module(processor, data_args=data_args)

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )

    # 检查是否有 checkpoint 可以恢复
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    import transformers
    train()
