#!/bin/bash

# 使用 Unsloth 加速训练 Qwen3-VL-2B 模型
# 相比原始的 DeepSpeed 训练，Unsloth 可以提供：
# - 2x 更快的训练速度
# - 更长的上下文支持
# - 优化的内核和内存管理
#
# 注意：Unsloth 目前不支持多 GPU 训练！
# 如果需要多 GPU 训练，请使用原始的 DeepSpeed 训练脚本

# Model configuration
llm=/root/workspace/models/Qwen3-VL-2B-Instruct/  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-5
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen_unsloth.py

# Dataset configuration (replace with public dataset names)
datasets=scalecua_data

# Output configuration
run_name="qwen3vl-unsloth"
output_dir=./output_unsloth

# Training arguments
args="
    --model_name_or_path "${llm}" \
    --meta_path /root/workspace/datasets/ScaleCUA-Data/meta.json \
    --coord_norm False \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --resume_from_checkpoint True \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 2109744 \
    --min_pixels 3136 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 40960 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training with Unsloth
# 注意：Unsloth 不需要 torchrun，直接运行即可
# Unsloth 目前不支持多 GPU 训练！如果需要多 GPU 训练，请使用原始的 DeepSpeed 训练脚本
echo "Starting training with Unsloth..."
echo "Model: ${llm}"
echo "Output: ${output_dir}"
echo "Batch size: ${batch_size} x ${grad_accum_steps} = $((batch_size * grad_accum_steps))"
echo "Training mode: 16-bit (no quantization)"
echo "Note: Unsloth single GPU training only (multi-GPU not supported)"

# 直接运行 Unsloth 训练（单 GPU）
python ${entry_file} ${args}
