#!/bin/bash
set -euo pipefail

# Single-node distributed training config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# Paths
MODEL_PATH=${MODEL_PATH:-/root/workspace/models/Qwen3-VL-4B-Instruct/}
HM_TRAIN_JSONL=${HM_TRAIN_JSONL:-/root/workspace/datasets/hm_data_converted/train.jsonl}
HM_TEST_JSONL=${HM_TEST_JSONL:-/root/workspace/datasets/hm_data_converted/test.jsonl}
SFT_DATA_DIR=${SFT_DATA_DIR:-/root/workspace/datasets/hm_data_sft}
OUTPUT_DIR=${OUTPUT_DIR:-./output_hm_sft_qwen3_4b}

# Hyperparameters
LR=${LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_PIXELS=${MAX_PIXELS:-1258291}
MIN_PIXELS=${MIN_PIXELS:-3136}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}

RUN_NAME=${RUN_NAME:-qwen3vl_4b_hm_data_sft}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-./scripts/zero3.json}

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SFT_DATA_DIR}"

python ./scripts/convert_hm_data_to_sft.py \
  --train_jsonl "${HM_TRAIN_JSONL}" \
  --test_jsonl "${HM_TEST_JSONL}" \
  --output_dir "${SFT_DATA_DIR}" \
  --response_style answer_tag \
  --conv_style chat \
  --meta_name hm_data_sft_train

META_PATH="${SFT_DATA_DIR}/meta_train.json"

ARGS="
  --deepspeed ${DEEPSPEED_CFG} \
  --model_name_or_path ${MODEL_PATH} \
  --meta_path ${META_PATH} \
  --data_flatten True \
  --coord_norm False \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --bf16 \
  --output_dir ${OUTPUT_DIR} \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size $((BATCH_SIZE*2)) \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --max_pixels ${MAX_PIXELS} \
  --min_pixels ${MIN_PIXELS} \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 2 \
  --learning_rate ${LR} \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length ${MODEL_MAX_LENGTH} \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --run_name ${RUN_NAME} \
  --report_to wandb"

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py ${ARGS}
