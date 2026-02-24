#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL_PATH="${MODEL_PATH:-/root/workspace/code/GUI-R1/GUI-R1/checkpoints/easy_r1/qwen3_vl_4b_hm_data_grpo}"
DATA_PATH="${DATA_PATH:-/root/workspace/datasets/hm_data_converted/test.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-./guir1/outputs}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
AUTO_MERGE="${AUTO_MERGE:-1}"

MICRO_BATCH="${MICRO_BATCH:-1}"
TP_SIZE="${TP_SIZE:-1}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.001}"
DEBUG_PRINT_N="${DEBUG_PRINT_N:-20}"

export CUDA_VISIBLE_DEVICES

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is empty."
  exit 1
fi
if [[ -z "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH is empty."
  exit 1
fi

MODEL_PATH="${MODEL_PATH%/}"

# Support passing checkpoint root/save path:
# 1) .../checkpoints/.../qwen3_vl_4b_hm_data_grpo
# 2) .../global_step_xxx
# 3) .../global_step_xxx/actor
# 4) .../global_step_xxx/actor/huggingface
if [[ -f "${MODEL_PATH}/latest_global_step.txt" ]]; then
  STEP="$(cat "${MODEL_PATH}/latest_global_step.txt")"
  MODEL_PATH="${MODEL_PATH}/global_step_${STEP}"
fi
if [[ -d "${MODEL_PATH}/actor" ]]; then
  MODEL_PATH="${MODEL_PATH}/actor"
fi

if [[ "${AUTO_MERGE}" == "1" ]] && compgen -G "${MODEL_PATH}/model_world_size_*_rank_0.pt" > /dev/null; then
  HF_DIR="${MODEL_PATH}/huggingface"
  if ! compgen -G "${HF_DIR}/*.safetensors" > /dev/null && ! compgen -G "${HF_DIR}/pytorch_model*.bin" > /dev/null; then
    echo "Merging sharded checkpoint -> ${HF_DIR}"
    python scripts/model_merger.py --local_dir "${MODEL_PATH}"
  fi
  MODEL_PATH="${HF_DIR}"
fi

echo "Use MODEL_PATH=${MODEL_PATH}"
echo "Use DATA_PATH=${DATA_PATH}"
echo "Use OUTPUT_PATH=${OUTPUT_PATH}"

python guir1/inference/inference_vllm_hm_data.py \
  --model_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --micro_batch "${MICRO_BATCH}" \
  --tensor_parallel_size "${TP_SIZE}" \
  --max_pixels "${MAX_PIXELS}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --max_tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --debug_print_n "${DEBUG_PRINT_N}" \
  --compute_metrics
