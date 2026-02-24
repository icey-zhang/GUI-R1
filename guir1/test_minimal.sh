#!/usr/bin/env bash
set -euo pipefail

# Minimal inference + eval for one split.
# Usage:
#   MODEL_PATH=/path/to/huggingface_or_checkpoint DATA_PATH=/root/workspace/datasets/GUI-R1/androidcontrol_high_test.parquet bash guir1/test_minimal.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL_PATH="${MODEL_PATH:-/root/workspace/code/GUI-R1/GUI-R1/checkpoints/easy_r1/qwen3_vl_4b_hm_data_grpo}"
DATA_PATH="${DATA_PATH:-/root/workspace/datasets/GUI-R1/androidcontrol_high_test.parquet}"
OUTPUT_PATH="${OUTPUT_PATH:-./guir1/outputs}"
NUM_ACTOR="${NUM_ACTOR:-1}"
AUTO_MERGE="${AUTO_MERGE:-1}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is empty."
  echo "Example: MODEL_PATH=/root/workspace/code/GUI-R1/GUI-R1/checkpoints/.../global_step_xxx/actor bash guir1/test_minimal.sh"
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

# Auto merge FSDP sharded checkpoint to HuggingFace format.
if [[ "${AUTO_MERGE}" == "1" ]] && compgen -G "${MODEL_PATH}/model_world_size_*_rank_0.pt" > /dev/null; then
  HF_DIR="${MODEL_PATH}/huggingface"
  if ! compgen -G "${HF_DIR}/*.safetensors" > /dev/null && ! compgen -G "${HF_DIR}/pytorch_model*.bin" > /dev/null; then
    echo "Merging sharded checkpoint -> ${HF_DIR}"
    python scripts/model_merger.py --local_dir "${MODEL_PATH}"
  fi
  MODEL_PATH="${HF_DIR}"
fi

MODEL_OUTPUT_DIR="$(basename "${MODEL_PATH}")"
MODEL_ID="${MODEL_ID:-${MODEL_OUTPUT_DIR}}"
DATA_BASENAME="$(basename "${DATA_PATH}")"
PRED_FILENAME="${DATA_BASENAME/.jsonl/_pred.jsonl}"
PRED_FILENAME="${PRED_FILENAME/.parquet/.json}"
PRED_FILE="${OUTPUT_PATH}/${MODEL_OUTPUT_DIR}/${PRED_FILENAME}"

echo "Use MODEL_PATH=${MODEL_PATH}"
echo "Use DATA_PATH=${DATA_PATH}"

python guir1/inference/inference_vllm_android.py \
  --model_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --num_actor "${NUM_ACTOR}"

python guir1/eval/eval_omni.py \
  --model_id "${MODEL_ID}" \
  --prediction_file_path "${PRED_FILE}"

echo "Done. Prediction file: ${PRED_FILE}"
