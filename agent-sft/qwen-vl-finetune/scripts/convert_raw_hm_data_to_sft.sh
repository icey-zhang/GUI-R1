#!/bin/bash
set -euo pipefail

# Raw hm_data directory (episode folders with trajectory.json / trace.jsonl / step_*.jpg)
INPUT_DIR=${INPUT_DIR:-/root/workspace/datasets/hm_data/hm_data/}

# Output converted SFT jsonl + meta
SFT_DATA_DIR=${SFT_DATA_DIR:-/root/workspace/datasets/hm_data_sft_raw}

# Split settings
TEST_RATIO=${TEST_RATIO:-0.05}
SEED=${SEED:-42}

# Conversation and output format
CONV_STYLE=${CONV_STYLE:-chat}
META_NAME=${META_NAME:-hm_data_raw_sft_train}
RESPONSE_STYLE=${RESPONSE_STYLE:-answer_tag}

# Optional: align split with existing GRPO split files.
USE_GRPO_SPLIT=${USE_GRPO_SPLIT:-1}
GRPO_TRAIN_JSONL=${GRPO_TRAIN_JSONL:-/root/workspace/datasets/hm_data_converted/train.jsonl}
GRPO_TEST_JSONL=${GRPO_TEST_JSONL:-/root/workspace/datasets/hm_data_converted/test.jsonl}

mkdir -p "${SFT_DATA_DIR}"

EXTRA_ARGS=""
if [[ "${USE_GRPO_SPLIT}" == "1" ]] && [[ -f "${GRPO_TRAIN_JSONL}" ]] && [[ -f "${GRPO_TEST_JSONL}" ]]; then
  EXTRA_ARGS="--grpo_train_jsonl ${GRPO_TRAIN_JSONL} --grpo_test_jsonl ${GRPO_TEST_JSONL}"
fi

python ./scripts/convert_raw_hm_data_to_sft.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${SFT_DATA_DIR}" \
  --test_ratio "${TEST_RATIO}" \
  --seed "${SEED}" \
  --conv_style "${CONV_STYLE}" \
  --meta_name "${META_NAME}" \
  --response_style "${RESPONSE_STYLE}" \
  ${EXTRA_ARGS}

echo "Done."
echo "train_sft: ${SFT_DATA_DIR}/train_sft.jsonl"
echo "test_sft:  ${SFT_DATA_DIR}/test_sft.jsonl"
echo "meta:      ${SFT_DATA_DIR}/meta_train.json"
