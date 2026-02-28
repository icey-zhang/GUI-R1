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

mkdir -p "${SFT_DATA_DIR}"

python ./scripts/convert_raw_hm_data_to_sft.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${SFT_DATA_DIR}" \
  --test_ratio "${TEST_RATIO}" \
  --seed "${SEED}" \
  --conv_style "${CONV_STYLE}" \
  --meta_name "${META_NAME}" \
  --response_style "${RESPONSE_STYLE}"

echo "Done."
echo "train_sft: ${SFT_DATA_DIR}/train_sft.jsonl"
echo "test_sft:  ${SFT_DATA_DIR}/test_sft.jsonl"
echo "meta:      ${SFT_DATA_DIR}/meta_train.json"
