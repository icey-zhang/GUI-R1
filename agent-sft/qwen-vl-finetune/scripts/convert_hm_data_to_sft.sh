#!/bin/bash
set -euo pipefail

# Input RL-style hm_data jsonl (from scripts/convert_hm_data.py)
HM_TRAIN_JSONL=${HM_TRAIN_JSONL:-/root/workspace/datasets/hm_data_converted/train.jsonl}
HM_TEST_JSONL=${HM_TEST_JSONL:-/root/workspace/datasets/hm_data_converted/test.jsonl}

# Output SFT-style files
SFT_DATA_DIR=${SFT_DATA_DIR:-/root/workspace/datasets/hm_data_sft}

# Assistant target format:
# - answer_tag: <thinking></thinking><answer>action(...)</answer>
# - action_only: action(...)
RESPONSE_STYLE=${RESPONSE_STYLE:-answer_tag}

# Conversation template style used by agent-sft data loader
CONV_STYLE=${CONV_STYLE:-chat}
META_NAME=${META_NAME:-hm_data_sft_train}

# Optional raw hm_data dir to locate trace.jsonl by image path.
# Example: /root/workspace/datasets/hm_data/hm_data
RAW_HM_DATA_DIR=${RAW_HM_DATA_DIR:-}

# Thinking extraction priority in trace rows.
THINKING_FIELDS=${THINKING_FIELDS:-thinking,explain,summary}

mkdir -p "${SFT_DATA_DIR}"

python ./scripts/convert_hm_data_to_sft.py \
  --train_jsonl "${HM_TRAIN_JSONL}" \
  --test_jsonl "${HM_TEST_JSONL}" \
  --output_dir "${SFT_DATA_DIR}" \
  --response_style "${RESPONSE_STYLE}" \
  --conv_style "${CONV_STYLE}" \
  --meta_name "${META_NAME}" \
  --raw_hm_data_dir "${RAW_HM_DATA_DIR}" \
  --thinking_fields "${THINKING_FIELDS}"

echo "Done."
echo "train_sft: ${SFT_DATA_DIR}/train_sft.jsonl"
echo "test_sft:  ${SFT_DATA_DIR}/test_sft.jsonl"
echo "meta:      ${SFT_DATA_DIR}/meta_train.json"
