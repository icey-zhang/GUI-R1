set -x

# 1) Convert hm_data to train/test jsonl
python scripts/convert_hm_data.py \
  --input_dir /root/workspace/datasets/hm_data/hm_data/ \
  --output_dir /root/workspace/datasets/hm_data_converted \
  --test_ratio 0.05 \
  --seed 42

# 2) Run training
MODEL_PATH=/root/workspace/models/Qwen3-VL-4B-Instruct/
SYSTEM_PROMPT=""""""
N_GPUS=${N_GPUS:-4}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints/easy_r1/qwen3_vl_4b_hm_data_grpo}
GUIR1_TRAIN_DEBUG_PRINT_N=${GUIR1_TRAIN_DEBUG_PRINT_N:-8}
GUIR1_TRAIN_DEBUG_PRINT_INTERVAL=${GUIR1_TRAIN_DEBUG_PRINT_INTERVAL:-1}
RESUME_ARGS=""

export GUIR1_TRAIN_DEBUG_PRINT_N
export GUIR1_TRAIN_DEBUG_PRINT_INTERVAL

if [ -f "${CHECKPOINT_ROOT}/latest_global_step.txt" ]; then
  STEP=$(cat "${CHECKPOINT_ROOT}/latest_global_step.txt")
  LOAD_CKPT="${CHECKPOINT_ROOT}/global_step_${STEP}"
  if [ -d "${LOAD_CKPT}" ]; then
    echo "Resume from checkpoint: ${LOAD_CKPT}"
    RESUME_ARGS="trainer.load_checkpoint_path=${LOAD_CKPT}"
  fi
fi

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/root/workspace/datasets/hm_data_converted/train.jsonl \
    data.val_files=/root/workspace/datasets/hm_data_converted/test.jsonl \
    data.rollout_batch_size=16 \
    data.train_num_workers=0 \
    data.val_num_workers=0 \
    data.val_batch_size=4 \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=16 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.padding_free=false \
    worker.actor.ulysses_sequence_parallel_size=1 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.ref.fsdp.enable_cpu_offload=false \
    worker.rollout.n=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_model_len=6144 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1gui \
    trainer.experiment_name=qwen3_vl_4b_hm_data_grpo \
    trainer.save_checkpoint_path=${CHECKPOINT_ROOT} \
    trainer.save_freq=10 \
    trainer.save_limit=3 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    data.max_pixels=1258291 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    ${RESUME_ARGS}
