set -x

# 1) Convert hm_data to train/test jsonl
python scripts/convert_hm_data.py \
  --input_dir /root/workspace/datasets/hm_data/hm_data/ \
  --output_dir /root/workspace/datasets/hm_data_converted \
  --test_ratio 0.05 \
  --seed 42

# 2) Run training with conservative memory settings
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct  # replace with your local model path if needed
SYSTEM_PROMPT=""""""
N_GPUS=${N_GPUS:-4}

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/root/workspace/datasets/hm_data_converted/train.jsonl \
    data.val_files=/root/workspace/datasets/hm_data_converted/test.jsonl \
    data.rollout_batch_size=16 \
    data.val_batch_size=4 \
    data.train_num_workers=0 \
    data.val_num_workers=0 \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=16 \
    worker.actor.padding_free=false \
    worker.actor.ulysses_sequence_parallel_size=1 \
    worker.rollout.n=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1gui \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.ref.fsdp.enable_cpu_offload=false \
    trainer.experiment_name=qwen3_vl_8b_hm_data_grpo_debug_safe \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    data.max_pixels=1258291 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024
