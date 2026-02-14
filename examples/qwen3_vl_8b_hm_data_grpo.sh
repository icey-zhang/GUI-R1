set -x

# 1) Convert hm_data to train/test jsonl
python scripts/convert_hm_data.py \
  --input_dir /Users/zhangjiaqing/Documents/agent/Open-AutoGLM/hm_data \
  --output_dir /Users/zhangjiaqing/Documents/agent/datasets/hm_data_converted \
  --test_ratio 0.05 \
  --seed 42

# 2) Run training
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct  # replace with your local model path if needed
SYSTEM_PROMPT=""""""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/Users/zhangjiaqing/Documents/agent/datasets/hm_data_converted/train.jsonl \
    data.val_files=/Users/zhangjiaqing/Documents/agent/datasets/hm_data_converted/test.jsonl \
    data.rollout_batch_size=64 \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1gui \
    trainer.experiment_name=qwen3_vl_8b_hm_data_grpo \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    data.max_pixels=1258291 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.val_batch_size=16
