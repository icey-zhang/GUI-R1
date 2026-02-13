# GUI-R1（Qwen3-VL 适配 + hm_data 训练）

本仓库已适配 `Qwen3-VL`，并支持将 Open-AutoGLM 的 `hm_data` 转换为 GUI-R1/verl 可训练格式。

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 新数据集转换（hm_data -> jsonl）

```bash
python scripts/convert_hm_data.py \
  --input_dir /Users/zhangjiaqing/Documents/agent/Open-AutoGLM/hm_data \
  --output_dir /Users/zhangjiaqing/Documents/agent/datasets/hm_data_converted \
  --test_ratio 0.05 \
  --seed 42
```

输出：
- `train.jsonl`
- `test.jsonl`

字段包含：
- `instruction`
- `history`
- `task_type`
- `image`
- `gt_bbox`
- `gt_action`
- `gt_input_text`

## 训练（Qwen3-VL）

8B：

```bash
bash examples/qwen3_vl_8b_hm_data_grpo.sh
```

4B：

```bash
bash examples/qwen3_vl_4b_hm_data_grpo.sh
```

说明：
- 训练脚本里已设置 `worker.rollout.tensor_parallel_size=1`，避免 Qwen3-VL 在 vLLM rollout 下的 TP 报错。
- 如使用本地模型，修改脚本中的 `MODEL_PATH` 即可。
