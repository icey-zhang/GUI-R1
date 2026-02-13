# GUI-R1（Qwen3-VL 适配版）

本仓库是 GUI-R1 在 `verl` 训练链路上的 Qwen3-VL 适配版本，目标是直接可运行训练与评测（不依赖 `verl_new/`）。

## 主要改动

- 适配 `Qwen3-VL` 的 RoPE 位置编码处理。
- 扩展 tokenizer/processor 流程，支持 `Qwen3VLProcessor`。
- 更新数据处理逻辑，使 Qwen2-VL / Qwen3-VL 共用多模态输入路径。
- 增加 Qwen3-VL 训练脚本：`examples/qwen3_vl_8b_gui_grpo.sh`。
- 更新依赖版本（`transformers>=4.57.0`，`vllm>=0.8.5`）。

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 数据准备

将 GUI-R1 数据集放到如下目录（示例）：

```text
datasets/GUI-R1/train.parquet
datasets/GUI-R1/test.parquet
```

## 训练（Qwen3-VL）

```bash
bash examples/qwen3_vl_8b_gui_grpo.sh
```

如需使用本地模型路径，修改脚本中的 `MODEL_PATH`。

## 推理与评测

```bash
cd guir1
bash inference.sh
bash eval.sh
```

## 说明

- 当前仓库已包含可运行主代码；`verl_new/` 仅作为对照目录，不参与运行。
- 若需进一步适配 Qwen3-VL-MoE 或更高版本 vLLM，可在此基础上继续扩展。
