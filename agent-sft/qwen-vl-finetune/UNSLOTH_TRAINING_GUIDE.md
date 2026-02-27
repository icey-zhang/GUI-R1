# 使用 Unsloth 加速训练 Qwen3-VL 模型

本指南说明如何使用 Unsloth 来加速 Qwen3-VL 模型的训练。

## 概述

Unsloth 是一个优化库，可以显著加速大语言模型的训练，相比标准的 HuggingFace Transformers + DeepSpeed 训练：

- **2x 更快的训练速度**
- **70% 更少的显存使用**
- **支持更长的上下文**
- **0% 精度损失**（所有优化都是精确的）

## 文件说明

### 新增文件

1. **`qwenvl/train/train_qwen_unsloth.py`** - 使用 Unsloth 的训练脚本
   - 使用 `FastVisionModel` 加载 Qwen3-VL 模型
   - 自动应用 Unsloth 的优化
   - 支持现有的数据处理器

2. **`scripts/sft_qwen3_2b_unsloth.sh`** - Unsloth 训练启动脚本
   - 配置训练参数
   - 启动训练过程

### 原有文件（保持不变）

- **`qwenvl/data/data_processor.py`** - 数据处理器（无需修改）
- **`qwenvl/train/argument.py`** - 训练参数定义（无需修改）

## 安装依赖

在使用 Unsloth 之前，需要安装相关依赖：

```bash
# 安装 Unsloth
pip install unsloth

# 或者使用特定版本的 PyTorch 和 CUDA
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

## 训练模式说明

### 当前配置

本脚本默认使用 **16-bit 训练**（不使用量化）。

### Unsloth 的优势（即使不使用 4-bit 量化）

即使不使用 4-bit 量化，Unsloth 相比原始的 DeepSpeed 训练仍然有显著优势：

- **2x 更快的训练速度**：优化的内核和内存管理
- **更长的上下文支持**：支持更长的序列长度
- **更好的内存管理**：优化的梯度累积和检查点
- **0% 精度损失**：所有优化都是精确的

### 如果需要使用 4-bit 量化

如果显存有限，可以选择启用 4-bit 量化以节省显存：

在 [`train_qwen_unsloth.py`](qwenvl/train/train_qwen_unsloth.py) 中修改：

```python
# 使用 4-bit 量化
load_in_4bit = True
load_in_16bit = False
```

### 4-bit vs 16-bit 对比

| 特性 | 4-bit 量化 | 16-bit 训练 |
|------|-------------|--------------|
| 显存使用 | ~25% | 100% |
| 训练速度 | 相同或更快 | 相同 |
| 精度损失 | 极小（<1%） | 无 |
| 适用场景 | 显存有限、需要大批次 | 显存充足、追求最高精度 |

## 使用方法

### 1. 基本使用

直接运行训练脚本：

```bash
cd agent-sft/qwen-vl-finetune
bash scripts/sft_qwen3_2b_unsloth.sh
```

### 2. 自定义参数

编辑 `scripts/sft_qwen3_2b_unsloth.sh` 文件中的参数：

```bash
# 模型路径
llm=/path/to/your/model

# 学习率
lr=1e-5

# 批次大小
batch_size=4

# 梯度累积步数
grad_accum_steps=4

# 输出目录
output_dir=./output_unsloth
```

### 3. 多 GPU 训练

**重要：Unsloth 目前不支持多 GPU 训练！**

Unsloth 主要针对单 GPU 训练进行优化。如果需要多 GPU 训练，请使用原始的 DeepSpeed 训练脚本 [`sft_qwen3_2b.sh`](sft_qwen3_2b.sh)。

**Unsloth 单 GPU 训练的优势：**
- 2x 更快的训练速度
- 更长的上下文支持
- 优化的内核和内存管理
- 0% 精度损失

**如果需要多 GPU 训练：**

使用原始的 DeepSpeed 训练脚本：

```bash
bash scripts/sft_qwen3_2b.sh
```

**如果只需要单 GPU 训练：**

使用 Unsloth 训练脚本：

```bash
bash scripts/sft_qwen3_2b_unsloth.sh
```

## 与原始训练的对比

### 原始训练（DeepSpeed）

```bash
# scripts/sft_qwen3_2b.sh
deepspeed ./scripts/zero3.json \
    python qwenvl/train/train_qwen.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    ...
```

### Unsloth 训练

```bash
# scripts/sft_qwen3_2b_unsloth.sh
python qwenvl/train/train_qwen_unsloth.py \
    --model_name_or_path "${llm}" \
    ...
```

主要区别：
1. **不需要 DeepSpeed 配置** - Unsloth 内置优化
2. **不需要 torchrun**（单 GPU）- 直接运行即可
3. **自动 4-bit 量化** - 节省显存
4. **更快的训练速度** - 优化的内核

## LoRA 训练

Unsloth 对 LoRA 训练有特别优化。在训练脚本中启用 LoRA：

```python
# 在 train_qwen_unsloth.py 中
model = FastVisionModel.get_peft_model(
    model,
    r=64,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=40960,
)
```

## 保存和加载模型

### 保存模型

训练完成后，模型会自动保存到 `output_dir` 目录：

```bash
output_unsloth/
├── checkpoint-1000/
├── checkpoint-2000/
└── final_model/
```

### 加载模型进行推理

```python
from unsloth import FastVisionModel
from transformers import AutoProcessor

# 加载训练好的模型
model, tokenizer = FastVisionModel.from_pretrained(
    "./output_unsloth/final_model",
    load_in_4bit=True,
)

# 加载 processor
processor = AutoProcessor.from_pretrained("./output_unsloth/final_model")

# 进行推理
# ...
```

### 合并 LoRA 权重（可选）

如果使用 LoRA 训练，可以合并权重：

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "./output_unsloth/final_model",
    load_in_4bit=True,
)

# 合并 LoRA 权重
model = FastVisionModel.merge_and_unload(model)

# 保存合并后的模型
model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

## 性能优化建议

### 1. 批次大小

Unsloth 支持更大的批次大小，可以尝试：

```bash
batch_size=8  # 原始可能是 4
```

### 2. 梯度累积

减少梯度累积步数，增加批次大小：

```bash
batch_size=8
grad_accum_steps=2  # 原始可能是 4
```

### 3. 序列长度

Unsloth 支持更长的序列：

```bash
# 在训练参数中
model_max_length=81920  # 原始可能是 40960
```

### 4. 混合精度

确保使用 bfloat16（如果 GPU 支持）：

```bash
--bf16  # 在训练参数中
```

## 故障排除

### 1. 显存不足

如果遇到 OOM 错误：

- 减小批次大小：`batch_size=2`
- 启用梯度检查点：`--gradient_checkpointing True`
- 使用 4-bit 量化：`load_in_4bit=True`（默认已启用）

### 2. 导入错误

确保 Unsloth 在其他库之前导入：

```python
# 正确的顺序
from unsloth import FastVisionModel
from transformers import AutoProcessor

# 错误的顺序
from transformers import AutoProcessor
from unsloth import FastVisionModel  # 这会导致警告
```

### 3. 多 GPU 支持

Unsloth 目前主要针对单 GPU 优化。多 GPU 支持仍在开发中。

## 参考资料

- [Unsloth 官方文档](https://unsloth.ai/docs)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Qwen3-VL 模型文档](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

## 许可证

本代码遵循 Apache License 2.0。
