import math
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import ray
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from datasets import load_dataset
from datasets import Dataset as hf_dataset
from PIL import Image
from io import BytesIO
try:
    from verl.utils.reward_score.r1gui import extract_action, extract_coord, extract_input_text
except ModuleNotFoundError:
    # Support running script directly without installing project as a package.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from verl.utils.reward_score.r1gui import extract_action, extract_coord, extract_input_text
# 初始化 Ray
ray.init()

# 模型路径
MODEL_PATH = ""

# 数据路径
DATA_PATH = ""

class MultiModalDataset(Dataset):
    def __init__(self, data, processor, max_pixels: int = 0):
        self.data = data
        self.processor = processor
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, str):
            image = Image.open(image)
        else:
            raise ValueError("Unsupported image format. Expect {'bytes': ...} or image path.")
        image = image.convert("RGB")
        orig_width, orig_height = image.size
        if self.max_pixels and (orig_width * orig_height > self.max_pixels):
            scale = (self.max_pixels / float(orig_width * orig_height)) ** 0.5
            new_w = max(1, int(orig_width * scale))
            new_h = max(1, int(orig_height * scale))
            image = image.resize((new_w, new_h), Image.BILINEAR)
        text = sample["instruction"]
        history="None" if 'history' not in sample else sample['history']

        # sys_prompt='''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> nd <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'''
        text=(
            f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue "
            f"executing the command '{text}', with the action history being '{history}'.\n"
            "Please output exactly ONE action call using hm_data format.\n"
            "All coordinates must be in 0-1000 relative coordinate system.\n"
            "Output the thinking process in <thinking> </thinking> tags, and the final answer in <answer> </answer> tags as follows:\n"
            "<thinking> ... </thinking> <answer>action(params...)</answer>\n"
            "Available actions and signatures:\n"
            "click(point='x1,y1')\n"
            "long_press(point='x1,y1')\n"
            "type(content='')\n"
            "swipe(start_point='x1,y1', end_point='x2,y2', velocity=600)\n"
            "open_app(app_name='')\n"
            "drag(start_point='x1,y1', end_point='x2,y2')\n"
            "press_home()\n"
            "press_back()\n"
            "wait(t='t')\n"
            "finished(content='')\n"
            "call_user(content='')\n"
            "back_information(content='')"
        )
        text = '<image>\n' + text
        message = [
            # {"role":"system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

        # 生成推理所需的 prompt 和多模态输入
        prompt = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        # prompt.replace("<|vision_start|><|image_pad|><|vision_end|>","")
        # prompt.replace("<image>","<|vision_start|><|image_pad|><|vision_end|>")

        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        # Keep original image size for downstream eval while allowing resized inference.
        sample["image_size"] = [orig_width, orig_height]

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
            "original_sample": sample,
        }


def custom_collate_fn(batch):
    collated_batch = {
        "prompts": [],
        "multi_modal_data": [],
        "mm_processor_kwargs": [],
        "original_samples": [],
    }
    for item in batch:
        collated_batch["prompts"].append(item["prompt"])
        collated_batch["multi_modal_data"].append(item["multi_modal_data"])
        collated_batch["mm_processor_kwargs"].append(item["mm_processor_kwargs"])
        collated_batch["original_samples"].append(item["original_sample"])
    return collated_batch


@ray.remote(num_gpus=1)
class Worker:
    def __init__(
        self,
        model_path,
        sampling_params,
        max_model_len: int,
        gpu_memory_utilization: float,
        debug_print_n: int = 0,
    ):
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1, "video": 1},
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.sampling_params = sampling_params
        self.debug_print_n = max(0, int(debug_print_n))

    def process_data(self, dataloader):
        results = []
        printed = 0

        for batch in tqdm(dataloader):
            prompts = batch["prompts"]
            multi_modal_data = batch["multi_modal_data"]
            mm_processor_kwargs = batch["mm_processor_kwargs"]
            original_samples = batch["original_samples"]

            llm_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": mm_kwargs,
                }
                for prompt, mm_data, mm_kwargs in zip(prompts, multi_modal_data, mm_processor_kwargs)
            ]

            # 执行推理
            outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)

            # 保存结果

            # print(outputs)
            
            for original_sample, output in zip(original_samples, outputs):
                generated_text = output.outputs[0].text
                original_sample["pred"] = generated_text
                pred_coord, _ = extract_coord(generated_text)
                # Keep 0-1000 coordinate output to align with training/action format.
                original_sample["pred_coord"] = [pred_coord[0], pred_coord[1]]
                pred_action = extract_action(generated_text)
                original_sample["pred_action"] = pred_action
                original_sample["pred_input_text"]=extract_input_text(generated_text)

                if printed < self.debug_print_n:
                    print(
                        f"[DEBUG][pid={os.getpid()}] instruction={str(original_sample.get('instruction', ''))[:200]}",
                        flush=True,
                    )
                    print(f"[DEBUG][pid={os.getpid()}] pred={generated_text}", flush=True)
                    print(
                        f"[DEBUG][pid={os.getpid()}] parsed action={pred_action}, coord={original_sample['pred_coord']}, input={original_sample['pred_input_text']}",
                        flush=True,
                    )
                    printed += 1

                original_sample["image"]=''
                results.append(original_sample)

        return results


def main(args):
    # 将数据分成 8 份
    MODEL_PATH=args.model_path
    DATA_PATH=args.data_path
    if DATA_PATH.endswith('parquet'):
        data=load_dataset("parquet", data_files=DATA_PATH, split="train")
    else:
        data = [json.loads(s) for s in open(DATA_PATH, "r")] if DATA_PATH.endswith(".jsonl") else json.load(open(DATA_PATH,"r"))
    # 输出路径
    OUTPUT_DIR = args.output_path
    num_actors = args.num_actor
    OUTPUT_DIR = os.path.join(OUTPUT_DIR,MODEL_PATH.split('/')[-1])
    NEW_FILE = os.path.join(OUTPUT_DIR, DATA_PATH.split("/")[-1].replace(".jsonl", "_pred.jsonl").replace('.parquet','.json'))
    print(NEW_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_chunks = [hf_dataset.from_dict(data[i::num_actors]) for i in range(num_actors)]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens,
        stop_token_ids=[],
    )

    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    # processor.max_pixels=1048576
    # processor.min_pixels=

    # 创建 8 个 Actor，每个 Actor 分配到一个 GPU
    workers = [
        Worker.remote(
            MODEL_PATH,
            sampling_params,
            args.max_model_len,
            args.gpu_memory_utilization,
            math.ceil(args.debug_print_n / max(1, num_actors)),
        )
        for _ in range(num_actors)
    ]

    # 使用 PyTorch Dataset 和 DataLoader
    futures = []
    for i, chunk in enumerate(data_chunks):
        dataset = MultiModalDataset(chunk, processor, max_pixels=args.max_pixels)
        dataloader = DataLoader(
            dataset,
            batch_size=args.micro_batch,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
        )
        futures.append(workers[i].process_data.remote(dataloader))

    # 收集所有结果
    all_results = ray.get(futures)

    # 将结果写入文件
    with open(NEW_FILE, "w") as ans_file:
        for worker_results in all_results:
            for sample in worker_results:
                ans_file.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='<model_path>')
    parser.add_argument('--data_path', type=str, default="<data_path>")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_actor', type=int, default=8)
    parser.add_argument('--micro_batch', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_pixels', type=int, default=458752)  # 512x896
    parser.add_argument('--max_model_len', type=int, default=4096)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.72)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--debug_print_n', type=int, default=0)
    args = parser.parse_args()
    main(args)
