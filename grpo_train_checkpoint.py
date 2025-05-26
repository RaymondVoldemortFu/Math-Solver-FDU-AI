import os
import logging
import torch
import json
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from utils.utils import *
from grpo_train import reward_fn
import swanlab
import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SwanLab登录
swanlab.login(api_key=config.swanlab_key)

# 初始化SwanLab
swanlab.init(
    project="qwen-math-solver",
    experiment_name="grpo-training-resume",
    config={"model": "Qwen3-0.6n"},
    mode="cloud"
)

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {device}")

# 加载数据集
logger.info("载入数据集")
dataset_path = "augmented_data/train_augmented.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)
dataset = Dataset.from_list(data)

# 构造 prompt
def build_prompt(example):
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{example['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n")

train_dataset = dataset.map(lambda x: {"prompt": build_prompt(x)})

# 定义reward函数（可复用之前的reward_fn）
def reward_fn(prompts, completions, **kwargs):
    # 省略具体实现，直接复用之前的逻辑
    pass

# 配置训练参数
output_dir = "./output/qwen-math-grpo"
checkpoint_dir = "./output/qwen-math-grpo/checkpoint-last"  # 检查点路径
model_path = r"pretrained_models/qwen3"

training_args = GRPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=200,
    learning_rate=1e-5,
    num_train_epochs=2,
    report_to="none",
)

# 初始化训练器
logger.info("初始化 GRPO 训练器")
trainer = GRPOTrainer(
    model=model_path,
    args=training_args,
    reward_funcs=[reward_fn],
    train_dataset=train_dataset,
)

# 从checkpoint恢复训练
if os.path.exists(checkpoint_dir):
    logger.info(f"从检查点 {checkpoint_dir} 恢复训练")
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    logger.error(f"检查点 {checkpoint_dir} 不存在，无法恢复训练")

# 保存模型
logger.info(f"保存模型到 {output_dir}")
trainer.save_model(output_dir)

# 完成训练
swanlab.finish()
logger.info("训练完成")