import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from utils.utils import SYSTEM_PROMPT, extract_answer, extract_thinking
from transformers import logging as transformers_logging
import pandas as pd
import re
import json
import logging
import os
import swanlab

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_info()

# 初始化SwanLab
swanlab.init(
    project="qwen-math-solver",
    experiment_name="grpo-training",
    config={"model": "Qwen3-0.6n"},
    mode="local"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {device}")
# 1. 加载句向量模型（适用于中文）
model_name = r"pretrained_models/bge-large-zh"
logger.info(f"加载句向量模型: {model_name}")
sentence_model = SentenceTransformer(model_name_or_path=model_name)
sentence_model.to(device=device)
# 2. 载入数据集并预处理
logger.info("载入数据集")
dataset_path = "augmented_data/train_augmented.json"

try:
    # 读取JSON数据
    with open("augmented_data/train_augmented.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 数据预处理，确保所有字段都是简单类型
    processed_data = []
    for item in data:
        # 确保每个问题是字符串类型
        if isinstance(item.get("question"), list):
            item["question"] = " ".join(item["question"])  # 或其他合适的处理方式
        processed_data.append(item)

    # 直接从处理后的数据创建Dataset
    dataset = Dataset.from_list(processed_data)
    print(f"成功创建数据集，包含 {len(dataset)} 个样本")

except Exception as e:
    print(f"创建数据集出错：{e}")
    raise e

# 创建问题到答案和思维链的映射，提高查询效率
question_map = {}
for item in dataset:
    question_map[item["question"]] = {
        "answer": str(item["answer"]),
        "chain_of_thought": item.get("chain_of_thought", "")
    }


# 3. 构造 prompt：instruction + question
def build_prompt(example):
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{example['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n")


# 4. 准备数据集
logger.info("准备训练数据集")
train_dataset = dataset.map(lambda x: {"prompt": build_prompt(x)})


# 5. 计算语义相似度
def semantic_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        emb1 = sentence_model.encode(a, convert_to_tensor=True)
        emb2 = sentence_model.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))
    except Exception as e:
        logger.error(f"计算语义相似度出错: {str(e)}")
        return 0.0


# 6. 构造 reward 函数
def reward_fn(prompts, responses):
    rewards = []

    for prompt, response in zip(prompts, responses):
        try:
            # 从prompt中提取问题
            match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt)
            if not match:
                logger.error(f"无法从提示中提取问题{prompt}")
                rewards.append(0.0)
                continue

            question = match.group(1).strip()

            # 使用预先构建的映射查找参考答案
            if question not in question_map:
                rewards.append(0.0)
                logger.error(f"问题 {question} 不在参考数据集中")
                continue

            ref_data = question_map[question]
            expected_cot = ref_data["chain_of_thought"]
            expected_ans = ref_data["answer"]

            # 从回答中提取思维链和答案
            try:
                # 提取思维链
                cot = extract_thinking(response) or response

                # 提取答案
                pred_ans = extract_answer(response)

                # 计算思维链相似度
                cot_sim = semantic_similarity(cot, expected_cot)

                # 答案匹配度 (完全匹配或部分匹配)
                if pred_ans == expected_ans:
                    answer_match = 1.0
                elif expected_ans in pred_ans or pred_ans in expected_ans:
                    answer_match = 0.5
                else:
                    answer_match = 0.0

                # 综合奖励值（可调整权重）
                reward = 0.6 * cot_sim + 0.4 * answer_match

                # 记录评价结果
                if len(rewards) % 10 == 0:
                    logger.info(f"问题: {question[:50]}...")
                    logger.info(f"思维链相似度: {cot_sim:.2f}, 答案匹配: {answer_match}")
                    logger.info(f"总奖励: {reward:.2f}")

                rewards.append(reward)

            except Exception as e:
                logger.error(f"处理回答时出错: {str(e)}")
                rewards.append(0.0)

        except Exception as e:
            logger.error(f"奖励计算出错: {str(e)}")
            rewards.append(0.0)

    return rewards


# 7. 配置训练参数
output_dir = "./output/qwen-math-grpo"
os.makedirs(output_dir, exist_ok=True)
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
    max_steps=2000,
    # use_vllm=True,
    # vllm_server_host="127.0.0.1",
    report_to="none",
)

# 8. 初始化训练器并训练
logger.info("初始化 GRPO 训练器")
trainer = GRPOTrainer(
    model=model_path,
    args=training_args,
    reward_funcs=[reward_fn],
    train_dataset=train_dataset,
)

logger.info("开始训练")
trainer.train()

# 保存模型
logger.info(f"保存模型到 {output_dir}")
trainer.save_model(output_dir)

# 完成训练
swanlab.finish()
logger.info("训练完成")
