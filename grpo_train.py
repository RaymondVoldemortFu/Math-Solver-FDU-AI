from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from utils.utils import SYSTEM_PROMPT
import re
import json

# 1. 加载句向量模型（适用于中文）
model = SentenceTransformer("BAAI/bge-large-zh")

# 2. 载入数据集
dataset = load_dataset("json", data_files=r"augmented_data/train_augmented.json")["train"]


# 3. 构造 prompt：instruction + question
def build_prompt(example):
    return (f"<system>{SYSTEM_PROMPT}</system>\n"
            f"<questions>{example['question']}</questions>\n")


train_json_new_path = "train.json"

dataset = dataset.map(lambda x: {"prompt": build_prompt(x)})

# 4. 从文本中抽取预测答案
def extract_answer(text):
    numbers = re.findall(r"\d+", text.strip())
    return numbers[-1] if numbers else ""

# 5. 计算语义相似度
def semantic_similarity(a: str, b: str) -> float:
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))


# 6. 构造 reward_fn：用于 GRPOTrainer
def reward_fn(prompts, responses):
    rewards = []

    for prompt, response in zip(prompts, responses):
        # 获取与 prompt 匹配的原始数据（可优化为字典查询）
        question = prompt.split('\n')[-1].strip()
        matches = dataset.filter(lambda x: x['question'] == question)

        if len(matches) == 0:
            rewards.append(0.0)
            continue

        ref = matches[0]
        expected_cot = ref["chain_of_thought"]
        expected_ans = str(ref["answer"])

        # 相似度（思维链）
        cot_sim = semantic_similarity(response, expected_cot)

        # 答案是否匹配
        pred_ans = extract_answer(response)
        answer_match = 1.0 if pred_ans == expected_ans else 0.0

        # 最终 reward（可调整权重）
        reward = 0.7 * cot_sim + 0.3 * answer_match
        rewards.append(reward)

    return rewards
