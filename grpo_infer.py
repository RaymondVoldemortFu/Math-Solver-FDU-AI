import json
import torch
import logging
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer
from utils.utils import SYSTEM_PROMPT, extract_answer
from datetime import datetime


def setup_logger():
    """设置日志记录器"""
    if not os.path.exists("logs"):
        os.makedirs("logs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/infer_checkpoint_log_{timestamp}.log"

    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def batch_predict(test_data, model, tokenizer, batch_size=4, logger=None):
    """批量推理函数"""
    results = {}
    progress_bar = tqdm(total=len(test_data), desc="推理进度")

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        batch_prompts = [
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{item['question']}<|im_end|>\n"
            for item in batch
        ]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        for j, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            try:
                answer = extract_answer(response)
                results[batch[j]['id']] = answer
                if logger:
                    logger.info(f"ID: {batch[j]['id']} - 问题: {batch[j]['question']} - 回答: {answer}")
            except ValueError as e:
                results[batch[j]['id']] = "提取答案失败"
                if logger:
                    logger.error(f"ID: {batch[j]['id']} - 提取答案失败: {e}")

        progress_bar.update(len(batch))

    progress_bar.close()
    return results


def main():
    logger = setup_logger()
    logger.info("开始推理过程")

    checkpoint_dir = "./output/qwen-math-grpo/checkpoint-last"
    test_json_path = "test.json"

    if not os.path.exists(checkpoint_dir):
        logger.error(f"指定的checkpoint目录 {checkpoint_dir} 不存在")
        return

    logger.info("加载测试数据...")
    with open(test_json_path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    logger.info(f"加载完成，共 {len(test_data)} 条测试数据")

    logger.info("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained("pretrained_models/qwen3", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    logger.info("模型加载完成")

    logger.info("开始批量推理...")
    batch_size = 4
    results = batch_predict(test_data, model, tokenizer, batch_size, logger=logger)

    logger.info("写入结果到CSV文件...")
    with open("submit_checkpoint.csv", 'w', encoding='utf-8') as file:
        for item in test_data:
            id_value = item['id']
            response = results.get(id_value, "无结果")
            file.write(f"{id_value},{response}\n")

    logger.info("推理完成!")


if __name__ == "__main__":
    main()
