import json
import torch
import logging
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
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


def batch_predict(test_data, model, tokenizer, batch_size=4, use_system_prompt=True, max_retries=20, logger=None):
    """批量预测函数，支持重试逻辑"""
    results = {}
    error_cases = {}
    retry_counts = {}
    pending_items = test_data.copy()

    total_items = len(test_data)
    progress_bar = tqdm(total=total_items, desc="预测进度")
    processed_count = 0

    while pending_items:
        batch_to_process = []
        batch_messages = []
        batch_ids = []
        batch_retry_counts = []

        for i, item in enumerate(pending_items[:batch_size]):
            item_id = item['id']
            retry_count = retry_counts.get(item_id, 0)

            batch_to_process.append(item)
            batch_retry_counts.append(retry_count)

            system_prompt = SYSTEM_PROMPT
            if retry_count > 0:
                system_prompt += (
                    f"\n这是第{retry_count + 1}次尝试，请务必以<answer>数字</answer>格式给出答案，不要使用其他格式。"
                    f"推理的时候减少过多的思考，以给出答案为重")

            if use_system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item['question']}
                ]
            else:
                messages = [
                    {"role": "system", "content": item['instruction']},
                    {"role": "user", "content": item['question']}
                ]
            batch_messages.append(messages)
            batch_ids.append(item_id)

        pending_items = pending_items[len(batch_to_process):]

        if not batch_to_process:
            break

        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                 for msg in batch_messages]

        with torch.no_grad():
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024
            )

        items_to_retry = []
        newly_completed = 0

        for j, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            item_id = batch_ids[j]
            retry_count = batch_retry_counts[j]
            generated_part = output_ids[len(input_ids):]
            response = tokenizer.decode(generated_part, skip_special_tokens=True)

            if logger:
                question = batch_to_process[j]['question']
                attempt_info = f"第{retry_count + 1}次尝试" if retry_count > 0 else "首次尝试"
                logger.info(f"ID: {item_id} - {attempt_info}")
                logger.info(f"问题: {question}")
                logger.info(f"回答: {response}")

            try:
                answer = extract_answer(response)
                if item_id not in results:
                    newly_completed += 1
                results[item_id] = answer

                if logger:
                    logger.info(f"成功提取答案: {answer}")
                    if retry_count > 0:
                        logger.info(f"在第{retry_count + 1}次尝试后成功获取答案")
                    logger.info("-" * 50)

            except ValueError as e:
                if retry_count < max_retries - 1:
                    retry_counts[item_id] = retry_count + 1
                    items_to_retry.append(batch_to_process[j])

                    if logger:
                        logger.warning(f"提取答案失败: {e}")
                        logger.info(f"将进行第{retry_count + 2}次尝试")
                        logger.info("-" * 50)

                else:
                    error_cases[item_id] = str(e)
                    results[item_id] = response.replace('\n', ' ')
                    if item_id not in results:
                        newly_completed += 1

                    if logger:
                        logger.error(f"达到最大重试次数({max_retries})，无法提取标准答案")
                        logger.error(f"错误原因: {e}")
                        logger.info("-" * 50)

        processed_count += newly_completed
        progress_bar.update(newly_completed)
        progress_bar.set_postfix({"已完成": processed_count, "待处理": len(pending_items), "重试": len(items_to_retry)})

        if logger:
            logger.info(f"批次处理完成，当前进度: {processed_count}/{total_items}，"
                        f"待处理: {len(pending_items)}，需重试: {len(items_to_retry)}")

        pending_items = items_to_retry + pending_items

        del model_inputs, generated_ids, texts
        torch.cuda.empty_cache()

    progress_bar.close()

    if error_cases and logger:
        logger.warning(f"共有 {len(error_cases)} 条数据在{max_retries}次尝试后仍无法提取标准答案格式")

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

    logger.info("开始批量预测...")
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
