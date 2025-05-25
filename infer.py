import json
import torch
import logging
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils.utils import SYSTEM_PROMPT, extract_answer
from datetime import datetime


def setup_logger():
    """设置日志记录器"""
    # 创建logs目录（如果不存在）
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # 创建带有时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/infer_log_{timestamp}.log"

    # 配置日志记录器
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)

    return logger


def batch_predict(test_data, model, tokenizer, batch_size=4, use_system_prompt=True, max_retries=20):
    """批量预测函数，支持重试逻辑"""
    results = {}
    error_cases = {}  # 用于记录重试后仍然无法提取答案的情况
    retry_counts = {}  # 记录每个样本的重试次数
    pending_items = test_data.copy()  # 待处理的样本列表

    # 创建总体进度条，初始值为样本总数
    total_items = len(test_data)
    progress_bar = tqdm(total=total_items, desc="预测进度")
    processed_count = 0

    # 循环直到没有待处理项或全部达到重试上限
    while pending_items:
        batch_to_process = []  # 当前需要处理的批次
        batch_messages = []
        batch_ids = []
        batch_retry_counts = []  # 记录当前批次中每个样本的重试次数

        # 从待处理项中构建当前批次，最多处理batch_size个
        for i, item in enumerate(pending_items[:batch_size]):
            item_id = item['id']
            retry_count = retry_counts.get(item_id, 0)

            batch_to_process.append(item)
            batch_retry_counts.append(retry_count)

            # 根据重试次数调整系统提示词
            system_prompt = SYSTEM_PROMPT
            if retry_count > 0:
                system_prompt += (f"\n这是第{retry_count + 1}次尝试，请务必以<answer>数字</answer>格式给出答案，不要使用其他格式。"
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

        # 移除已经选择处理的项，避免重复处理
        pending_items = pending_items[len(batch_to_process):]

        # 如果批次为空，结束处理
        if not batch_to_process:
            break

        # print("batch_messages:", batch_messages)

        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                 for msg in batch_messages]

        # 批量编码和推理
        with torch.no_grad():
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024
            )

        # 存储需要重试的项
        items_to_retry = []
        newly_completed = 0

        for j, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            item_id = batch_ids[j]
            retry_count = batch_retry_counts[j]
            generated_part = output_ids[len(input_ids):]
            response = tokenizer.decode(generated_part, skip_special_tokens=True)
            # print(f"ID: {item_id} - 生成的回答: {response}")
            # 尝试提取标准格式的答案
            try:
                answer = extract_answer(response)
                # 只有首次成功才计入进度
                if item_id not in results:
                    newly_completed += 1
                results[item_id] = answer  # 成功提取答案，保存并不再重试
                if retry_count > 0:
                    print(f"ID: {item_id} - 在第{retry_count + 1}次尝试后成功获取答案")
            except ValueError as e:
                # 如果当前重试次数小于最大重试次数，添加到重试列表
                if retry_count < max_retries - 1:
                    retry_counts[item_id] = retry_count + 1
                    # 将原始项添加到重试列表
                    items_to_retry.append(batch_to_process[j])
                    print(f"ID: {item_id} - 提取答案失败，将进行第{retry_count + 2}次尝试: {e}")
                else:
                    # 达到最大重试次数，保存原始响应
                    error_cases[item_id] = str(e)
                    results[item_id] = response.replace('\n', ' ')
                    # 只有首次达到最大重试次数才计入进度
                    if item_id not in results:
                        newly_completed += 1
                    print(f"ID: {item_id} - 达到最大重试次数({max_retries})，无法提取标准答案")

        # 更新进度条
        processed_count += newly_completed
        progress_bar.update(newly_completed)
        progress_bar.set_postfix({"已完成": processed_count, "待处理": len(pending_items), "重试": len(items_to_retry)})

        # 将需要重试的项添加到待处理列表的前面，确保优先处理
        pending_items = items_to_retry + pending_items

        # 释放内存
        del model_inputs, generated_ids, texts
        torch.cuda.empty_cache()

    # 关闭进度条
    progress_bar.close()

    # 输出错误统计
    if error_cases:
        print(f"共有 {len(error_cases)} 条数据在{max_retries}次尝试后仍无法提取标准答案格式")

    return results


# 主函数
test_json_new_path = "test.json"

# 加载测试数据
print("正在加载测试数据...")
with open(test_json_new_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)
print(f"加载完成，共 {len(test_data)} 条测试数据")

# 加载模型和分词器
print("正在加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/qwen3", use_fast=False, trust_remote_code=True)
# 设置左侧填充以解决警告
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained("./pretrained_models/qwen3", device_map="auto", torch_dtype=torch.bfloat16)
# model = PeftModel.from_pretrained(model, model_id="./output/Qwen3/checkpoint-3750/")
model.eval()  # 设置为评估模式
print("模型加载完成")

# 批量预测
print("开始批量预测...")
batch_size = 8  # 可根据GPU内存调整
results = batch_predict(test_data, model, tokenizer, batch_size)

# 写入结果，保持原始顺序
print("写入结果到CSV文件...")
with open("submit.csv", 'w', encoding='utf-8') as file:
    for item in tqdm(test_data, desc="写入进度"):
        id_value = item['id']
        response = results[id_value]
        file.write(f"{id_value},{response}\n")

print("处理完成!")

