import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils.utils import SYSTEM_PROMPT, extract_answer


def batch_predict(test_data, model, tokenizer, batch_size=4, use_system_prompt=True, max_retries=20):
    """批量预测函数，支持重试逻辑"""
    results = {}
    error_cases = {}  # 用于记录重试后仍然无法提取答案的情况
    retry_counts = {}  # 记录每个样本的重试次数

    # 添加进度条显示批处理进度
    for i in tqdm(range(0, len(test_data), batch_size), desc="批量预测进度"):
        batch = test_data[i:i + batch_size]
        batch_to_process = []  # 当前需要处理的批次
        batch_messages = []
        batch_ids = []
        batch_retry_counts = []  # 记录当前批次中每个样本的重试次数

        # 准备当前批次
        for item in batch:
            item_id = item['id']
            retry_count = retry_counts.get(item_id, 0)

            # 跳过已经成功提取答案或达到最大重试次数的样本
            if item_id in results or retry_count >= max_retries:
                continue

            batch_to_process.append(item)
            batch_retry_counts.append(retry_count)

            # 根据重试次数调整系统提示词
            system_prompt = SYSTEM_PROMPT
            if retry_count > 0:
                system_prompt += f"\n这是第{retry_count + 1}次尝试，请务必以<answer>数字</answer>格式给出答案，不要使用其他格式。"

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

        # 如果没有需要处理的样本，跳过当前批次
        if not batch_to_process:
            continue

        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                 for msg in batch_messages]

        # 批量编码和推理
        with torch.no_grad():
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=4096
            )

        for j, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            item_id = batch_ids[j]
            retry_count = batch_retry_counts[j]
            generated_part = output_ids[len(input_ids):]
            response = tokenizer.decode(generated_part, skip_special_tokens=True)

            # 尝试提取标准格式的答案
            try:
                answer = extract_answer(response)
                results[item_id] = answer  # 成功提取答案，保存并不再重试
                if item_id in retry_counts and retry_count > 0:
                    print(f"ID: {item_id} - 在第{retry_count + 1}次尝试后成功获取答案")
            except ValueError as e:
                # 如果当前重试次数小于最大重试次数，计划进行下一次重试
                if retry_count < max_retries - 1:
                    retry_counts[item_id] = retry_count + 1
                    print(f"ID: {item_id} - 提取答案失败，将进行第{retry_count + 2}次尝试: {e}")
                else:
                    # 达到最大重试次数，保存原始响应
                    error_cases[item_id] = str(e)
                    results[item_id] = response.replace('\n', ' ')
                    print(f"ID: {item_id} - 达到最大重试次数({max_retries})，无法提取标准答案")

        # 释放内存
        del model_inputs, generated_ids, texts
        torch.cuda.empty_cache()

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

