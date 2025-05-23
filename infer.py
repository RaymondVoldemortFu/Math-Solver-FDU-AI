import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def batch_predict(test_data, model, tokenizer, batch_size=4):
    """批量预测函数"""
    results = {}

    # 添加进度条显示批处理进度
    for i in tqdm(range(0, len(test_data), batch_size), desc="批量预测进度"):
        batch = test_data[i:i + batch_size]
        batch_messages = []
        batch_ids = []

        for item in batch:
            messages = [
                {"role": "system", "content": item['instruction']},
                {"role": "user", "content": item['question']}
            ]
            batch_messages.append(messages)
            batch_ids.append(item['id'])

        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                 for msg in batch_messages]

        # 批量编码和推理
        with torch.no_grad():
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024
            )

        for j, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            generated_part = output_ids[len(input_ids):]
            response = tokenizer.decode(generated_part, skip_special_tokens=True)
            results[batch_ids[j]] = response.replace('\n', ' ')

        # 释放内存
        del model_inputs, generated_ids, texts
        torch.cuda.empty_cache()

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
model = PeftModel.from_pretrained(model, model_id="./output/Qwen3/checkpoint-3750/")
model.eval()  # 设置为评估模式
print("模型加载完成")

# 批量预测
print("开始批量预测...")
batch_size = 4  # 可根据GPU内存调整
results = batch_predict(test_data, model, tokenizer, batch_size)

# 写入结果，保持原始顺序
print("写入结果到CSV文件...")
with open("submit.csv", 'w', encoding='utf-8') as file:
    for item in tqdm(test_data, desc="写入进度"):
        id_value = item['id']
        response = results[id_value]
        file.write(f"{id_value},{response}\n")

print("处理完成!")

