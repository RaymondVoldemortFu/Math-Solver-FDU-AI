import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def interact_with_model(model_path, lora_path=None, system_prompt=None, response_format=None):
    """
    与模型交互的命令行工具

    Args:
        model_path: 模型路径
        lora_path: LoRA 模型路径（可选）
        system_prompt: 系统提示词
        response_format: 响应格式说明
    """
    # 设置默认系统提示词
    if system_prompt is None:
        system_prompt = "你是一个有用的AI助手。"

    # 设置默认响应格式
    if response_format is None:
        response_format = "请提供简明、有帮助的回答。"

    print(f"正在加载模型 {model_path}...")
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.padding_side = 'left'  # 设置左侧填充

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 如果提供了LoRA路径，加载LoRA权重
    if lora_path:
        print(f"加载LoRA权重 {lora_path}...")
        model = PeftModel.from_pretrained(model, model_id=lora_path)

    model.eval()
    print("模型加载完成！")

    # 显示当前设置
    print("\n===== 当前设置 =====")
    print(f"系统提示词: {system_prompt}")
    print(f"响应格式: {response_format}")
    print("===================\n")

    print("输入 'exit' 或 'quit' 结束对话")

    # 交互循环
    while True:
        # 获取用户输入
        user_input = input("\n用户输入 >>> ")
        if user_input.lower() in ['exit', 'quit']:
            print("对话结束")
            break

        # 构建完整提示
        messages = [
            {"role": "system", "content": f"{system_prompt}\n{response_format}"},
            {"role": "user", "content": user_input}
        ]

        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print("\n生成回答中...")

        # 获取生成的回答
        with torch.no_grad():
            model_input = tokenizer(text, return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                model_input.input_ids,
                max_new_tokens=99999,
                temperature=0.7,
                top_p=0.9
            )

            # 提取生成部分
            generated_part = generated_ids[0, len(model_input.input_ids[0]):]
            response = tokenizer.decode(generated_part, skip_special_tokens=True)

        print(f"\n助手回答 >>> {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="与语言模型交互的命令行工具")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--lora_path", type=str, help="LoRA模型路径（可选）")
    parser.add_argument("--system_prompt", type=str, help="系统提示词（可选）")
    parser.add_argument("--response_format", type=str, help="响应格式（可选）")


    args = parser.parse_args()

    interact_with_model(
        args.model_path,
        args.lora_path,
        args.system_prompt,
        args.response_format
    )
