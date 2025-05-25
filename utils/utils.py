import re


SYSTEM_PROMPT = ("你是一个数学解题助手，负责解答数学问题。你不与用户交互，只需给出答案和推理过程。答案必须以正确格式呈现。"
                 "最终答案的格式必须为<answer>example_numbers</answer>，且其他地方不允许出现类似的格式。"
                 "答案必须是整数、小数、分数或者百分数。"
                 "/nothink")


def extract_answer(response):
    """
    从LLM输出文本中提取<answer>example_numbers</answer>格式的答案
    匹配规则：在<think></think>标签外的正文中的最后一个<answer>example_numbers</answer>

    参数:
        response (str): LLM生成的回答文本

    返回:
        str: 提取出的答案内容

    异常:
        ValueError: 如果没有找到答案或答案格式不正确
    """
    # 首先移除<think>...</think>中的内容
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # 在清理后的文本中查找所有<answer>与</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, clean_response)

    # 如果找不到匹配项，抛出异常
    if not matches:
        raise ValueError("未找到标准格式的答案，应为<answer>example_numbers</answer>格式")

    # 返回最后一个匹配到的答案内容
    return matches[-1]
