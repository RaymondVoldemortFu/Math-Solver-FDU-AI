import re
from typing import Optional


SYSTEM_PROMPT = ("你是一个数学解题助手，负责解答数学问题。你不与用户交互，只需给出答案和有限的解题过程。答案必须以正确格式呈现。"
                 "最终答案的格式必须为<answer>example_numbers</answer>，且其他地方不允许出现类似的格式。"
                 "答案必须是整数、小数、分数或者百分数。答案必须是直接的结果，每道题必须有答案。"
                )


def extract_answer(response):
    """
    从LLM输出文本中提取<answer>example_numbers</answer>格式的答案
    匹配规则：在<think></think>标签外的正文中的最后一个<answer>example_numbers</answer>

    参数:
        response (str): LLM生成的回答文本

    返回:
        str: 提取出的答案内容，只保留数字、分数线/和小数点.

    异常:
        ValueError: 如果没有找到答案或答案格式不正确
    """
    # 移除所有空格和换行符
    response = re.sub(r'[\s\n]+', '', response)

    # 首先移除<think>...</think>中的内容
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # 在清理后的文本中查找所有<answer>与</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, clean_response)

    # 如果找不到匹配项，抛出异常
    if not matches:
        raise ValueError("未找到标准格式的答案，应为<answer>example_numbers</answer>格式")

    # 获取最后一个匹配到的答案内容
    raw_answer = matches[-1]

    # 过滤答案中的所有中英文字符，只保留数字、分数线/和小数点.
    # 使用正则表达式保留数字、/和.，移除其他所有字符
    filtered_answer = re.sub(r'[^\d/.%-]', '', raw_answer)

    # 如果过滤后的答案为空，抛出异常
    if not filtered_answer:
        raise ValueError("答案格式不正确，过滤后没有有效数字")

    return filtered_answer


def extract_chains_of_thought(self, text: str) -> Optional[str]:
    """
    从文本中提取<chains_of_thought>标签之间的内容

    Args:
        text: 输入的文本内容

    Returns:
        提取的内容，如果没有匹配则返回None
    """
    pattern = r'<chains_of_thought>(.*?)</chains_of_thought>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

