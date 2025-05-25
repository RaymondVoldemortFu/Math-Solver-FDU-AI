import json
import os
import time
from config import api_key as API_KEY
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_request.llm_requestor import LLMRequestor


class DataAugmenter:
    """使用LLM为训练数据添加思维链或其他增强字段的工具"""

    def __init__(self, llm_requestor: LLMRequestor, output_dir: str = "augmented_data",
                 augment_field: str = "chain_of_thought", system_prompt: str = None,
                 max_workers: int = 5, save_interval: int = 50, max_retries: int = 40,
                 retry_delay: float = 1.0):
        """
        初始化数据增强器

        Args:
            llm_requestor: LLM请求器实例
            output_dir: 增强数据输出目录
            augment_field: 要添加的字段名称
            system_prompt: 发送给LLM的系统提示词
            max_workers: 最大并行工作线程数
            save_interval: 每处理多少条数据保存一次中间结果
            max_retries: 单条记录请求失败时的最大重试次数
            retry_delay: 重试之间的延迟时间（秒）
        """
        self.llm_requestor = llm_requestor
        self.output_dir = output_dir
        self.augment_field = augment_field
        self.max_workers = max_workers
        self.save_interval = save_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置默认系统提示词
        self.system_prompt = system_prompt or (
            "你是一个专业的思维链生成助手。请为给定的问题和答案生成详细的思考过程，"
            "说明从问题到答案的推理步骤。思维链应该简洁准确，并展示解决问题的完整思考路径。"
            "过程不易过长，清晰即可"
            "请遵循以下格式：\n"
            "<chains_of_thought>content</chains_of_thought>\n"
        )

        self.logger = llm_requestor.logger

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载训练数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"成功加载数据文件: {file_path}, 共 {len(data)} 条记录")
            return data
        except Exception as e:
            self.logger.error(f"加载数据文件失败: {str(e)}")
            raise

    def save_data(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """保存增强后的数据"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, ensure_ascii=False, indent=2, fp=f)
        self.logger.info(f"数据已保存至: {file_path}")

    def generate_enhancement(self, item: Dict[str, Any]) -> str:
        """为单个数据项生成增强内容"""
        # 构建提示词
        question = item.get('question', '')
        answer = item.get('answer', '')

        # 基本提示词
        base_prompt = f"""请为以下问题生成详细的思维链:

    问题: {question}

    参考答案: {answer}

    请生成从问题到答案的思考过程，展示解题的逻辑步骤。"""

        prompt = base_prompt
        format_attempts = 0
        max_format_attempts = 3  # 最多尝试调整格式的次数

        # 请求LLM并实现重试逻辑
        for attempt in range(self.max_retries):
            try:
                response = self.llm_requestor.ask(
                    question=prompt,
                    system_prompt=self.system_prompt
                )

                # 检查是否包含正确的格式
                content = self.extract_chains_of_thought(response)

                if content is not None:
                    return content  # 返回提取的内容

                # 如果格式不正确且尝试次数未超过限制，调整提示词重新生成
                if format_attempts < max_format_attempts:
                    format_attempts += 1
                    self.logger.warning(
                        f"未检测到正确的思维链格式，调整提示词并重试 ({format_attempts}/{max_format_attempts})")

                    # 调整提示词，强调格式要求
                    prompt = f"""{base_prompt}

    【格式要求】你必须使用下面的格式输出思维链：
    <chains_of_thought>
    你的思维链内容
    </chains_of_thought>

    请注意，思维链必须包含在这些标签之间，这非常重要！"""

                    continue  # 使用新的提示词重试
                else:
                    # 如果多次尝试后仍无法获得正确格式，返回原始响应
                    self.logger.warning(f"多次尝试后仍未获得正确格式的思维链，返回原始响应")
                    return response

            except Exception as e:
                self.logger.warning(f"生成增强内容失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    # 添加指数退避延迟
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.info(f"等待 {delay:.2f} 秒后重试...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"达到最大重试次数 ({self.max_retries})，放弃请求")
                    return f"ERROR: 在 {self.max_retries} 次尝试后增强内容生成失败: {str(e)}"

    def augment_data(self, data: List[Dict[str, Any]],
                     checkpoint_file: str = None) -> List[Dict[str, Any]]:
        """增强数据集中的每一条记录"""
        augmented_data = []
        checkpoint_path = checkpoint_file or os.path.join(self.output_dir, "augment_checkpoint.json")

        # 检查是否有检查点
        start_index = 0
        if os.path.exists(checkpoint_path):
            try:
                augmented_data = self.load_data(checkpoint_path)
                start_index = len(augmented_data)
                self.logger.info(f"从检查点恢复，已处理 {start_index} 条数据")
            except Exception as e:
                self.logger.warning(f"无法加载检查点，将从头开始: {str(e)}")
                augmented_data = []

        # 如果已经完成所有处理，直接返回
        if start_index >= len(data):
            self.logger.info("所有数据已处理完成，无需继续")
            return augmented_data

        # 准备待处理数据
        remaining_data = data[start_index:]

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, item in enumerate(remaining_data):
                future = executor.submit(self.generate_enhancement, item)
                futures[future] = (i, item)

            # 跟踪进度并处理结果
            with tqdm(total=len(remaining_data), desc="增强数据") as pbar:
                completed_count = 0
                for future in as_completed(futures):
                    i, item = futures[future]
                    try:
                        enhancement = future.result()
                        # 创建增强后的数据项
                        augmented_item = item.copy()
                        augmented_item[self.augment_field] = enhancement

                        # 添加到结果集
                        augmented_data.append(augmented_item)

                        # 更新进度
                        completed_count += 1
                        pbar.update(1)

                        # 定期保存检查点
                        if completed_count % self.save_interval == 0:
                            self.save_data(augmented_data, checkpoint_path)
                            self.logger.info(f"已处理 {len(augmented_data)}/{len(data)} 条数据，保存检查点")

                    except Exception as e:
                        self.logger.error(f"处理第 {start_index + i} 条数据时出错: {str(e)}")
                        # 添加错误标记的项
                        augmented_item = item.copy()
                        augmented_item[self.augment_field] = f"ERROR: {str(e)}"
                        augmented_data.append(augmented_item)
                        pbar.update(1)

        return augmented_data

    def process_file(self, input_file: str, output_file: str = None) -> None:
        """处理输入文件并保存增强后的数据"""
        # 设置输出文件名
        if output_file is None:
            file_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(self.output_dir, f"{file_name}_augmented.json")

        # 加载数据
        data = self.load_data(input_file)

        # 增强数据
        self.logger.info(f"开始增强数据，共 {len(data)} 条记录")
        start_time = time.time()
        augmented_data = self.augment_data(data)
        elapsed_time = time.time() - start_time

        # 保存结果
        self.save_data(augmented_data, output_file)

        # 记录统计信息
        self.logger.info(f"数据增强完成!")
        self.logger.info(f"处理了 {len(data)} 条记录，耗时: {elapsed_time:.2f} 秒")
        self.logger.info(f"增强后的数据已保存至: {output_file}")


def main():
    """主函数，用于命令行调用"""
    import argparse
    from dotenv import load_dotenv

    # 加载环境变量
    load_dotenv()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用LLM对训练数据进行增强")
    parser.add_argument("input_file", help="输入的训练数据JSON文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--field", "-f", default="chain_of_thought", help="要添加的字段名称")
    parser.add_argument("--api-key", help="API密钥，如不提供则从环境变量中读取")
    parser.add_argument("--base-url", default="https://www.dmxapi.com/v1", help="API基础URL")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="使用的LLM模型")
    parser.add_argument("--max-workers", type=int, default=5, help="最大并行工作线程数")
    parser.add_argument("--log-dir", default="logs", help="日志目录")

    args = parser.parse_args()

    # 获取API密钥
    api_key = args.api_key or API_KEY
    if not api_key:
        print("错误: 未提供API密钥。请通过--api-key参数或LLM_API_KEY环境变量提供。")
        return

    # 创建LLM请求器
    llm_requestor = LLMRequestor(
        api_key=api_key,
        base_url=args.base_url,
        default_model=args.model,
        log_dir=args.log_dir
    )

    # 创建数据增强器
    augmenter = DataAugmenter(
        llm_requestor=llm_requestor,
        augment_field=args.field,
        max_workers=args.max_workers
    )

    # 处理数据
    augmenter.process_file(args.input_file, args.output)


if __name__ == "__main__":
    main()