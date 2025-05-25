import requests
import json
import time
import logging
import os
import uuid
import re
from typing import Dict, List, Union, Optional, Any
from datetime import datetime


class LLMRequestor:
    """大语言模型请求器，管理和控制向大模型提问、处理回复并记录日志"""

    def __init__(self, api_key: str, base_url: str = "https://www.dmxapi.com/v1",
                 default_model: str = "gpt-3.5-turbo", max_retries: int = 3,
                 retry_delay: float = 1.0, log_dir: Optional[str] = None):
        """
        初始化LLM请求器

        Args:
            api_key: API密钥
            base_url: API基础URL
            default_model: 默认使用的模型名称
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间(秒)
            log_dir: 日志目录路径，None表示不记录到文件
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_request_time = 0

        # 设置日志记录
        self.logger = self._setup_logger(log_dir)
        self.logger.info("LLMRequestor 初始化完成")

    def _setup_logger(self, log_dir: Optional[str]) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger("LLMRequestor")
        logger.setLevel(logging.INFO)

        # 清除现有的处理器
        logger.handlers = []

        # 添加控制台处理器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 如果指定了日志目录，添加文件处理器
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f"llm_requests_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            'Accept': 'application/json',
            'Authorization': f'{self.api_key}',
            'User-Agent': 'DMXAPI/1.0.0 (https://www.dmxapi.com)',
            'Content-Type': 'application/json'
        }

    def request(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                temperature: float = 0.7, max_tokens: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        发送请求给LLM模型

        Args:
            messages: 消息列表
            model: 使用的模型，如果为None则使用默认模型
            temperature: 温度参数
            max_tokens: 最大生成的token数
            **kwargs: 其他参数

        Returns:
            模型的响应内容
        """
        model = model or self.default_model
        url = f"{self.base_url}/chat/completions"

        # 构建请求参数
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # 添加其他参数
        payload.update(kwargs)

        # 记录请求信息
        req_id = int(time.time() * 1000)
        self.logger.info(f"请求ID: {req_id}, 模型: {model}")

        # 发送请求，如果失败则重试
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    url,
                    headers=self._get_headers(),
                    data=json.dumps(payload)
                )
                elapsed_time = time.time() - start_time

                if response.status_code == 200:
                    response_json = response.json()
                    self.logger.info(f"请求ID: {req_id} 成功, 耗时: {elapsed_time:.2f}秒")
                    return response_json
                else:
                    self.logger.warning(
                        f"请求ID: {req_id} 失败, 状态码: {response.status_code}, 尝试: {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        self.logger.error(f"请求ID: {req_id} 最终失败, 响应: {response.text}")
                        raise Exception(f"请求失败, 状态码: {response.status_code}, 响应: {response.text}")
            except Exception as e:
                self.logger.error(f"请求ID: {req_id} 发生异常: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def ask(self, question: str, system_prompt: str = "You are a helpful assistant.",
            model: Optional[str] = None, **kwargs) -> str:
        """
        向模型提问并获取回答

        Args:
            question: 用户问题
            system_prompt: 系统提示
            model: 使用的模型
            **kwargs: 其他参数

        Returns:
            模型的回答内容
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        response = self.request(messages, model=model, **kwargs)
        return response['choices'][0]['message']['content']

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        进行多轮对话

        Args:
            messages: 消息列表
            model: 使用的模型
            **kwargs: 其他参数

        Returns:
            模型的完整响应
        """
        return self.request(messages, model=model, **kwargs)




