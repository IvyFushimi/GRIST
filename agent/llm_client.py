from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# 优先按 UTF-8 读取 .env；若本机的 .env 是 PowerShell `echo > .env` 生成的 UTF-16，
# 再回退一次，避免读出乱码导致 API Key 取不到。
if not load_dotenv():
    load_dotenv(encoding="utf-16")

_USER_MSG_LIMIT = 500


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        system_prompt: str = "你是游戏行业的资深数据分析师。",
    ):
        self.model = model
        self.system_prompt = system_prompt
        self._client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )

    def call(self, user_prompt: str, temperature: float = 0.7) -> str:
        """普通文本调用，user 消息截断到 500 字符。"""
        prompt = user_prompt[:_USER_MSG_LIMIT]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def call_json(self, user_prompt: str, system_prompt: str | None = None) -> dict:
        """返回 JSON 的调用，temperature=0.1，user 消息截断到 500 字符。
        system_prompt 若传入则覆盖 self.system_prompt。"""
        prompt = user_prompt[:_USER_MSG_LIMIT]
        # DeepSeek json_object 模式要求 prompt 中必须出现 "json"
        system = system_prompt or self.system_prompt
        if "json" not in system.lower():
            system = system + "\n请以JSON格式输出。"
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
