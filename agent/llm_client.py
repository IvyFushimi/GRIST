from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# .env 可能是 UTF-8，也可能是 PowerShell `echo > .env` 生成的 UTF-16。
# 逐个编码尝试：UTF-16 文件以 0xff/0xfe BOM 开头，用 utf-8 读会抛 UnicodeDecodeError，
# 必须 try/except 捕获后回退，不能靠 `if not load_dotenv()`（异常会直接冒泡而非返回 False）。
for _enc in ("utf-8", "utf-16"):
    try:
        if load_dotenv(encoding=_enc):
            break
    except UnicodeDecodeError:
        continue

_USER_MSG_LIMIT = 500

# 默认指向 DeepSeek，但只是「默认」——任何 OpenAI 兼容服务商都可通过 .env 覆盖。
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-v4-flash"


def _first_env(*names: str) -> str | None:
    """按优先级返回第一个非空环境变量值。"""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


class LLMClient:
    """任意 OpenAI 兼容 LLM 服务的轻封装。

    配置优先级：显式构造参数 > 环境变量 > 内置默认值。
    只需在 .env 里设置以下变量即可切换服务商，无需改代码：
        LLM_API_KEY   = sk-...                         # 密钥（也兼容 OPENAI_API_KEY / DEEPSEEK_API_KEY）
        LLM_BASE_URL  = https://api.openai.com/v1      # 服务地址（也兼容 OPENAI_BASE_URL）
        LLM_MODEL     = gpt-4o-mini                     # 模型名

    例：
        用 OpenAI：LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o-mini
        用 Moonshot：LLM_BASE_URL=https://api.moonshot.cn/v1  LLM_MODEL=moonshot-v1-8k
        用本地 Ollama：LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        system_prompt: str = "你是游戏行业的资深数据分析师。",
    ):
        api_key = api_key or _first_env("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
        base_url = base_url or _first_env("LLM_BASE_URL", "OPENAI_BASE_URL") or _DEFAULT_BASE_URL
        self.model = model or _first_env("LLM_MODEL") or _DEFAULT_MODEL
        self.system_prompt = system_prompt

        if not api_key:
            raise ValueError(
                "未找到 API Key：请在 .env 设置 LLM_API_KEY（或兼容的 "
                "OPENAI_API_KEY / DEEPSEEK_API_KEY），或构造时显式传入 api_key。"
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)

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
