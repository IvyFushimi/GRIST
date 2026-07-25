"""
情感分类模块（纯流程逻辑，无游戏专属内容）
system prompt 由 config_loader 按 game_key 动态组装。
"""
from .llm_client import LLMClient
from .config_loader import build_sentiment_system


def classify(text: str, client: LLMClient, game_key: str = "endfield") -> dict:
    """对单条评论做情感分类。system 由 game_key 决定，内容截断在 client 内处理。"""
    system = build_sentiment_system(game_key)
    return client.call_json(user_prompt=text, system_prompt=system)
