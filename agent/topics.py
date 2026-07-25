"""
议题分类模块（纯流程逻辑，无游戏专属内容）
system prompt 由 config_loader 按 game_key 动态组装。
输出: {'topics': [...], 'primary_topic': '...', 'confidence': 0.x, 'reason': '...'}
"""
from .llm_client import LLMClient
from .config_loader import build_topics_system


def classify_topics(text: str, client: LLMClient, game_key: str = "endfield") -> dict:
    """对单条评论做议题分类。system 由 game_key 决定，内容截断在 client 内处理。"""
    system = build_topics_system(game_key)
    return client.call_json(user_prompt=text, system_prompt=system)
