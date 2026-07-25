"""
配置器：把 prompts/*.yaml 模板与 config/games/*.yaml 游戏配置组装成
可直接喂给 LLM 的完整 system prompt。

模板里的占位符用 {game_name}、{topic_definitions} 标记，
用 _render() 做安全替换（不用 str.format，避免与提示词中字面量 JSON 的
大括号 {"sentiment": ...} 冲突）。
"""
import yaml
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).parent.parent  # endfield/
PROMPTS_DIR = ROOT / "prompts"
GAMES_DIR = ROOT / "config" / "games"


def _render(template: str, **fields) -> str:
    """安全占位符替换：只替换 {key}，忽略提示词里其它字面量大括号。"""
    for key, value in fields.items():
        template = template.replace("{" + key + "}", str(value))
    return template


@lru_cache(maxsize=None)
def load_prompt(prompt_name: str) -> dict:
    """读取 prompt 模板 (prompts/<prompt_name>.yaml)。"""
    path = PROMPTS_DIR / f"{prompt_name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def load_game_config(game_key: str) -> dict:
    """读取游戏配置 (config/games/<game_key>.yaml)。"""
    path = GAMES_DIR / f"{game_key}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_examples(examples: list[dict]) -> str:
    """把 examples 列表拼成 '示例:' 文本块。"""
    if not examples:
        return ""
    lines = ["\n\n示例:"]
    for ex in examples:
        lines.append(f"输入: {ex['input']}")
        lines.append(f"输出: {ex['output']}")
    return "\n".join(lines)


def build_sentiment_system(game_key: str) -> str:
    """组装情感分类的完整 system prompt。"""
    prompt = load_prompt("sentiment")
    game = load_game_config(game_key)
    system = _render(prompt["system"], game_name=game["game_name"])
    return system + _format_examples(prompt.get("examples", []))


def build_topics_system(game_key: str) -> str:
    """组装议题归类的完整 system prompt。"""
    prompt = load_prompt("topics")
    game = load_game_config(game_key)
    topic_defs = "\n".join(
        f"- {t['name']}: {t['description']}" for t in game["topics"]
    )
    system = _render(
        prompt["system"],
        game_name=game["game_name"],
        topic_definitions=topic_defs,
    )
    return system + _format_examples(prompt.get("examples", []))


if __name__ == "__main__":
    # 自检：打印两款游戏各自组装后的 system prompt
    for key in ("endfield", "genshin"):
        print("=" * 60)
        print(f"[{key}] SENTIMENT")
        print("=" * 60)
        print(build_sentiment_system(key))
        print("=" * 60)
        print(f"[{key}] TOPICS")
        print("=" * 60)
        print(build_topics_system(key))
