# 用LLM分析一条TapTap评论的情感倾向
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(encoding='utf-16') 

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

review_text = "这游戏一定要暴死啊，评分又涨回去了。500一抽依然是单抽，10抽跟摆设一样。"

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": """你是游戏评论情感分析师。
对玩家评论返回JSON格式：
{
  "sentiment": "positive/negative/neutral",
  "intensity": 1-5,
  "main_complaint": "主要不满点(若有)",
  "keywords": ["关键词1", "关键词2"]
}"""},
        {"role": "user", "content": review_text}
    ],
    temperature=0,  # 分析任务用0，避免随机
    response_format={"type": "json_object"},  # 强制JSON输出
)

import json
result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2, ensure_ascii=False))