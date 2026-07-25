from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(encoding='utf-16') 

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "你是一个游戏行业的资深分析师。"},
        {"role": "user", "content": "结合联网搜索，用一句话评价《明日方舟：终末地》当前的市场表现。"}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)