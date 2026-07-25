import pandas as pd
import os
import sys
sys.path.insert(0, '..')

df = pd.read_csv("data/taptap_reviews_cleaned.csv")

# 抽 50 条 (用不同 random_state 避开昨天 sentiment 用过的100条)
eval_topics = df.sample(50, random_state=99).reset_index(drop=True)

# 只保留关键字段
eval_topics = eval_topics[['review_id', 'score', 'content', 'created_date', 'version']]

# 加空列
eval_topics['label_topics'] = ''       # 人工标注(多标签用|分隔)
eval_topics['label_primary'] = ''      # 人工判定的主议题
eval_topics['pred_topics'] = ''        # LLM预测(list被转成字符串)
eval_topics['pred_primary'] = ''       # LLM预测的主议题
eval_topics['pred_confidence'] = 0.0
eval_topics['pred_reason'] = ''

eval_topics.to_csv("eval/eval_topics.csv", index=False, encoding='utf-8-sig')
print(f"✅ 已存 50 条")
print(f"版本分布:\n{eval_topics['version'].value_counts()}")
print(f"评分分布:\n{eval_topics['score'].value_counts().sort_index()}")