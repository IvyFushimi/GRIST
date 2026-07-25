import pandas as pd
import sys
sys.path.insert(0, '..')

eval_df = pd.read_csv("eval/topics_eval.csv")

# 过滤有效数据
valid = eval_df[
    (eval_df['label_topics'].notna()) & 
    (eval_df['label_topics'] != '') &
    (eval_df['pred_topics'].notna()) &
    (eval_df['pred_topics'] != 'error')
].copy()

print(f"有效样本: {len(valid)}/{len(eval_df)}")

# 转换成集合方便交集运算
def to_set(s):
    if pd.isna(s) or s == '':
        return set()
    return set(str(s).split('|'))

valid['label_set'] = valid['label_topics'].apply(to_set)
valid['pred_set'] = valid['pred_topics'].apply(to_set)

# 指标1: 覆盖率
valid['is_covered'] = valid.apply(
    lambda r: len(r['label_set'] & r['pred_set']) > 0, 
    axis=1
)
coverage = valid['is_covered'].mean()
print(f"\n★ 覆盖率: {coverage:.1%}")

# 指标2: 一致性(primary)
valid['primary_match'] = valid['pred_primary'] == valid['label_primary']
consistency = valid['primary_match'].mean()
print(f"★ Primary 一致性: {consistency:.1%}")

# 分议题精确率
print(f"\n分议题一致率:")
for topic in ['玩法', '商业化', '技术', '角色', '运营', '其他']:
    subset = valid[valid['label_primary'] == topic]
    if len(subset) >= 3:
        acc = (subset['pred_primary'] == topic).mean()
        print(f"  {topic}: {acc:.1%} (n={len(subset)})")

# 混淆矩阵(primary)
from sklearn.metrics import confusion_matrix
topics = ['玩法', '商业化', '技术', '角色', '运营', '其他']
cm = confusion_matrix(valid['label_primary'], valid['pred_primary'], labels=topics)
cm_df = pd.DataFrame(cm, index=[f'true_{t}' for t in topics], columns=[f'pred_{t}' for t in topics])
print(f"\nPrimary 混淆矩阵:\n{cm_df}")

# 错误样本清单
errors = valid[~valid['primary_match']][
    ['score', 'content', 'label_topics', 'label_primary', 'pred_topics', 'pred_primary', 'pred_reason']
]
print(f"\n错误样本 (n={len(errors)}):")
errors.to_csv("eval/topics_errors.csv", index=False, encoding='utf-8-sig')
print("已存 eval/topics_errors.csv")