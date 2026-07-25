# TapTap 情感分析 Prompt 迭代记录 · 7/18

## 数据集
- 源数据：taptap_reviews_full.csv（TapTap 热门排序，共 ~1000 条）
- Eval v1：errors_v1.csv（人工标注 27 条错误样本，label_true vs pred 不一致）
- 标签体系：positive / negative / neutral 三分类
- 人工标注维度：score（星级）+ content（评论正文）→ label_true

## 迭代记录

### Round 1：baseline
**prompt 版本**：v1（仅基础格式约束 + 2 条 few-shot）

**prompt 核心内容**：
- 三分类任务，禁止输出多余字段
- 2 条 few-shot（正面 + 反讽负面各一）

**错误分析来源**：errors_v1.csv（27 条，人工核查）

**错误分布**：

| 真实标签 → 预测标签 | 数量 | 占比 |
|---|---|---|
| neutral → negative | 14 | 52% |
| neutral → positive | 7 | 26% |
| positive → negative | 5 | 19% |
| negative → positive | 1 | 4% |

**错误模式归类**（27 条）：
- 负面关键词过敏（混合评论被判 negative）：14（52%）← 最大弱项
- 正面词主导（中性评论被判 positive）：7（26%）
- 长篇吐槽 + 最终推荐（positive 被判 negative）：5（19%）
- 建议体差评（negative 被判 positive）：1（4%）

**核心问题**：
- 模型按关键词计票，忽略玩家最终立场
- neutral 无显式定义，模型默认回避中性标签
- few-shot 只有 2 条，建议体 / 混合情感均无示例

---

### Round 2：加判断逻辑 + 陷阱规则 + few-shot 扩充
**prompt 版本**：v2（见 sentiment.py `SENT_SYSTEM`，注释 `# Round 2 版本`）

**修改内容**：

1. **两步判断法**：要求先定"最终立场"，再选标签，而非逐句统计情感词
2. **neutral 显式定义**："有好有坏、总体接受但保留态度"，打破模型逃避中性的倾向
3. **抱怨 vs 建议区分规则**（新增独立章节）：
   - 建议 + 正面上下文 / 表达期待 → neutral 或 positive
   - 建议 + 整体失望 / 无正面内容 → negative（包装抱怨）
4. **陷阱规则**：4 条（长篇吐槽结尾推荐 / 反讽 / 负面关键词≠negative / 3 星多批评≠negative）
5. **few-shot 扩至 6 条**：补充 neutral 示例、建议体对比示例（正面上下文 vs 包装抱怨）

**待评估**：Round 2 准确率、各类别精确率（计划用 eval_v2 或重跑 eval_v1 对比）

---

## 结论（当前）
- Round 1 核心弱项：neutral 识别（模型偏向极端标签）
- Round 2 针对性修复了 neutral 定义模糊 + 建议体误判两类主要问题
- 改进效果待 eval_v2 验证

## 定版
sentiment.py 当前采用 Round 2 版本，待 eval_v2 结果出来后决定是否进入 Step 3 批量跑

## 未来优化方向
- 评估 Round 2 在 eval_v1 上的错误数变化（对比基准）
- 加长度阈值分支：< 15 字的极短评论单独处理
- 尝试 CoT prompting：让模型输出"最终立场理由"后再给标签（可提升 neutral 准确率）
- 反讽检测：考虑加一个 `is_sarcasm` 中间字段辅助判断，最后不输出
