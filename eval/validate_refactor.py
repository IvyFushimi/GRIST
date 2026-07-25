"""
抽样验证：用重构后的 classify() 重跑已标注样本，对比人工标签算准确率。
目的：一次同时确认 (1)config_loader 重构没跑坏链路 (2)分类质量没退化。

用法:
    python eval/validate_refactor.py                 # 默认验证情感，endfield
    python eval/validate_refactor.py --task topics    # 验证议题
    python eval/validate_refactor.py --game genshin   # 换游戏配置
    python eval/validate_refactor.py --n 20           # 只抽前20条快速看
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.llm_client import LLMClient
from agent.sentiment import classify
from agent.topics import classify_topics

_BASE = Path(__file__).parent


def validate_sentiment(df, client, game_key):
    correct = 0
    rows = []
    for _, r in df.iterrows():
        pred = classify(str(r["content"]), client, game_key=game_key)
        p = pred.get("sentiment")
        ok = (p == r["label_true"])
        correct += ok
        rows.append({"review_id": r["review_id"], "label": r["label_true"],
                     "pred": p, "conf": pred.get("confidence"), "ok": ok,
                     "preview": str(r["content"])[:40]})
    return correct, pd.DataFrame(rows)


def validate_topics(df, client, game_key):
    """议题按 primary_topic 是否命中人工标注的 label_primary 算对。"""
    correct = 0
    rows = []
    for _, r in df.iterrows():
        pred = classify_topics(str(r["content"]), client, game_key=game_key)
        p = pred.get("primary_topic")
        ok = (p == r["label_primary"])
        correct += ok
        rows.append({"review_id": r["review_id"], "label": r["label_primary"],
                     "pred": p, "conf": pred.get("confidence"), "ok": ok,
                     "preview": str(r["content"])[:40]})
    return correct, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sentiment", "topics"], default="sentiment")
    ap.add_argument("--game", default="endfield")
    ap.add_argument("--n", type=int, default=None, help="只跑前 N 条")
    args = ap.parse_args()

    if args.task == "sentiment":
        df = pd.read_csv(_BASE / "eval_v2.csv")
        df = df[df["label_true"].notna() & (df["label_true"] != "")]
        runner, label_col = validate_sentiment, "label_true"
    else:
        df = pd.read_csv(_BASE / "topics_eval.csv")
        df = df[df["label_primary"].notna() & (df["label_primary"] != "")]
        runner, label_col = validate_topics, "label_primary"

    if args.n:
        df = df.head(args.n)
    print(f"任务={args.task}  游戏={args.game}  样本数={len(df)}")

    client = LLMClient()
    correct, res = runner(df, client, args.game)
    acc = correct / len(res) if len(res) else 0

    print(f"\n准确率: {correct}/{len(res)} = {acc:.1%}")
    print("\n混淆分布 (label → pred):")
    print(res.groupby(["label", "pred"]).size().to_string())

    mismatches = res[~res["ok"]]
    if len(mismatches):
        print(f"\n错判 {len(mismatches)} 条:")
        print(mismatches[["review_id", "label", "pred", "conf", "preview"]].to_string(index=False))

    out = _BASE / f"validate_{args.task}_{args.game}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n明细已存: {out}")


if __name__ == "__main__":
    main()
