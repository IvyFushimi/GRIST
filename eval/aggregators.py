# -*- coding: utf-8 -*-
"""
群体差异假设检验
判断两个群体（如不同版本、不同设备档次）在某议题上的情感分布是否显著不同。

用法:
    from eval.aggregators import load_results, test_group_difference, batch_compare

    df = load_results()                       # 默认读 agent_full_results.csv 并造好 group 列
    r = test_group_difference(df, ["1.0"], ["1.3"], "商业化")
    print(r)

    batch_compare(df, ["1.0"], ["1.3"])       # 对所有议题批量跑

统计口径:
    - 列联表: 群体(2) × 情感(positive/neutral/negative)
    - 显著性: 卡方检验 p 值（p<0.05 认为差异非随机）
    - 效应量: Cramer's V（>0.1 弱, >0.3 中, >0.5 强；不受样本量膨胀影响）
    - 期望频数 <5 时卡方失真，自动降级为 2×2 Fisher 精确检验并标注
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

_BASE = Path(__file__).parent.parent
_DEFAULT_RESULTS = _BASE / "agent_full_results.csv"

SENTIMENTS = ["positive", "neutral", "negative"]
ALL_TOPICS = ["玩法", "商业化", "技术", "角色", "运营", "其他"]


def load_results(path=None, group_from="version") -> pd.DataFrame:
    """读取 sentiment+topics 合并结果，并从 version 抽出 group 列（如 '1.0公测'→'1.0'）。"""
    df = pd.read_csv(path or _DEFAULT_RESULTS)
    if group_from == "version" and "version" in df.columns:
        df["group"] = df["version"].astype(str).str.extract(r"(\d\.\d)")
    return df


def has_topic(cell, topic: str) -> bool:
    """topics 列存的是 '玩法|商业化' 这种 | 分隔字符串，按分隔切开精确匹配（避免子串碰撞）。"""
    if pd.isna(cell):
        return False
    return topic in str(cell).split("|")


def _subset(data, group_vals, topic, group_col):
    return data[
        data[group_col].isin(group_vals)
        & data["topics"].apply(lambda x: has_topic(x, topic))
    ]


def test_group_difference(
    data: pd.DataFrame,
    group_a_vals: list,
    group_b_vals: list,
    topic: str,
    group_col: str = "group",
    min_n: int = 30,
) -> dict:
    """卡方检验：两群体在 topic 议题上的情感分布是否显著不同。

    group_a_vals / group_b_vals: group 列取值的列表，如 ["1.0"]。
    返回 dict；样本不足时只返回 warning。
    """
    a = _subset(data, group_a_vals, topic, group_col)
    b = _subset(data, group_b_vals, topic, group_col)

    # 最小样本量校验
    if len(a) < min_n or len(b) < min_n:
        return {"topic": topic, "warning": f"样本量不足 (a={len(a)}, b={len(b)})，结果不可信"}

    # 2×3 列联表: 群体 × 情感
    contingency = np.array([
        [(a["sentiment"] == s).sum() for s in SENTIMENTS],
        [(b["sentiment"] == s).sum() for s in SENTIMENTS],
    ])

    chi2, p, dof, expected = chi2_contingency(contingency)
    n = contingency.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1))))

    result = {
        "topic": topic,
        "contingency": contingency.tolist(),
        "chi2": round(float(chi2), 3),
        "p_value": float(p),
        "cramers_v": round(cramers_v, 4),
        "effect": _effect_label(cramers_v),
        "sample_a": len(a),
        "sample_b": len(b),
        "neg_rate_a": round(float((a["sentiment"] == "negative").mean()), 4),
        "neg_rate_b": round(float((b["sentiment"] == "negative").mean()), 4),
        "significant": bool(p < 0.05),
        "low_expected_freq": False,
    }

    # 期望频数 <5：卡方失真，降级为 2×2 Fisher（positive vs 非positive）
    if expected.min() < 5:
        result["low_expected_freq"] = True
        collapsed = np.array([
            [contingency[0, 0], contingency[0, 1] + contingency[0, 2]],
            [contingency[1, 0], contingency[1, 1] + contingency[1, 2]],
        ])
        _, p_fisher = fisher_exact(collapsed)
        result["p_value_fisher_2x2"] = float(p_fisher)
        result["significant"] = bool(p_fisher < 0.05)
        result["note"] = "期望频数<5，卡方不可靠，significant 已改用 2×2 Fisher(pos vs 非pos) 判定"

    return result


def _effect_label(v: float) -> str:
    if v < 0.1:
        return "可忽略"
    if v < 0.3:
        return "弱"
    if v < 0.5:
        return "中"
    return "强"


def batch_compare(
    data: pd.DataFrame,
    group_a_vals: list,
    group_b_vals: list,
    topics: list = None,
    group_col: str = "group",
    min_n: int = 30,
) -> pd.DataFrame:
    """对多个议题批量跑检验，返回汇总表（按 cramers_v 降序）。"""
    rows = []
    for t in (topics or ALL_TOPICS):
        r = test_group_difference(data, group_a_vals, group_b_vals, t, group_col, min_n)
        if "warning" in r:
            rows.append({"topic": t, "note": r["warning"]})
        else:
            rows.append({
                "topic": t, "sample_a": r["sample_a"], "sample_b": r["sample_b"],
                "neg_rate_a": r["neg_rate_a"], "neg_rate_b": r["neg_rate_b"],
                "p_value": r["p_value"], "cramers_v": r["cramers_v"],
                "effect": r["effect"], "significant": r["significant"],
                "note": r.get("note", ""),
            })
    out = pd.DataFrame(rows)
    if "cramers_v" in out.columns:
        out = out.sort_values("cramers_v", ascending=False, na_position="last").reset_index(drop=True)
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # 避免 Windows 控制台 GBK 乱码

    df = load_results()
    print("各 group 样本量:", df["group"].value_counts().to_dict())

    print("\n=== Sanity check: 1.0公测 vs 1.3 · 商业化 ===")
    r = test_group_difference(df, ["1.0"], ["1.3"], "商业化")
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== 批量: 1.0 vs 1.3 全议题 ===")
    print(batch_compare(df, ["1.0"], ["1.3"]).to_string(index=False))
