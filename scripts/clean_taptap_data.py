"""
TapTap 评论数据清洗
- 输入: data/taptap_reviews_v2.csv
- 输出: data/taptap_reviews_cleaned.csv
- 步骤: 去重 → 时间标准化 → 空评论过滤 → 极短过滤 → 字段规范
"""
import pandas as pd
import os
from datetime import datetime

INPUT = "data/taptap_reviews_v2.csv"
OUTPUT = "data/taptap_reviews_cleaned.csv"
STATS_LOG = "cleaning_report.log"

def clean():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始清洗")
    
    # ===== 读入 =====
    df = pd.read_csv(INPUT)
    n_original = len(df)
    print(f"\n📥 原始数据: {n_original} 条")
    print(f"字段: {list(df.columns)}")
    
    stats = {'原始': n_original}
    
    # ===== Step 1: 去重 =====
    before = len(df)
    df = df.drop_duplicates(subset=['review_id'], keep='first')
    dropped = before - len(df)
    print(f"\n🔄 去重 (review_id): -{dropped} 条 → {len(df)}")
    stats['去重后'] = len(df)
    
    # ===== Step 2: 时间字段标准化 =====
    # 原始 created_time 是 Unix 秒数，转成 date 和 datetime
    df['created_time'] = pd.to_numeric(df['created_time'], errors='coerce')
    df = df[df['created_time'].notna()]  # 无效时间戳的删掉
    df['created_datetime'] = pd.to_datetime(df['created_time'], unit='s')
    df['created_date'] = df['created_datetime'].dt.date.astype(str)
    df['created_month'] = df['created_datetime'].dt.strftime('%Y-%m')
    df['created_hour'] = df['created_datetime'].dt.hour
    
    print(f"\n⏰ 时间字段标准化完成")
    print(f"时间跨度: {df['created_date'].min()} → {df['created_date'].max()}")
    stats['时间标准化后'] = len(df)
    
    # ===== Step 3: 空评论过滤 =====
    before = len(df)
    df['content'] = df['content'].fillna('').astype(str).str.strip()
    df = df[df['content'] != '']
    dropped = before - len(df)
    print(f"\n📝 空评论过滤: -{dropped} 条 → {len(df)}")
    stats['非空评论'] = len(df)
    
    # ===== Step 4: 极短评论过滤 =====
    df['content_length'] = df['content'].str.len()
    before = len(df)
    df = df[df['content_length'] >= 5]  # <5字符的过滤
    dropped = before - len(df)
    print(f"\n✂️ 极短评论过滤(<5字符): -{dropped} 条 → {len(df)}")
    stats['长度≥5'] = len(df)
    
    # ===== Step 5: score 字段规范 =====
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    before = len(df)
    df = df[df['score'].between(1, 5)]  # 只保留合法1-5分
    dropped = before - len(df)
    if dropped > 0:
        print(f"\n⭐ score 规范(1-5): -{dropped} 条 → {len(df)}")
    stats['score合法'] = len(df)
    
    # ===== Step 6: 加版本标签 =====
    # 关键版本节点(基于你之前 endfield_timeline.xlsx 的数据)
    def assign_version(date_str):
        d = pd.Timestamp(date_str)
        if d < pd.Timestamp('2025-11-28'): return 'pre-测试'
        elif d < pd.Timestamp('2026-01-22'): return '全面测试期'
        elif d < pd.Timestamp('2026-03-12'): return '1.0公测'
        elif d < pd.Timestamp('2026-04-17'): return '1.1新潮起'
        elif d < pd.Timestamp('2026-06-05'): return '1.2春晓时'
        else: return '1.3寻遗散记'
    
    df['version'] = df['created_date'].apply(assign_version)
    print(f"\n🏷️ 版本标签已加")
    print(df['version'].value_counts().sort_index())
    
    # ===== Step 7: 字段整理 =====
    # 保留有用列
    keep_cols = [
        'review_id', 'score', 'content', 'content_length',
        'author_name', 'device', 
        'created_date', 'created_datetime', 'created_month', 'created_hour',
        'version',
        'ups', 'downs', 'comments_count', 'pv_total',
        'played_spent_sec', 'played_spent_hour', 'stage_label',
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]  # 只保留存在的
    df = df[keep_cols].reset_index(drop=True)
    
    # ===== 输出 =====
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*50}")
    print(f"✅ 清洗完成: {n_original} → {len(df)} 条 (保留率 {len(df)/n_original*100:.1f}%)")
    print(f"💾 输出: {OUTPUT}")
    
    # ===== 清洗报告 =====
    with open(STATS_LOG, 'w', encoding='utf-8') as f:
        f.write(f"清洗时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入: {INPUT}\n")
        f.write(f"输出: {OUTPUT}\n\n")
        f.write("=== 数据流失清单 ===\n")
        for step, count in stats.items():
            f.write(f"{step}: {count}\n")
        f.write(f"\n最终保留: {len(df)} ({len(df)/n_original*100:.1f}%)\n")
        f.write(f"\n=== 版本分布 ===\n")
        for v, c in df['version'].value_counts().sort_index().items():
            f.write(f"{v}: {c}\n")
    
    print(f"📋 清洗报告: {STATS_LOG}")
    return df

if __name__ == "__main__":
    df = clean()