"""
批量跑 TapTap 评论议题分析
- 输入: taptap_reviews_cleaned.csv
- 输出: topics_results.csv
- 特性: 断点续跑 + 增量落盘 + 错误日志
"""
import sys
import os
import time
import logging
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, '.')
from agent.llm_client import LLMClient
from agent.topics import classify_topics

# ============ 配置区 ============
INPUT_CSV = "data/taptap_reviews_cleaned.csv"
OUTPUT_CSV = "topics_results.csv"
ERROR_LOG = "topics_errors.log"

BATCH_SAVE_SIZE = 50      # 每50条落盘一次
SLEEP_BETWEEN_CALLS = 0.5 # 每次调用后sleep秒数(限流)
MAX_ROWS = None           # None=跑全量, 或改成 20 测试

# ============ 日志设置 ============
logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)

# ============ 断点续跑函数 ============
def load_already_processed():
    """读取已处理的 review_id，用于跳过"""
    if not os.path.exists(OUTPUT_CSV):
        return set()
    try:
        done = pd.read_csv(OUTPUT_CSV)
        return set(done['review_id'].tolist())
    except Exception as e:
        print(f"⚠️ 读取已处理文件失败: {e}")
        return set()

# ============ 增量落盘函数 ============
def append_results(rows, output_path):
    """把rows追加到CSV，首次写入带header"""
    df_new = pd.DataFrame(rows)
    if not os.path.exists(output_path):
        df_new.to_csv(output_path, index=False, encoding='utf-8-sig')
    else:
        df_new.to_csv(output_path, mode='a', header=False,
                      index=False, encoding='utf-8-sig')

# ============ 主流程 ============
def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始批量跑议题分析")
    print(f"输入: {INPUT_CSV}")
    print(f"输出: {OUTPUT_CSV}")
    print(f"错误日志: {ERROR_LOG}")

    # 读输入
    df = pd.read_csv(INPUT_CSV)
    print(f"总评论数: {len(df)}")

    # 断点续跑：过滤已处理
    processed_ids = load_already_processed()
    df_todo = df[~df['review_id'].isin(processed_ids)]
    print(f"已处理: {len(processed_ids)}, 待处理: {len(df_todo)}")

    if MAX_ROWS:
        df_todo = df_todo.head(MAX_ROWS)
        print(f"⚠️ 测试模式: 只跑前 {MAX_ROWS} 条")

    if len(df_todo) == 0:
        print("✅ 全部处理完毕，无待处理数据")
        return

    # 初始化
    client = LLMClient()
    buffer = []           # 缓冲区
    success_count = 0
    error_count = 0

    # 主循环
    try:
        pbar = tqdm(df_todo.iterrows(), total=len(df_todo),
                    desc="议题分析中")
        for i, row in pbar:
            review_id = row['review_id']
            content = row.get('content', '')

            # 空评论跳过
            if not content or pd.isna(content) or len(str(content).strip()) < 3:
                logging.warning(f"review_id={review_id} 内容为空/过短，跳过")
                error_count += 1
                continue

            try:
                result = classify_topics(str(content), client)

                # topics 是 list，用 | 拼接方便存 CSV
                topics_list = result.get('topics', [])
                if not isinstance(topics_list, list):
                    topics_list = [str(topics_list)]

                # 组织输出行
                row_out = {
                    'review_id': review_id,
                    'score': row.get('score'),
                    'created_date': row.get('created_date'),
                    'version': row.get('version', ''),
                    'content_preview': str(content)[:100],
                    'topics': '|'.join(topics_list),
                    'primary_topic': result.get('primary_topic', ''),
                    'confidence': result.get('confidence', 0),
                    'reason': result.get('reason', ''),
                    'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
                buffer.append(row_out)
                success_count += 1

                # 每 BATCH_SAVE_SIZE 条落盘一次
                if len(buffer) >= BATCH_SAVE_SIZE:
                    append_results(buffer, OUTPUT_CSV)
                    pbar.set_postfix({
                        'saved': success_count,
                        'errors': error_count,
                        'buffer': 0
                    })
                    buffer = []

                time.sleep(SLEEP_BETWEEN_CALLS)

            except KeyboardInterrupt:
                print("\n⚠️ 收到中断信号，保存缓冲区...")
                raise

            except Exception as e:
                logging.error(f"review_id={review_id} | error: {e} | content={str(content)[:100]}")
                error_count += 1
                continue

    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C")

    finally:
        # 无论成功/失败/中断，把缓冲区剩余的落盘
        if buffer:
            print(f"落盘缓冲区剩余 {len(buffer)} 条...")
            append_results(buffer, OUTPUT_CSV)

        print(f"\n{'='*50}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 结束")
        print(f"✅ 成功: {success_count}")
        print(f"❌ 错误: {error_count}")
        print(f"输出文件: {OUTPUT_CSV}")
        print(f"错误日志: {ERROR_LOG}")


if __name__ == "__main__":
    main()