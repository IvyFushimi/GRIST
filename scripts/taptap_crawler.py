# scripts/taptap_crawler.py
import requests
import pandas as pd
import time
import os
from datetime import datetime

BASE_URL = "https://www.taptap.cn/webapiv2/review/v2/list-by-app"

FIXED_PARAMS = {
    "app_id": "232326",
    "filter_platform": "",
    "label": "",
    "mapping": "",
    "session_id": "9cda92ce-03fe-4404-9584-75f4c85e67d6",
    "sort": "hot",
    "source_type": "",
    "stage_type": "2",
    "X-UA": "V=1&PN=WebApp&LANG=zh_CN&VN_CODE=102&LOC=CN&PLT=PC&DS=PC&UID=f0fe3bf5-8831-4f2d-9fac-c39eb0bc8482&OS=Windows&OSV=10&DT=PC",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.taptap.cn/app/232326/review",
    "Accept": "application/json",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def fetch_page(from_offset=0, limit=10):
    params = {**FIXED_PARAMS, "from": from_offset, "limit": limit}
    try:
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
        print(f"  [from={from_offset}] status={resp.status_code}")
        if resp.status_code != 200:
            print(f"  ⚠️ 异常响应前300字: {resp.text[:300]}")
            return None
        return resp.json()
    except Exception as e:
        print(f"  ❌ fetch err: {e}")
        return None


def parse_reviews(json_data):
    """从data.list里提取评论"""
    if not json_data or not json_data.get("success"):
        return []
    
    items = json_data.get("data", {}).get("list", [])
    rows = []
    
    for item in items:
        try:
            moment = item.get("moment") or {}
            review = moment.get("review") or {}
            author_user = (moment.get("author") or {}).get("user") or {}
            stat = moment.get("stat") or {}
            contents = review.get("contents") or {}
            
            # 跳过非review类型(可能有视频/动态混在list里)
            if not review:
                continue
            
            row = {
                "review_id": review.get("id"),
                "score": review.get("score"),
                "content": contents.get("text", "") or contents.get("raw_text", ""),
                "author_id": author_user.get("id"),
                "author_name": author_user.get("name", ""),
                "device": moment.get("device", ""),
                "created_time": moment.get("created_time"),
                "created_date": datetime.fromtimestamp(moment.get("created_time", 0)).strftime("%Y-%m-%d") if moment.get("created_time") else "",
                "edited_time": moment.get("edited_time"),
                "played_spent_sec": review.get("played_spent", 0),
                "played_spent_hour": round(review.get("played_spent", 0) / 3600, 1),
                "stage_label": review.get("stage_label", ""),
                "ups": stat.get("ups", 0),
                "downs": stat.get("downs", 0),
                "comments_count": stat.get("comments", 0),
                "pv_total": stat.get("pv_total", 0),
            }
            rows.append(row)
        except Exception as e:
            print(f"  parse err: {e}")
            continue
    
    return rows


def sample_run(n_pages=5, output_name="taptap_sample.csv"):
    """跑n_pages页，目标拿n_pages*10条样本"""
    all_rows = []
    
    for p in range(n_pages):
        print(f"\n📡 抓第 {p+1}/{n_pages} 页 (from={p*10})...")
        data = fetch_page(from_offset=p * 10, limit=10)
        rows = parse_reviews(data)
        
        if not rows:
            print(f"  ⚠️ 本页空，停止")
            break
        
        print(f"  ✅ 拿到 {len(rows)} 条")
        all_rows.extend(rows)
        time.sleep(2)
    
    if not all_rows:
        print("\n❌ 一条都没抓到，检查session_id/X-UA是否过期")
        return None
    
    df = pd.DataFrame(all_rows).drop_duplicates("review_id")
    
    os.makedirs("data", exist_ok=True)
    output_path = f"data/{output_name}"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n{'='*50}")
    print(f"✅ 完成，共 {len(df)} 条")
    print(f"💾 已存 {output_path}")
    print(f"\n📊 评分分布:")
    print(df["score"].value_counts().sort_index())
    print(f"\n⏰ 时间跨度: {df['created_date'].min()} → {df['created_date'].max()}")
    print(f"🎮 游戏时长(小时) 中位数: {df['played_spent_hour'].median()}h")
    print(f"\n📝 前3条预览:")
    for i, row in df.head(3).iterrows():
        print(f"\n  [{row['score']}星 | {row['played_spent_hour']}h | {row['author_name']}]")
        print(f"  {(row['content'] or '')[:100]}...")
    
    return df


def full_run(max_pages=100, output_name="taptap_reviews_full.csv"):
    """批量爬，最多max_pages页"""
    all_rows = []
    for p in range(max_pages):
        print(f"\n📡 第 {p+1}/{max_pages} 页...")
        data = fetch_page(from_offset=p * 10, limit=10)
        rows = parse_reviews(data)
        if not rows:
            print(f"  到底了，共 {len(all_rows)} 条")
            break
        all_rows.extend(rows)
        time.sleep(2)
    
    df = pd.DataFrame(all_rows).drop_duplicates("review_id")
    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/{output_name}", index=False, encoding="utf-8-sig")
    print(f"\n✅ 完成，共 {len(df)} 条 → data/{output_name}")
    return df

def full_run_multi_sort(max_pages=100):
    all_rows = []
    for sort in ["hot", "new", "spent"]:
        print(f"\n===== sort={sort} =====")
        FIXED_PARAMS["sort"] = sort
        
        for p in range(max_pages):
            data = fetch_page(from_offset=p*10, limit=10)
            rows = parse_reviews(data)
            if not rows:
                print(f"  到底了")
                break
            all_rows.extend(rows)
            time.sleep(3)
    
    df = pd.DataFrame(all_rows).drop_duplicates("review_id")
    df.to_csv("data/taptap_reviews_v2.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 去重后共 {len(df)} 条")


if __name__ == "__main__":
    # 第一步：先跑5页拿50条样本
    #sample_run(n_pages=5)
    
    # 验证OK后，把上面注释掉，启用下面批量
    #full_run(max_pages=100)
    
    full_run_multi_sort(max_pages=100)