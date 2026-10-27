# scripts/make_sample_data.py — 生成主表 CSV（含渠道字段，数据分布更贴近真实业务）
import os
import random
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
random.seed(42)
np.random.seed(42)

regions = ["华东", "华南", "华北", "西南", "东北"]
cats = ["家电", "数码", "服饰", "食品", "美妆"]
channels = ["线上", "线下"]
rows = []
for _ in range(300):
    quantity = random.randint(1, 50)
    sales = round(np.random.gamma(2, 500), 2)
    rows.append({
        "region": random.choice(regions),
        "category": random.choice(cats),
        "sales": sales,
        "quantity": quantity,
        "unit_price": round(sales / quantity, 2),  # 单价 = 销售额 / 数量，业务一致
        "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=random.randint(0, 364)),
        "channel": random.choice(channels),
    })

df = pd.DataFrame(rows)
df.loc[df.sample(10, random_state=1).index, "sales"] = np.nan
df.loc[df.sample(3, random_state=2).index, "sales"] = 99999  # 注入异常值
df.to_csv("data/sample_sales.csv", index=False, encoding="utf-8-sig")
print(f"✅ 已生成 data/sample_sales.csv，{len(df)} 行 × {len(df.columns)} 列")
