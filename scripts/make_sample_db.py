# scripts/make_sample_db.py — 生成三表电商业务库（orders / customers / products）
import sqlite3
import random
import os
import pandas as pd

os.makedirs("data", exist_ok=True)
db_path = "data/sample_ecommerce.db"

random.seed(42)
CITIES = ["昆明", "成都", "重庆", "贵阳", "西安"]
REGIONS = ["西南", "华南", "华北", "华东", "东北"]
CHANNELS = ["线上", "线下"]
VIP_POOL = ["是", "否"]

# ---- customers ----
customers = []
for i in range(120):
    customers.append({
        "customer_id": i + 1,
        "name": f"用户{i + 1:03d}",
        "city": random.choice(CITIES),
        "register_date": f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "is_vip": random.choice(VIP_POOL),
    })

# ---- products ----
products = []
cats = [("家电", 2000.0), ("数码", 1000.0), ("服饰", 300.0), ("食品", 50.0), ("美妆", 200.0)]
for cat, base in cats:
    for j in range(3):
        products.append({
            "product_id": len(products) + 1,
            "name": f"{cat}商品{j + 1}",
            "category": cat,
            "unit_price": round(base * random.uniform(0.8, 1.2), 2),
        })

# ---- orders（800 笔）----
orders = []
for i in range(800):
    cust = random.choice(customers)
    prod = random.choice(products)
    qty = random.randint(1, 10)
    amount = round(prod["unit_price"] * qty, 2)
    orders.append({
        "order_id": i + 1,
        "customer_id": cust["customer_id"],
        "product_id": prod["product_id"],
        "quantity": qty,
        "unit_price": prod["unit_price"],
        "total_amount": amount,
        "order_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "region": random.choice(REGIONS),
        "channel": random.choice(CHANNELS),
    })

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 建表（幂等）
cur.execute("DROP TABLE IF EXISTS orders")
cur.execute("DROP TABLE IF EXISTS customers")
cur.execute("DROP TABLE IF EXISTS products")
cur.execute("""CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT,
    register_date TEXT, is_vip TEXT)""")
cur.execute("""CREATE TABLE products (
    product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, unit_price REAL)""")
cur.execute("""CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
    quantity INTEGER, unit_price REAL, total_amount REAL,
    order_date TEXT, region TEXT, channel TEXT)""")

cur.executemany("INSERT INTO customers VALUES (:customer_id,:name,:city,:register_date,:is_vip)", customers)
cur.executemany("INSERT INTO products VALUES (:product_id,:name,:category,:unit_price)", products)
cur.executemany("""INSERT INTO orders VALUES
    (:order_id,:customer_id,:product_id,:quantity,:unit_price,:total_amount,:order_date,:region,:channel)""", orders)
conn.commit()

# 建索引（演示真实库习惯）
cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)")
conn.commit()

df = pd.read_sql_query(
    "SELECT o.order_id, c.name, c.city, c.is_vip, p.category, p.name AS product_name, "
    "o.quantity, o.unit_price, o.total_amount, o.order_date, o.region, o.channel "
    "FROM orders o JOIN customers c ON o.customer_id=c.customer_id "
    "JOIN products p ON o.product_id=p.product_id LIMIT 3", conn)
print(f"✅ 数据库 {db_path} 建库完成！")
print(f"customers={len(customers)} products={len(products)} orders={len(orders)}")
print("三表联查前 3 行预览:")
print(df)
conn.close()
