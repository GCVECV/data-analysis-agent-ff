# 数据分析 Agent 服务镜像
FROM python:3.11-slim

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目
COPY . .

# 数据目录可挂载覆盖
VOLUME ["/app/data"]

EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
