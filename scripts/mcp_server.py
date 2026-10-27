# scripts/mcp_server.py — 将数据分析 Agent 的核心工具注册为 MCP 服务
# 启动：python scripts/mcp_server.py（streamable-http 传输）
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from core.config import LLMConfig
from core.datasource import CSVDataSource, SQLiteDataSource
from core.tools_csv import get_data_overview as _csv_overview
from core.tools_csv import group_aggregate as _group_agg
from core.tools_sql import run_sql as _run_sql

mcp = FastMCP("data-analysis-tools")

# 默认数据源：CSV 用内置样例，SQL 用内置示例库
_ROOT = Path(__file__).resolve().parents[1]
_csv_path = _ROOT / "data" / "sample_sales.csv"
_db_path = _ROOT / "data" / "sample_ecommerce.db"

import pandas as pd


def _csv_ds() -> CSVDataSource:
    return CSVDataSource(pd.read_csv(_csv_path))


@mcp.tool()
def get_data_overview() -> str:
    """获取当前数据集的行列数、字段类型和缺失情况。分析第一步必须调用。"""
    r = _csv_overview({}, _csv_ds())
    return json.dumps(r.data, ensure_ascii=False)


@mcp.tool()
def group_aggregate(group_column: str, agg_column: str, agg_func: str = "sum") -> str:
    """按某列分组对另一列聚合，返回聚合结果。"""
    r = _group_agg({"group_column": group_column, "agg_column": agg_column, "agg_func": agg_func}, _csv_ds())
    if not r.ok:
        return f"ERROR: {r.text}"
    return json.dumps(r.data, ensure_ascii=False)


@mcp.tool()
def run_sql(sql: str) -> str:
    """对内置 SQLite 示例库执行只读 SELECT 查询，返回行数、列名和前 20 行预览。"""
    r = _run_sql({"sql": sql}, SQLiteDataSource(str(_db_path)))
    if not r.ok:
        return f"ERROR: {r.text}"
    return json.dumps(r.data, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
