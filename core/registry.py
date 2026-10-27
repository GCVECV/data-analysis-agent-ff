from __future__ import annotations
from typing import Callable
from .tools_csv import(

    get_data_overview, calculate_stats, group_aggregate, plot_chart, data_clean,
)
from .tools_sql import list_tables, describe_table, run_sql, plot_result


_CSV_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_data_overview",
        "description": "获取数据集基本信息（行列数、列名、类型、缺失、数值摘要）。分析第一步必须调用。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "calculate_stats",
        "description": "计算指定列统计量或分类分布。",
        "parameters": {"type": "object", "properties": {
            "column": {"type": "string", "description": "列名，必须来自 overview 的 columns"}},
            "required": ["column"]}}},
    {"type": "function", "function": {
        "name": "group_aggregate",
        "description": "按某列分组对另一列聚合。若需要按时间趋势聚合，可设置 granularity=month/quarter/day 按时间粒度分组；聚合结果会自动缓存，供后续 plot_chart 直接绘图。",
        "parameters": {"type": "object", "properties": {
            "group_column": {"type": "string"},
            "agg_column": {"type": "string"},
            "agg_func": {"type": "string", "enum": ["sum", "mean", "count", "max", "min"]},
            "granularity": {"type": "string", "enum": ["day", "month", "quarter"], "description": "日期分组粒度，默认 day"}},
            "required": ["group_column", "agg_column", "agg_func"]}}},
    {"type": "function", "function": {
        "name": "plot_chart",
        "description": "生成图表：bar/line/pie/hist/scatter。若最近一次 group_aggregate 的分组列与 x_column 一致，将直接基于聚合结果绘图（推荐做法：先聚合再画图）。",
        "parameters": {"type": "object", "properties": {
            "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "hist", "scatter"]},
            "x_column": {"type": "string"},
            "y_column": {"type": "string"},
            "title": {"type": "string"}},
            "required": ["chart_type", "x_column"]}}},
    {"type": "function", "function": {
        "name": "data_clean",
        "description": "清洗数据：drop_na / fill_na / to_datetime / remove_outliers。会修改当前数据集。",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["drop_na", "fill_na", "to_datetime", "remove_outliers"]},
            "column": {"type": "string"},
            "fill_value": {"type": "string"}},
            "required": ["action"]}}},
]

_SQL_SCHEMAS = [               
    {"type": "function", "function": {   
        "name": "list_tables",
        "description": "列出数据库所有表。第一步必须调用。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "describe_table",
        "description": "查看某张表的列定义。",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"}}, "required": ["table"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "执行只读 SELECT 查询，返回行数、列名、前 20 行预览。",
        "parameters": {"type": "object", "properties": {
            "sql": {"type": "string", "description": "仅 SELECT 查询"}},
            "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "plot_chart",
        "description": "基于最近一次 run_sql 结果生成图表（bar/line/pie/hist/scatter）。必须先 run_sql 拿到结果。",
        "parameters": {"type": "object", "properties": {
            "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "hist", "scatter"]},
            "x_column": {"type": "string"}, "y_column": {"type": "string"},
            "title": {"type": "string"}},
            "required": ["chart_type", "x_column"]}}},
]

_FUNCS: dict[str, Callable] = {
    "get_data_overview": get_data_overview,
    "calculate_stats": calculate_stats,
    "group_aggregate": group_aggregate,
    "plot_chart": plot_chart,
    "data_clean": data_clean,
    "list_tables": list_tables,
    "describe_table": describe_table,
    "run_sql": run_sql,
    "plot_result": plot_result,
}



def get_schemas(kind: str) -> list[dict]:
    return _SQL_SCHEMAS if kind == "sqlite" else _CSV_SCHEMAS

def dispatch(name: str, args: dict, ds) -> ToolResult:   
    
    if ds.kind == "csv":
        allowed = {"get_data_overview", "calculate_stats", "group_aggregate", "plot_chart", "data_clean"}
        fn = _FUNCS.get(name)
    else:
        allowed = {"list_tables", "describe_table", "run_sql", "plot_chart"}
        fn = plot_result if name == "plot_chart" else _FUNCS.get(name)
    if name not in allowed:
        return ToolResult(ok=False, text=f"工具 {name} 在当前数据源不可用", error=f"tool not allowed for {ds.kind}: {name}")
    if fn is None:
        return ToolResult(ok=False, text=f"未知工具 {name}", error=f"unknown tool {name}")
    try:
        return fn(args, ds)
    except Exception as e:
        return ToolResult(ok=False, text=f"工具执行异常: {e}", error=str(e))