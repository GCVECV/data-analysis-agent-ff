import json
from .tools_csv import ToolResult, _ok, _err, plot_chart

def list_tables(args, ds) -> ToolResult:
    return _ok({"tables": ds.overview().get("tables", [])})

def describe_table(args, ds) -> ToolResult:
    table = args.get("table", "")
    try:
        cur = ds._conn.cursor()
        cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
        if not cols:
            return _err(f"表 '{table}' 不存在", ds.overview().get("tables", []))
        return _ok({"table": table,
                    "columns": [{"name": c[1], "type": c[2]} for c in cols]})
    except Exception as e:
        return _err(f"describe 失败: {e}")


def run_sql(args, ds) -> ToolResult:
    sql = args.get("sql", "")
    if not sql.strip():
        return _err("sql 不能为空")
    try:
        df = ds.run_sql(sql)
    except Exception as e:
        return _err(f"SQL 执行失败: {e}")
    preview = df.head(20).to_dict(orient="records")
    return _ok({
        "rows": len(df),
        "columns": list(df.columns),
        "preview": preview,
        "sql": sql,
    })


def plot_result(args, ds) -> ToolResult:
    try:
        _ = ds.last_result()
    except Exception as e:
        return _err(str(e))
    return plot_chart(args, ds)
