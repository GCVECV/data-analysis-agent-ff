from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import sqlite3
import pandas as pd


class DataSource(ABC):
    """所有数据源的统一接口。Agent 层只依赖此接口。"""

    kind: str = "base" 
    
    @abstractmethod
    def overview(self) -> dict[str, Any]: ...
    

    @abstractmethod
    def schema_text(self) -> str: ...

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """CSV 模式返回主 DataFrame；SQL 模式返回当前主表，供绘图。"""

    def run_sql(self, sql: str) -> pd.DataFrame:
        raise NotImplementedError("当前数据源不支持 SQL")


class CSVDataSource(DataSource):
    kind = "csv"
    

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._last_agg: dict | None = None  

    def get_dataframe(self) -> pd.DataFrame:
        return self._df

    def update(self, df: pd.DataFrame) -> None:
        self._df = df

    def set_last_agg(self, agg: dict | None) -> None:
        self._last_agg = agg

    def last_agg(self) -> dict | None:
        return self._last_agg

    def overview(self) -> dict[str, Any]:
        df = self._df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        key_stats: dict[str, dict] = {}
        for c in num_cols[:8]:
            s = df[c].dropna()
            if len(s) == 0:
                continue
            key_stats[c] = {
                "mean": round(float(s.mean()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
            }
        return {
            "kind": "csv",
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            
            "missing": {c: int(v) for c, v in df.isnull().sum().items() if v > 0},
            "numeric_summary": key_stats,
        }

    def schema_text(self) -> str:
        df = self._df
        lines = [f"表: data ({len(df)} 行 × {len(df.columns)} 列)"]
        for c in df.columns:
            sample = df[c].dropna().head(2).tolist()
            lines.append(f"  - {c} ({df[c].dtype}), 示例: {sample}")
        return "\n".join(lines)


class SQLiteDataSource(DataSource):
    kind = "sqlite"
    _MAX_ROWS = 500

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._last_result: pd.DataFrame | None = None

    def _tables(self) -> list[str]: 
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            
        ).fetchall()
        return [r[0] for r in rows]
    
    
    def overview(self) -> dict[str, Any]:
        return {"kind": "sqlite", "tables": self._tables()}

    def schema_text(self) -> str:
        cur = self._conn.cursor()
        parts = []
        for t in self._tables():
            cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
            col_str = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            parts.append(f"表 {t}({col_str})")
        return "\n".join(parts)

    def run_sql(self, sql: str) -> pd.DataFrame:
        s = sql.strip().rstrip(";")
        head = s.lstrip().lower()
        
        if not (head.startswith("select") or head.startswith("with")):
            raise ValueError("仅允许 SELECT / WITH 查询")
        
        for kw in ("insert", "update", "delete", "drop", "alter", "attach", "pragma"):
            if kw in head:
                raise ValueError(f"查询包含禁用关键字: {kw}")
        if "limit" not in head:
            s = s + f" LIMIT {self._MAX_ROWS}"
            
        df = pd.read_sql_query(s, self._conn)
        self._last_result = df
        return df

    def last_result(self) -> pd.DataFrame:
        if self._last_result is None:
            raise ValueError("尚未执行任何 SQL 查询")
        return self._last_result

    def get_dataframe(self) -> pd.DataFrame:
        if self._last_result is not None:
            return self._last_result
        tables = self._tables()
        if not tables:
            raise ValueError("数据库为空")
        return pd.read_sql_query(f"SELECT * FROM {tables[0]}", self._conn)