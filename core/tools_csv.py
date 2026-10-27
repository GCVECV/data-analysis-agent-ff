from __future__ import annotations
import io, base64, json

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

@dataclass
class ToolResult:
    ok: bool
    text: str
    data: dict = field(default_factory=dict)
    
    artifacts: list = field(default_factory=list)
    
    error: str = ""


def _ok(data, text=""):
    return ToolResult(ok=True, text=text, data=data)
"""
_ok(概览值) 执行时：
_ok 的 data 形参（形参就是函数定义里的参数名）收到 概览值（概览值就是 ds.overview() 返回的结果）
_ok 的 text 形参（形参就是函数定义里的参数名）没有收到第二个实参（实参就是调用时真正传进去的值），所以使用默认值空字符串 ""
然后 _ok 创建 ToolResult 实例（实例就是按类造出来的具体对象），其中：
ok=True（表示成功）
text=""（文本说明是空字符串）
data=概览值（概览数据放在 data 字段里）
"""

def _err(msg, cols=None):

    full = msg + (f"\n可用列（只能从中选择）：{cols}" if cols is not None else "")
    return ToolResult(ok=False, text=full, error=msg)
    
def get_data_overview(args, ds):
    
    return _ok(ds.overview())

def calculate_stats(args, ds):
    df = ds.get_dataframe()
    col = args.get("column", "")
    if col not in df.columns:
        return _err(f"列 '{col}' 不存在", list(df.columns))
       
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        d = {"column": col, "count": int(s.count()),
             "mean": round(float(s.mean()), 4) if s.count() else None,
             "median": round(float(s.median()), 4) if s.count() else None,
             "min": float(s.min()) if s.count() else None, 
             "max": float(s.max()) if s.count() else None,
             "std": round(float(s.std()), 4) if s.count() > 1 else None,
             "missing": int(s.isnull().sum())}
        
    else:
        d = {"column": col, "type": "categorical", "nunique": int(s.nunique()),
             "top5": {str(k): int(v) for k, v in s.value_counts().head(5).items()},
             "missing": int(s.isnull().sum())}
    return _ok(d)


def group_aggregate(args, ds):
    df = ds.get_dataframe().copy()
    
    gcol, acol, func = args.get("group_column", ""), args.get("agg_column", ""), args.get("agg_func", "sum")
    granularity = args.get("granularity", "day")
    for c in (gcol, acol):
        if c not in df.columns: return _err(f"列 '{c}' 不存在", list(df.columns))
    if func not in {"sum","mean","count","max","min"}:
        return _err(f"不支持的聚合: {func}")
    if func != "count" and not pd.api.types.is_numeric_dtype(df[acol]):
        return _err(f"聚合列 '{acol}' 非数值，无法 {func}")
    
    if granularity in ("month", "quarter"):
        if not pd.api.types.is_datetime64_any_dtype(df[gcol]):
            try:
                df[gcol] = pd.to_datetime(df[gcol], errors="coerce", format="mixed")
            except Exception:
                return _err(f"列 '{gcol}' 无法解析为日期，不能按 {granularity} 聚合")
        df[gcol] = df[gcol].dt.to_period("M").astype(str) if granularity == "month" else df[gcol].dt.to_period("Q").astype(str)

    elif granularity == "day":
        if pd.api.types.is_datetime64_any_dtype(df[gcol]):
            df[gcol] = df[gcol].dt.strftime("%Y-%m-%d")
            

    try:
        r = df.groupby(gcol)[acol].agg(func).sort_values(ascending=False).head(20)
        res_dict = {str(k): (round(float(v),4) if isinstance(v,(int,float,np.floating)) else v) for k,v in r.items()}
        ds.set_last_agg({"group": gcol, "agg_column": acol, "agg_func": func, "granularity": granularity, "result": res_dict})
        
        return _ok({"group_column": gcol, "agg_column": acol, "agg_func": func,
                    "granularity": granularity, "result": res_dict})
    except Exception as e:
        return _err(f"聚合失败: {e}")

def plot_chart(args, ds):
    df = ds.get_dataframe()
    ct, x, y, title = args.get("chart_type",""), args.get("x_column",""), args.get("y_column",""), args.get("title","")
    if x not in df.columns:
        lr = ds.last_result() if hasattr(ds, "last_result") else None
        if lr is not None and x in lr.columns:
            df = lr
        else:
            return _err(f"X 列 '{x}' 不存在", list(df.columns))
    
    y_cols = [col.strip() for col in y.split(",")] if y else []
    
    last_agg = ds.last_agg() if hasattr(ds, "last_agg") else None
    use_agg = False
    if last_agg and last_agg.get("group") == x and ct in ("bar", "line", "pie"):
        agg_col = last_agg.get("agg_column", "")
        if (not y) or (y_cols and y_cols[0] == agg_col):
            use_agg = True

    try:
        fig, ax = plt.subplots(figsize=(10,5))
        if use_agg:
            items = list(last_agg["result"].items())
            if ct == "line":
                items.sort(key=lambda kv: str(kv[0]))
            elif ct == "bar":
                items.sort(key=lambda kv: -kv[1])
            labels = [str(k) for k, _ in items]
            values = [float(v) for _, v in items]
            if ct == "pie":
                ax.pie(values, labels=labels, autopct="%1.1f%%"); ax.axis("equal")
            elif ct == "line":
                ax.plot(labels, values, marker="o", markersize=4)
            else:
                ax.bar(labels, values)
            ax.set_xlabel(x)
            ax.set_ylabel(agg_col)
            plt.xticks(rotation=45, ha="right")
        elif ct == "bar":
            if not y_cols: return _err("柱状图必须指定 y_column")
            for col in y_cols:
                if col not in df.columns: return _err(f"Y 列 '{col}' 不存在", list(df.columns))
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return _err(f"Y 列 '{col}' 非数值", list(df.select_dtypes("number").columns))
            
            if len(y_cols) > 1:
                data = df.set_index(x)[y_cols].head(15)
                data.plot(kind="bar", ax=ax)
            else:
                data = df.groupby(x)[y_cols[0]].sum().sort_values(ascending=False).head(15)
                ax.bar(data.index.astype(str), data.values)
            
            ax.set_xlabel(x)
            ax.set_ylabel("数值" if len(y_cols) > 1 else y_cols[0])
            plt.xticks(rotation=45, ha="right")

        elif ct == "line":
            if not y_cols: return _err("折线图必须指定 y_column")
            col = y_cols[0]
            if col not in df.columns: return _err(f"Y 列 '{col}' 不存在", list(df.columns))
            d = df[[x,col]].dropna().copy()
            if not pd.api.types.is_datetime64_any_dtype(d[x]):
                try: d[x] = pd.to_datetime(d[x])
                except Exception: pass
            d = d.sort_values(x); ax.plot(d[x], d[col], marker="o", markersize=3)
            ax.set_xlabel(x); ax.set_ylabel(col)

        elif ct == "pie":
            data = df[x].value_counts().head(10)
            ax.pie(data.values, labels=data.index.astype(str), autopct="%1.1f%%"); ax.axis("equal")

        elif ct == "hist":
            if not pd.api.types.is_numeric_dtype(df[x]): return _err(f"直方图需数值列，'{x}' 非数值")
            ax.hist(df[x].dropna(), bins=30, edgecolor="black"); ax.set_xlabel(x); ax.set_ylabel("频次")

        elif ct == "scatter":
            if not y_cols: return _err("散点图必须指定 y_column")
            col = y_cols[0]
            if col not in df.columns: return _err(f"Y 列 '{col}' 不存在", list(df.columns))
            ax.scatter(df[x], df[col], alpha=0.5, s=10); ax.set_xlabel(x); ax.set_ylabel(col)
        else:
            return _err(f"不支持的图表类型 '{ct}'")
            
        ax.set_title(title or f"{ct} - {x}" + (f" vs {y}" if y else ""))
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=100, bbox_inches="tight"); plt.close(fig)
        buf.seek(0); img = base64.b64encode(buf.read()).decode()
    except Exception as e:
        return _err(f"绘图失败: {e}")
    
    text = f"图表已生成并展示: {ct} ({x}{' vs '+y if y else ''})"
    return ToolResult(ok=True, text=text, data={"chart_type": ct, "x": x, "y": y}, artifacts=[{"title": title or f"{ct} {x}", "image": img}])

def data_clean(args, ds):
    df = ds.get_dataframe().copy()
    action, col, fv_in = args.get("action",""), args.get("column",""), args.get("fill_value","")
    if action not in {"drop_na","fill_na","to_datetime","remove_outliers"}:
        return _err(f"不支持: {action}")
    if col and col not in df.columns: return _err(f"列 '{col}' 不存在", list(df.columns))
    if action == "drop_na":
        before = len(df); df = df.dropna(subset=[col]) if col else df.dropna(); ds.update(df)
        return _ok({"action":"drop_na","scope":col or "all","removed":before-len(df),"remain":len(df)})
    if action == "fill_na":
        if not col: return _err("fill_na 必须指定 column")
        s = df[col]; miss = int(s.isnull().sum())
        if miss == 0: return _ok({"action":"fill_na","column":col,"filled":0})
        if pd.api.types.is_numeric_dtype(s):
            if fv_in == "mean": fv = float(s.mean())
            elif fv_in == "median": fv = float(s.median())
            else:
                try: fv = float(fv_in) if fv_in != "" else 0
                except ValueError: return _err(f"填充值需数字或 mean/median")
            df[col] = s.fillna(fv)
        else:
            fv = fv_in or (s.mode().iloc[0] if not s.mode().empty else "未知")
            df[col] = s.fillna(fv)
        ds.update(df)
        return _ok({"action":"fill_na","column":col,"fill_value":str(fv),"filled":miss})
    if action == "to_datetime":
        targets = [col] if col else []
        if not col:
            kws = ["时间","日期","date","time","day","month","年","月"]
            for c in df.columns:
                if df[c].dtype != object: continue
                sample = df[c].dropna().head(10)
                if len(sample) == 0: continue
                hit = any(k in str(c).lower() for k in kws)
                try:
                    if hit or pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.8: targets.append(c)
                except Exception: pass
        ok, fail = [], []
        for c in targets:
            try: df[c] = pd.to_datetime(df[c], errors="coerce"); ok.append(c)
            except Exception: fail.append(c)
        ds.update(df)
        return _ok({"action":"to_datetime","converted":ok,"failed":fail})
    cols = [col] if col else list(df.select_dtypes("number").columns)
    if not cols: return _err("没有数值列")
    if col and not pd.api.types.is_numeric_dtype(df[col]): return _err(f"列 '{col}' 非数值")
    before = len(df); mask = pd.Series(True, index=df.index); detail = {}
    for c in cols:
        s = df[c].dropna()
        if len(s) < 4: detail[c]=0; continue
        q1,q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3-q1
        if iqr == 0: detail[c]=0; continue
        lo, hi = q1-1.5*iqr, q3+1.5*iqr
        cm = df[c].isna() | df[c].between(lo,hi); detail[c]=int((~cm).sum()); mask &= cm
    df = df[mask]; ds.update(df)
    return _ok({"action":"remove_outliers","columns":cols,"detail":detail,"removed":before-len(df),"remain":len(df)})