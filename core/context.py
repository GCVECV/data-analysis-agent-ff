from __future__ import annotations
import json
from .llm import count_tokens

def _short(v):
    if isinstance(v, float): return round(v, 2)
    if isinstance(v, dict) and len(v) > 15:
        return {**dict(list(v.items())[:10]), "_note": f"共 {len(v)} 项"}
    if isinstance(v, list) and len(v) > 10:
        return v[:10] + [f"... 共 {len(v)} 项"]
    return v

_PRIORITY_KEYS = [
    "columns", "result", "numeric_summary", "preview", "error", 
    "dtypes", "rows", "cols", "kind", "missing", "mean", "median", 
    "min", "max", "std", "count", "top5", "nunique", "group_column", 
    "agg_column", "agg_func", "converted", "failed", "removed", 
    "remain", "action", "column", "fill_value", "filled", "detail"
]

def compress_tool_result(result, max_chars=2000):
    def recurse(obj):
        if isinstance(obj, dict): return {k: _short(recurse(v)) for k, v in obj.items()}
        if isinstance(obj, list): return [_short(recurse(x)) for x in obj[:20]]
        return obj

    compressed = recurse(result)
    text = json.dumps(compressed, ensure_ascii=False, default=str)
    if len(text) <= max_chars: return text
    if not isinstance(compressed, dict): return text[:max_chars] + "\n【已截断】"

    keep: list[str] = [k for k in _PRIORITY_KEYS if k in compressed]
    for k in compressed:
        if k not in keep: keep.append(k)

    while keep:
        reduced = {k: compressed[k] for k in keep}
        body = json.dumps(reduced, ensure_ascii=False, default=str)
        if len(body) <= max_chars:
            return body + "\n【已压缩】"
        keep.pop()

    return text[:max_chars] + "\n【已截断】"

def fit_history(messages, budget):
    if not messages: return messages
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    def total(msgs): return sum(count_tokens(str(m.get("content", ""))) for m in msgs)
    while rest and total(system + rest) > budget and len(rest) > 2:
        rest.pop(0)
    return system + rest