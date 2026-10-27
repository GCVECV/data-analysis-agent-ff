# server/app.py — FastAPI 服务化
#给机器用的接口
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.config import LLMConfig, AgentConfig
from core.llm import LLMClient
from core.datasource import CSVDataSource, SQLiteDataSource
from core.agent import run_agent
from core.memory import Memory

app = FastAPI(title="数据分析 Agent API", version="2.0.0", description="Plan → Execute → Report 的数据分析 Agent，支持记忆、反思与动态重规划。")

_ROOT = Path(__file__).resolve().parents[1]
_memory = Memory(store_path=str(_ROOT / "memory_store.jsonl"))  # 服务级单例：跨请求共享长期记忆


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="自然语言分析需求")
    kind: str = Field("csv", description="数据源类型: csv / sqlite")
    csv_path: str | None = Field(None, description="CSV 文件路径（kind=csv 时必填，缺省用内置示例）")
    db_path: str | None = Field(None, description="SQLite 文件路径（kind=sqlite 时必填，缺省用内置示例库）")
    use_memory: bool = Field(True, description="是否启用记忆")


def _load_ds(req: AnalyzeRequest):
    root = Path(__file__).resolve().parents[1]
    if req.kind == "sqlite":
        db_path = req.db_path or str(root / "data" / "sample_ecommerce.db")
        if not Path(db_path).exists():
            raise HTTPException(404, f"SQLite 文件不存在: {db_path}")
        return SQLiteDataSource(db_path)
    if req.kind != "csv":
        raise HTTPException(400, "kind 仅支持 csv / sqlite")
    csv_path = req.csv_path or str(root / "data" / "sample_sales.csv")
    if not Path(csv_path).exists():
        raise HTTPException(404, f"CSV 文件不存在: {csv_path}")
    import pandas as pd
    return CSVDataSource(pd.read_csv(csv_path))


@app.get("/health")
def health():
    return {"status": "ok", "memory_records": len(_memory._long)}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """同步分析：返回报告、计划、执行步骤与用量统计。"""
    llm = LLMClient(LLMConfig())
    ds = _load_ds(req)
    memory = _memory if req.use_memory else None
    run = run_agent(req.query, ds, llm, AgentConfig(), memory=memory)
    return {
        "report": run.report,
        "plan": run.plan,
        "steps": run.steps,
        "usage": run.usage,
        "has_charts": bool(run.artifacts),
    }


@app.post("/analyze/stream")
def analyze_stream(req: AnalyzeRequest):
    """SSE 流式分析：先输出 meta（计划/图表状态），再流式输出报告文本。"""
    llm = LLMClient(LLMConfig())
    ds = _load_ds(req)
    memory = _memory if req.use_memory else None

    def gen():
        chunks: list[str] = []
        run = run_agent(req.query, ds, llm, AgentConfig(), memory=memory, stream_out=chunks.append)
        meta = {
            "type": "meta",
            "plan": run.plan,
            "has_charts": bool(run.artifacts),
            "steps": len(run.steps),
            "usage": run.usage,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        for c in chunks:
            yield f"data: {json.dumps({'type': 'text', 'content': c}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
