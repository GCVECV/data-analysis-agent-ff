# tests/test_agent_flow.py — Agent 主流程测试（mock LLM（模拟大模型），不消耗真实 API）
import json
from types import SimpleNamespace

import pandas as pd
import pytest  

from core.agent import run_agent
from core.config import AgentConfig
from core.datasource import CSVDataSource
from core.memory import Memory

"""
# 真实 OpenAI SDK 返回的消息对象里，有一个 tool_calls 列表。
#它告诉程序：“我想调用名字叫 xxx 的工具，参数是 yyy”。
#列表里每个元素就是一个工具调用对象，结构大致是：
{
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "get_data_overview",
        "arguments": "{}"
    }
}
"""
# ---------- mock 工具 ----------
def _tool_call(name, arguments): #辅助函数 _tool_call，用来模拟一次“工具调用”
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )
def _message(content=None, tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,  #表示“工具调用列表”
        model_dump=lambda: {"role": "assistant", "content": content, "tool_calls": tool_calls},
    )

"""响应对象：
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...]
)
#响应对象里最重要的字段是 choices
响应对象 resp
  └── choices 列表
        └── choices[0] 候选结果对象
              └── message 消息对象
                    ├── role
                    ├── content
                    └── tool_calls"""


def _resp(content=None, tool_calls=None): #用来模拟 OpenAI 返回的“响应对象”
    return SimpleNamespace(choices=[SimpleNamespace(message=_message(content, tool_calls))])

class FakeLLM:
    """按顺序弹出预设响应，用于确定性测试 Agent 主流程。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.usage = SimpleNamespace(calls=0, prompt_tokens=0, completion_tokens=0)
        self.chat_log: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.usage.calls += 1
        self.chat_log.append(messages)
        if not self._responses:
            raise AssertionError("FakeLLM 响应耗尽，实际调用次数超过预设")
        return self._responses.pop(0), 0.01

    def stream(self, messages, tools=None):
        self.usage.calls += 1
        self.chat_log.append(messages)
        yield "流式报告内容"


@pytest.fixture
def ds():
    df = pd.DataFrame({
        "region": ["A", "A", "B", "B", "C"],
        "sales": [10, 20, 30, 40, 500],
        "quantity": [1, 2, 3, 4, 5],
    })
    return CSVDataSource(df)


def _plan_resp(steps):
    return _resp(content=json.dumps({"steps": steps}, ensure_ascii=False))


def test_plan_execute_report_flow(ds):
    """正常流程：规划 → 工具调用 → 结论 → 报告。"""
    llm = FakeLLM([
        _plan_resp([
            {"id": 1, "goal": "查看数据概览", "tool_hint": "get_data_overview"},
            {"id": 2, "goal": "汇总生成报告", "tool_hint": ""},
        ]),
        _resp(tool_calls=[_tool_call("get_data_overview", {})]),
        _resp(content="STEP_DONE: 数据共 5 行 3 列"),
        _resp(content="# 报告\n数据共 5 行 3 列"),
    ])
    run = run_agent("数据有多少行", ds, llm, AgentConfig())
    assert run.report.startswith("# 报告")
    assert len(run.steps) == 1
    assert run.steps[0]["tool"] == "get_data_overview"
    assert run.steps[0]["ok"] is True
    assert llm.usage.calls == 4


def test_memory_injection_and_save(ds, tmp_path):
    """记忆：规划上下文注入历史；执行后写入短期与长期记忆。"""
    mem = Memory(store_path=str(tmp_path / "mem.jsonl"))
    mem.remember_long("数据有多少行", "5 行 3 列")  # 预置一条长期记忆
    llm = FakeLLM([
        _plan_resp([
            {"id": 1, "goal": "查看数据概览", "tool_hint": "get_data_overview"},
            {"id": 2, "goal": "汇总生成报告", "tool_hint": ""},
        ]),
        _resp(tool_calls=[_tool_call("get_data_overview", {})]),
        _resp(content="STEP_DONE: 完成"),
        _resp(content="# 报告\n完成"),
    ])
    run_agent("数据有多少行", ds, llm, AgentConfig(), memory=mem)
    # 规划器 user 消息应包含长期记忆内容
    planner_user = llm.chat_log[0][1]["content"]
    assert "历史任务参考" in planner_user
    # 执行后短期与长期记忆均被写入
    assert any(m["kind"] == "需求" for m in mem.short)
    assert len(mem._long) == 2


def test_reflect_on_failed_step(ds):
    """反思：工具失败后插入修正步骤，最终完成分析。"""
    llm = FakeLLM([
        _plan_resp([
            {"id": 1, "goal": "查看数据概览", "tool_hint": "get_data_overview"},
            {"id": 2, "goal": "汇总生成报告", "tool_hint": ""},
        ]),
        _resp(tool_calls=[_tool_call("calculate_stats", {"column": "不存在"})]),
        _resp(content="STEP_DONE: 列不存在，需要先查看概览"),
        _resp(content=json.dumps(
            {"reason": "列名错误", "next_goal": "先查看数据概览获取真实列名", "tool_hint": "get_data_overview"},
            ensure_ascii=False)),
        _resp(tool_calls=[_tool_call("get_data_overview", {})]),
        _resp(content="STEP_DONE: 概览完成"),
        _resp(content="# 报告\n完成"),
    ])
    run = run_agent("分析销售额", ds, llm, AgentConfig())
    tools = [s["tool"] for s in run.steps]
    assert "calculate_stats" in tools      # 失败调用被记录
    assert "get_data_overview" in tools    # 反思补丁步骤被执行
    assert run.report.startswith("# 报告")
    # 失败记录带 error
    failed = [s for s in run.steps if not s["ok"]]
    assert failed and failed[0]["error"]


def test_replan_after_repeated_failures(ds):
    """动态重规划：连续失败达到阈值（反思无有效方案）后重写计划。"""
    llm = FakeLLM([
        _plan_resp([
            {"id": 1, "goal": "统计列A", "tool_hint": "calculate_stats"},
            {"id": 2, "goal": "统计列B", "tool_hint": "calculate_stats"},
            {"id": 3, "goal": "汇总生成报告", "tool_hint": ""},
        ]),
        # p0：失败（两轮）
        _resp(tool_calls=[_tool_call("calculate_stats", {"column": "列A"})]),
        _resp(content="STEP_DONE: 失败"),
        # 反思1：无有效修正方案（不返回 JSON → reflect 返回 None）
        _resp(content="无法修正，请换一个思路"),
        # p1：失败（两轮，参数不同避免 seen_calls 去重）
        _resp(tool_calls=[_tool_call("calculate_stats", {"column": "列B"})]),
        _resp(content="STEP_DONE: 仍失败"),
        # 反思2：仍无有效方案 → fail_streak=2 触发重规划
        _resp(content="仍然无法修正"),
        # 重规划：给出新计划
        _resp(content=json.dumps({"steps": [
            {"id": 1, "goal": "用概览工具查看数据", "tool_hint": "get_data_overview"},
            {"id": 2, "goal": "汇总生成报告", "tool_hint": ""}]}, ensure_ascii=False)),
        # 新计划步骤：成功（两轮）
        _resp(tool_calls=[_tool_call("get_data_overview", {})]),
        _resp(content="STEP_DONE: 概览完成"),
        # 报告
        _resp(content="# 报告\n完成"),
    ])
    run = run_agent("分析销售额", ds, llm, AgentConfig())
    tools = [s["tool"] for s in run.steps]
    assert tools.count("calculate_stats") == 2   # 两个失败步骤都被记录
    assert "get_data_overview" in tools          # 重规划后的新步骤被执行
    assert run.report.startswith("# 报告")
    assert llm.usage.calls == 11


def test_stream_report(ds):
    """流式报告：stream_out 收集 chunk。"""
    llm = FakeLLM([
        _plan_resp([
            {"id": 1, "goal": "查看数据概览", "tool_hint": "get_data_overview"},
            {"id": 2, "goal": "汇总生成报告", "tool_hint": ""},
        ]),
        _resp(tool_calls=[_tool_call("get_data_overview", {})]),
        _resp(content="STEP_DONE: 完成"),
    ])
    chunks: list[str] = []
    run = run_agent("数据有多少行", ds, llm, AgentConfig(), stream_out=chunks.append)
    assert "".join(chunks) == "流式报告内容"
    assert run.report == "流式报告内容"
