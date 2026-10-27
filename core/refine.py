from __future__ import annotations
import json

_REFINE_SYS = """你是数据分析 Agent 的反思器。当前步骤执行失败或没有取得有效结论，请分析失败原因，并输出下一步修正方案。

必须只输出 JSON：{"reason": "失败原因一句话", "next_goal": "修正后的目标一句话", "tool_hint": "工具名（可为空字符串）"}

约束：
1. 不要重复与失败记录中完全相同的工具调用（避免死循环）。
2. 如果证据表明当前计划整体不可行，next_goal 应改为"汇总生成报告"。
3. next_goal 必须具体、可执行，不能只是"继续分析"。
"""

_REPLAN_SYS = """你是数据分析 Agent 的规划器。当前计划执行受阻，需要基于已执行的证据重新规划剩余步骤。

必须输出 JSON 计划：{"steps": [{"id": 1, "goal": "一句话", "tool_hint": "工具名"}]}

【强制要求】
1. 步骤数量 2~4 步。
2. 必须基于下方"已执行证据"，不要重复已经成功完成的步骤。
3. 最后一步固定为 {"goal": "汇总生成报告", "tool_hint": ""}。
"""


def _extract_json(text: str) -> dict | list | None:
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])

    except Exception:
        return None


def reflect(query: str, step_goal: str, recent_failures: list[dict], ds, llm) -> dict | None:
    """反思：分析失败原因并输出修正后的下一步行动。失败返回 None。"""
    if not recent_failures:
        return None
    evidence = "\n".join(
        f"- {e.get('tool')}: {e.get('error') or str(e.get('preview', ''))[:200]}"
        for e in recent_failures[-3:]
    )
    messages = [
        {"role": "system", "content": _REFINE_SYS},
        {"role": "user", "content": f"需求: {query}\n当前步: {step_goal}\n数据结构:\n{ds.schema_text()}\n最近失败记录:\n{evidence}\n输出 JSON。"},
    ]
    try:
        resp, _ = llm.chat(messages)
        obj = _extract_json(resp.choices[0].message.content or "")
        if not isinstance(obj, dict):
            return None
        goal = str(obj.get("next_goal", "")).strip()
        if not goal:
            return None
        return {"goal": goal, "tool_hint": str(obj.get("tool_hint") or "")}
    except Exception:
        return None


def replan(query: str, ds, llm, done_goals: list[str], evidence_text: str) -> list[dict]:
    """动态重规划：基于已执行证据重新生成剩余计划。失败返回空列表。"""
    messages = [
        {"role": "system", "content": _REPLAN_SYS},
        {"role": "user", "content": f"需求: {query}\n数据结构:\n{ds.schema_text()}\n已完成步骤: {done_goals}\n执行证据:\n{evidence_text[:2000]}\n输出 JSON。"},
    ]
    try:
        resp, _ = llm.chat(messages)
        obj = _extract_json(resp.choices[0].message.content or "")
        if not isinstance(obj, dict):
            return []
        steps = obj.get("steps", [])
        out = []
        for s in steps:
            if isinstance(s, dict) and str(s.get("goal", "")).strip():
                out.append({"goal": str(s["goal"]).strip(), "tool_hint": str(s.get("tool_hint") or "")})
        return out[:4]
    except Exception:
        return []
