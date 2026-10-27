
from __future__ import annotations
import json 
from dataclasses import dataclass, field

import pandas as pd
from .config import AgentConfig
from .llm import LLMClient, count_tokens
from .context import compress_tool_result, fit_history
from .registry import get_schemas, dispatch
from .refine import reflect, replan

@dataclass   
class AgentRun:
    report: str
    plan: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)


_PLAN_SYS = """你是数据分析 Agent 的规划器。输出 JSON 计划。

{"steps": [{"id": 1, "goal": "一句话", "tool_hint": "工具名"}]}

【强制要求】
1. 步骤数量 3~5 步（绝不能只有 1 步）。
2. 第 1 步固定为 get_data_overview（或 list_tables）。
3. 涉及"排名/对比/趋势"的需求，必须有一步使用 group_aggregate（或 run_sql）。
4. 如果需求包含"图表/对比图"，必须有一步使用 plot_chart（或 plot_result）。
5. 最后一步固定为 {"goal": "汇总生成报告", "tool_hint": ""}。
"""

_EXEC_SYS = """你是数据分析 Agent 执行器。
规则：
1) 当前步骤目标一旦达成，立即输出 "STEP_DONE: <结论>"。
2) 【极其重要】你绝对不能输出任何 Python / SQL 代码块。所有的数据查询和画图，必须严格通过提供的工具列表（Function Calling）完成！
3) 在调用画图工具（plot_chart / plot_result）之前，确保你已经用 group_aggregate / run_sql 获取了正确的数据，且列名与画图工具的参数完全一致。如果画图工具报错，必须修改参数重试，直到成功。
4) 【极其重要】目标要求画图时，必须真正调用 plot_chart / plot_result 完成画图；禁止只输出文字声称"图已绘制"。绘图工具会自动处理聚合结果的排序与展示。
"""

_REPORT_SYS = """你是数据分析师。基于执行证据生成 Markdown 报告。
硬约束：
1. 字段名与分类值保持原样。
2. 【极其重要】严禁在报告中输出代码块（```python 或 ```sql）。
3. 【系统状态拦截】如果你在系统状态中看到“本次执行未生成图表”，你必须在报告末尾写明“由于执行原因，本次未能成功生成图表”。严禁捏造“已生成图表”或描述不存在的图表画面！
"""

def _parse_plan(text):
    text = (text or "").strip()
    if text.startswith("```"): 
        text = text.strip("`")
        text = text[text.find("{"):]
        
    try:
        obj = json.loads(text)
        return obj.get("steps", [])
    except Exception:
        return []
    


def _plan(query, ds, llm, mem_ctx=""):
    user_body = ds.schema_text() + "\n需求: " + query
    if mem_ctx:
        user_body = mem_ctx + "\n\n" + user_body
    messages = [{"role": "system", "content": _PLAN_SYS}, {"role": "user", "content": user_body}]
    resp, _ = llm.chat(messages)
    steps = _parse_plan(resp.choices[0].message.content or "")
    
    
    if not steps:
        first = "get_data_overview" if ds.kind == "csv" else "list_tables"
        steps = [{"id": 1, "goal": "概览", "tool_hint": first}, {"id": 2, "goal": "分析", "tool_hint": ""}, {"id": 3, "goal": "汇总生成报告", "tool_hint": ""}]
        
        
        
        
    query_asks_chart = any(k in query for k in ("图", "柱状", "折线", "散点", "直方", "饼图", "chart", "plot"))
    
    
    if query_asks_chart and not any("plot" in str(s.get("tool_hint") or "").lower() for s in steps):
        
        steps = [s for s in steps if not _is_report_step(s)]
        
        plot_hint = "plot_chart" if ds.kind == "csv" else "plot_result"
        steps.append({"id": len(steps) + 1, "goal": "绘制图表展示结果", "tool_hint": plot_hint})
        steps.append({"id": len(steps) + 2, "goal": "汇总生成报告", "tool_hint": ""})
        
        
    query_needs_clean = any(k in query for k in ("缺失", "异常", "剔除", "清洗", "缺失值", "删除", "fill", "drop"))
    if query_needs_clean and not any("data_clean" in str(s.get("tool_hint") or "") for s in steps):
        steps = [s for s in steps if not _is_report_step(s)]
        steps.insert(1, {"id": len(steps) + 1, "goal": "数据清洗：处理缺失值或异常值", "tool_hint": "data_clean"})
        steps.append({"id": len(steps) + 2, "goal": "汇总生成报告", "tool_hint": ""})
    return steps[:6]

def _is_report_step(step) -> bool:
    goal = str(step.get("goal", ""))
    hint = str(step.get("tool_hint") or "")
    return ("报告" in goal or "汇总" in goal) and not hint

def _execute_step(step, query, ds, llm, cfg, exec_log, artifacts, seen_calls: set):
    
    schemas = get_schemas(ds.kind)
    messages = [
        {"role": "system", "content": _EXEC_SYS},
        {"role": "user", "content": "目标: " + query + "\n当前步: " + str(step.get("goal")) + "\n数据结构:\n" + ds.schema_text()}
    ]
    
    conclusion = ""
    failed = False
    goal = str(step.get("goal", ""))
    step_requires_plot = ("图" in goal or "plot" in goal.lower())
    step_requires_query = any(k in goal for k in ("统计", "计算", "查询", "聚合", "找出", "识别", "对比", "排序", "占比", "分析"))
    
    
    step_queried = any(s.get("tool") in ("run_sql", "get_data_overview", "group_aggregate", "calculate_stats") and s.get("ok")
                       for s in exec_log if s.get("step_id") == step.get("id"))
    
    
    
    max_rounds = cfg.max_steps_per_plan + (2 if step_requires_plot else 1 if step_requires_query else 0)
    
    
    for _ in range(max_rounds):
        messages = fit_history(messages, cfg.context_token_budget)
        resp, elapsed = llm.chat(messages, tools=schemas)
        msg = resp.choices[0].message
        messages.append(msg.model_dump() if hasattr(msg, "model_dump") else msg)

        if not msg.tool_calls:
            content = msg.content or ""
            if "```python" in content or "import matplotlib" in content:
                messages.append({
                    "role": "user", 
                    "content": "【系统警告】你刚刚输出了一段 Python 代码，这不符合规定！请立刻调用工具列表中的 `plot_chart` 或 `plot_result` 来完成画图。不要输出任何代码！"
                })
                continue
            
            step_drew = any(s.get("tool") in ("plot_chart", "plot_result") and s.get("ok")
                            for s in exec_log if s.get("step_id") == step.get("id"))
            if step_requires_plot and not step_drew:
                messages.append({
                    "role": "user",
                    "content": "【系统警告】当前步骤要求生成图表，但你还未真正调用绘图工具（plot_chart / plot_result）。"
                               "请先调用绘图工具完成画图（可基于最近一次 group_aggregate 结果直接绘图），再输出结论。不要用文字描述'图已绘制'来代替实际调用！"
                })
                continue
            
            step_requires_clean = any(k in goal for k in ("剔除", "清洗", "缺失", "异常", "删除", "填充"))
            step_cleaned = any(s.get("tool") == "data_clean" and s.get("ok")
                               for s in exec_log if s.get("step_id") == step.get("id"))
            if step_requires_clean and not step_cleaned:
                messages.append({
                    "role": "user",
                    "content": "【系统警告】当前步骤要求清洗数据（剔除异常值 / 处理缺失值），但你还未真正调用 data_clean 工具。"
                               "请先调用 data_clean（action=remove_outliers 或 drop_na / fill_na）完成清洗，再输出结论。不要用文字描述'已剔除/已处理'来代替实际调用！"
                })
                continue
            if step_requires_query and not step_queried:
                messages.append({
                    "role": "user",
                    "content": "【系统警告】当前步骤要求统计数据/计算结果，但你还未真正调用任何数据工具（run_sql / group_aggregate / "
                               "calculate_stats / get_data_overview）。请先调用数据工具完成取数与计算（基于真实返回结果），再输出结论。"
                               "不要用文字描述'无法获取''未执行查询'来代替实际调用！"
                })
                continue
            conclusion = content
            break

        executed = 0
        for tc in msg.tool_calls:
            name = tc.function.name
            try: args = json.loads(tc.function.arguments or "{}")
            except Exception: args = {}
            sig = name + "::" + json.dumps(args, sort_keys=True, ensure_ascii=False)
            if sig in seen_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "已调用过，请直接输出 STEP_DONE。"})
                continue
            seen_calls.add(sig)

            result = dispatch(name, args, ds)
            
            if result.ok:
                data_str = compress_tool_result(result.data, cfg.tool_content_max_chars)
                
                
                payload = result.text + "\n" + data_str if result.text and result.data else (result.text or data_str)
                artifacts.extend(result.artifacts)
            else:
                payload = result.text
                failed = True

            raw_str = json.dumps(result.data, ensure_ascii=False) if result.ok and result.data else result.text
            raw_t = count_tokens(raw_str)
            comp_t = count_tokens(payload)
            
            exec_log.append({
                "step_id": step.get("id"), 
                "tool": name, 
                "args": args,
                "ok": result.ok, 
                "error": result.error,
                "latency_s": round(elapsed,2),
                "preview": payload[:300],
                "full_payload": payload,
                "raw_tokens": raw_t,
                "compressed_tokens": comp_t,
                "compression_ratio": round(raw_t / comp_t, 2) if comp_t > 0 else 1.0,
            })
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": payload})
            executed += 1

        if executed == 0:
            conclusion = "(本步因重复调用终止)"
            failed = True
            break

    if step_requires_plot and not any(s.get("tool") in ("plot_chart", "plot_result") and s.get("ok")
                                      for s in exec_log if s.get("step_id") == step.get("id")):
        if _auto_plot(step, query, ds, cfg, exec_log, artifacts):
            failed = False
            if not conclusion:
                conclusion = "(自动绘图：已基于最近一次查询结果生成图表)"
    return conclusion, failed

def _evidence_text(steps: list[dict]) -> str:
    text = ""
    for s in steps:
        text += f"\n--- 步骤 {s['step_id']} | 工具 {s['tool']} ---\n"
        text += s["full_payload"] + "\n"
    return text

def _auto_plot(step, query, ds, cfg, exec_log, artifacts):
    """框架兜底：模型在画图步骤反复未调用绘图工具时，基于最近一次成功的
    查询/聚合结果自动绘制图表，保证"要图必有图"，防止 LLM 口头声称。
    返回 True 表示兜底绘图成功。
    """
    from .tools_csv import plot_chart
    goal = str(step.get("goal", ""))
    if "趋势" in goal or "折线" in goal or "line" in goal.lower():
        ct = "line"
    elif "散点" in goal or "scatter" in goal.lower():
        ct = "scatter"
    elif "直方" in goal or "hist" in goal.lower():
        ct = "hist"
    else:
        ct = "bar"
    try:
        if ds.kind == "csv":
            last_agg = ds.last_agg()
            if last_agg:
                res = plot_chart({"chart_type": ct, "x_column": last_agg["group"], "y_column": last_agg["agg_column"]}, ds)
            else:
                df = ds.get_dataframe()
                num_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not num_cols:
                    return False
                if ct == "hist":
                    res = plot_chart({"chart_type": ct, "x_column": num_cols[0]}, ds)
                else:
                    y = num_cols[1] if len(num_cols) > 1 else ""
                    res = plot_chart({"chart_type": ct, "x_column": num_cols[0], "y_column": y}, ds)
        else:
            lr = ds.last_result()
            if lr is None or len(lr.columns) < 1:
                return False
            x = str(lr.columns[0])
            num_cols = [str(c) for c in lr.columns if pd.api.types.is_numeric_dtype(lr[c])]
            y = num_cols[0] if num_cols else ""
            res = plot_chart({"chart_type": ct, "x_column": x, "y_column": y}, ds)
    except Exception:
        return False
    if not res.ok:
        return False
    payload = res.text + "\n" + json.dumps(res.data, ensure_ascii=False)
    raw_t = count_tokens(payload)
    exec_log.append({
        "step_id": step.get("id"),
        "tool": "plot_chart",
        "args": {"chart_type": ct, "auto": True},
        "ok": True,
        "error": "",
        "latency_s": 0.0,
        "preview": payload[:300],
        "full_payload": payload,
        "raw_tokens": raw_t,
        "compressed_tokens": raw_t,
        "compression_ratio": 1.0,
    })
    artifacts.extend(res.artifacts)
    return True

def run_agent(query, ds, llm, cfg, trace=None, memory=None, stream_out=None):
    """执行一次数据分析 Agent 任务。

    参数：
    - trace: 可选的进度回调 trace(msg)
    - memory: 可选的 Memory 对象（记忆注入与保存）
    - stream_out: 可选的报告流式回调 stream_out(chunk)，提供时报告用流式 API 生成
    """
    trace = trace or (lambda *_: None)
    run = AgentRun(report="")

    mem_ctx = ""
    if memory is not None:
        mem_ctx = (memory.recall(query) + "\n" + memory.short_context()).strip()

    run.plan = _plan(query, ds, llm, mem_ctx)
    trace("计划步数: " + str(len(run.plan)))
    
    plan_body = list(run.plan)
    if plan_body and _is_report_step(plan_body[-1]):
        plan_body = plan_body[:-1]

    seen_calls = set()
    fail_streak = 0
    replan_count = 0
    i = 0
    while i < len(plan_body) and len(run.steps) < cfg.max_tool_calls:
        step = plan_body[i]
        trace("执行步骤: " + str(step.get("goal")))
        conclusion, failed = _execute_step(step, query, ds, llm, cfg, run.steps, run.artifacts, seen_calls)
        if len(run.steps) >= cfg.max_tool_calls:
            break

        if failed:
            fail_streak += 1
            if fail_streak <= cfg.max_reflects:
                patch = reflect(query, step.get("goal"), run.steps, ds, llm)
                if patch and patch.get("goal"):
                    trace("反思修正: " + patch["goal"])
                    plan_body.insert(i + 1, {"id": 900 + i, "goal": patch["goal"], "tool_hint": patch.get("tool_hint", "")})
            
            
            if fail_streak >= cfg.replan_after_fails and replan_count < cfg.max_replans:
                replan_count += 1
                trace("触发动态重规划")
                done_goals = [str(s.get("goal", "")) for s in plan_body[:i + 1]]
                new_plan = replan(query, ds, llm, done_goals, _evidence_text(run.steps))
                if new_plan:
                    if _is_report_step(new_plan[-1]):
                        new_plan = new_plan[:-1]
                    plan_body = plan_body[:i + 1] + new_plan
                fail_streak = 0
        else:
            fail_streak = 0
        i += 1

    trace("生成报告")
    evidence_text = _evidence_text(run.steps)
        
    if run.artifacts:
        chart_status = "成功：系统已生成了图表并在前端渲染。"
    else:
        chart_status = "失败：系统当前没有任何图表。严禁在报告中声称'已生成图表'，必须诚实说明图表生成失败或未生成。"

    messages = [
        {"role": "system", "content": _REPORT_SYS},
        {"role": "user", "content": f"需求: {query}\n\n【系统状态】{chart_status}\n\n证据:\n{evidence_text}\n\n生成报告。"}
    ]
    messages = fit_history(messages, cfg.context_token_budget)

    if stream_out is not None:
        chunks = []
        for chunk in llm.stream(messages):
            chunks.append(chunk)
            stream_out(chunk)
        run.report = "".join(chunks) or "（生成失败）"
    else:
        resp, _ = llm.chat(messages)
        run.report = resp.choices[0].message.content or "（生成失败）"

    run.usage = {"calls": llm.usage.calls, "prompt_tokens": llm.usage.prompt_tokens, "completion_tokens": llm.usage.completion_tokens}

    if memory is not None:
        memory.remember_short("需求", query)
        memory.remember_short("结论", run.report[:200])
        memory.remember_long(query, run.report[:300])

    return run
