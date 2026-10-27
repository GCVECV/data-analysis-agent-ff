# eval/run_eval.py — 自动化评测（支持 CSV / SQLite、标准答案比对、语义打分、失败案例分析）
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from core.config import LLMConfig, AgentConfig
from core.llm import LLMClient
from core.datasource import CSVDataSource, SQLiteDataSource
from core.agent import run_agent

_JUDGE_SYS = """你是评测裁判。请根据"分析需求"对 Agent 生成的报告打分（0-5 分）。
评分标准：
- 5：答案正确、完整，包含需求要求的关键结论
- 4：答案正确但略有遗漏
- 3：答案部分正确，有遗漏或含糊
- 2：答案明显不完整或部分错误
- 1：答案基本错误
- 0：完全跑题
只输出一个 0-5 的整数，不要输出其他内容。"""


def load_golden(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _exp_hit(report: str, expected=None, expected_any=None) -> bool | None:
    """标准答案比对：expected 全部命中（纯数字 token 允许 2% 容差）；expected_any 任一命中即可。都没有则返回 None（不参与判分）。"""
    if expected is None and expected_any is None:
        return None

    def hit(tok):
        tok = str(tok)
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            nums = [float(m.replace(",", "")) for m in re.findall(r"[\d,]+\.?\d*", report)]
            return any(abs(n - float(tok)) <= max(1.0, float(tok) * 0.02) for n in nums)
        return tok in report

    exp_ok = all(hit(t) for t in expected) if expected is not None else True
    any_ok = any(hit(t) for t in expected_any) if expected_any is not None else True
    return exp_ok and any_ok


def judge_report(query: str, report: str, llm) -> int:
    """LLM-as-judge：对报告做语义打分（0-5），失败返回 -1。"""
    messages = [
        {"role": "system", "content": _JUDGE_SYS},
        {"role": "user", "content": f"分析需求: {query}\n\nAgent 报告:\n{report[:2000]}\n\n打分:"},
    ]
    try:
        resp, _ = llm.chat(messages)
        t = (resp.choices[0].message.content or "").strip()
        return int(t[0]) if t and t[0].isdigit() else -1
    except Exception:
        return -1


def make_ds_factory(data_path: str):
    if data_path.endswith((".db", ".sqlite")):
        def factory():
            return SQLiteDataSource(data_path)
        return factory
    df = pd.read_csv(data_path)

    def factory():
        return CSVDataSource(df.copy())
    return factory


def evaluate(golden_path: str, data_path: str, out_csv: str, limit: int | None = None,
             use_judge: bool = False):
    golden = load_golden(golden_path)
    if limit:
        golden = golden[:limit]
    ds_factory = make_ds_factory(data_path)
    rows = []
    failures = []

    for i, case in enumerate(golden, 1):
        print(f"[{i}/{len(golden)}] 正在评测: {case['id']} | {case['query']}", flush=True)
        ds = ds_factory()
        llm = LLMClient(LLMConfig())
        cfg = AgentConfig()
        t0 = time.time()
        try:
            run = run_agent(case["query"], ds, llm, cfg)
            report = run.report
            tools_used = [s["tool"] for s in run.steps]
            tool_ok = sum(1 for s in run.steps if s["ok"]) / max(1, len(run.steps))
            must_call_hit = all(t in tools_used for t in case.get("must_call", []))
            contain_hit = all(w in report for w in case.get("must_contain", []))
            exp_hit = _exp_hit(report, case.get("expected"), case.get("expected_any"))
            ratios = [s["compression_ratio"] for s in run.steps if s.get("compression_ratio") is not None]
            avg_ratio = round(sum(ratios) / len(ratios), 2) if ratios else 1.0
            judge_score = judge_report(case["query"], report, llm) if use_judge else None
        except Exception as e:
            report, tools_used, tool_ok = f"ERROR: {e}", [], 0
            must_call_hit = contain_hit = False
            avg_ratio, judge_score = 1.0, None

        # 判定：有标准答案(expected)时以答案对错为准；无标准答案时按 must_call/contain 硬校验
        if exp_hit is None:
            ok = must_call_hit and contain_hit and (judge_score is None or judge_score >= 3)
        else:
            ok = exp_hit and (judge_score is None or judge_score >= 3)
        rows.append({
            "id": case["id"],
            "difficulty": case.get("difficulty", "unsorted"),
            "query": case["query"],
            "ok": ok,
            "must_call_hit": must_call_hit,
            "must_contain_hit": contain_hit,
            "expected_hit": exp_hit if exp_hit is not None else "",
            "judge_score": judge_score if judge_score is not None else "",
            "tool_success_rate": round(tool_ok, 3),
            "avg_compression_ratio": avg_ratio,
            "steps": len(tools_used),
            "llm_calls": llm.usage.calls,
            "prompt_tokens": llm.usage.prompt_tokens,
            "completion_tokens": llm.usage.completion_tokens,
            "elapsed_s": round(time.time() - t0, 2),
        })
        if not ok:
            failures.append({
                "id": case["id"], "query": case["query"],
                "tools_used": tools_used,
                "must_call_hit": must_call_hit, "must_contain_hit": contain_hit,
                "expected_hit": exp_hit if exp_hit is not None else "",
                "judge_score": judge_score, "report": report[:800],
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    n = len(out)
    print(f"\n===== 评测汇总（{n} 条 | judge={'ON' if use_judge else 'OFF'}）=====")
    if n:
        print(f"任务通过率        {out['ok'].mean():.1%}")
        print(f"工具调用成功率    {out['tool_success_rate'].mean():.1%}")
        exp_col = out["expected_hit"]
        if exp_col.notna().any() and (exp_col != "").any():
            print(f"标准答案命中率    {(exp_col.astype(str) == 'True').mean():.1%}")
        print(f"平均步数          {out['steps'].mean():.2f}")
        print(f"平均 LLM 调用     {out['llm_calls'].mean():.2f}")
        print(f"平均 tokens       {(out['prompt_tokens'] + out['completion_tokens']).mean():.0f}")
        print(f"平均压缩比        {out['avg_compression_ratio'].mean():.2f}")
        print(f"平均耗时(s)       {out['elapsed_s'].mean():.2f}")
        if use_judge:
            valid = out[out["judge_score"] != ""]
            print(f"平均语义得分      {valid['judge_score'].astype(float).mean():.2f} / 5")
        # 按难度分组统计（无 difficulty 字段的旧题库统一归为 unsorted）
        print("\n===== 按难度分组通过率 =====")
        grp = out.groupby("difficulty")["ok"].agg(["mean", "count"])
        for diff, row in grp.iterrows():
            print(f"{str(diff):<12} 通过率 {row['mean']:.1%} （{int(row['count'])} 条）")

    # 失败案例分析
    if failures:
        fail_path = Path(out_csv).with_name("failure_report.md")
        lines = ["# 评测失败案例分析", "", f"共 {len(failures)} 条失败用例", ""]
        for f in failures:
            lines += [
                f"## {f['id']}: {f['query']}",
                f"- 工具调用: {f['tools_used']}",
                f"- must_call 命中: {f['must_call_hit']} | must_contain 命中: {f['must_contain_hit']} | judge: {f['judge_score']}",
                f"- 报告节选: {f['report'][:400]}",
                "",
            ]
        fail_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"失败分析已写入: {fail_path}")
    else:
        print("无失败用例 🎉")

    print(f"明细已写入: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="数据分析 Agent 评测")
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--data", default="data/sample_sales.csv")
    ap.add_argument("--out", default="eval/results.csv")
    ap.add_argument("--limit", type=int, default=None, help="只评测前 N 条")
    ap.add_argument("--judge", action="store_true", help="启用 LLM-as-judge 语义打分（额外消耗 tokens）")
    args = ap.parse_args()
    evaluate(args.golden, args.data, args.out, limit=args.limit, use_judge=args.judge)
