# 数据分析 Agent（Data Analysis Agent）

基于 LLM Function Calling 的数据分析 Agent：给定一个数据集和一句自然语言需求，自动完成 **规划 → 工具调用 → 反思修正 → 报告生成** 的全流程，并配有可量化的自动化评测体系。

> 秋招项目说明：本项目用于展示 LLM 应用开发（Function Calling / Agent 机制 / 上下文管理 / 评测闭环 / 服务化部署）的综合能力。

## 功能特性

- **双数据源**：CSV 文件 / SQLite 数据库（三表业务库），统一 `DataSource` 接口
- **9 个内置工具**：数据概览、统计、分组聚合、数据清洗、图表绘制（bar/line/pie/hist/scatter）、只读 SQL
- **三段式 Agent 流程**：Planner（规划）→ Executor（Function Calling 工具循环）→ Reporter（报告生成）
- **上下文管理**：工具结果压缩（token 控制）、历史裁剪、重复调用去重
- **防御性设计**：代码越狱拦截、图表状态全局校验（防 LLM 幻觉）、画图步骤强制工具调用（防"口头声称已绘图"）、计划阶段图表要求自动兜底、SQL 只读护栏（禁写 + 自动 LIMIT）、错误回传"可用列"引导自纠
- **记忆**：短期记忆（会话内）+ 长期记忆（跨会话 JSONL 持久化，关键词检索，零外部依赖）
- **反思与动态重规划**：步骤失败自动反思修正；连续失败触发重新规划（有次数上限防失控）
- **流式报告**：报告阶段支持流式生成
- **服务化**：FastAPI REST API + SSE 流式端点 + Dockerfile
- **交互界面**：Streamlit UI（支持上传数据、查看图表/报告/执行轨迹/token 用量）
- **评测体系**：60 题双评测集（30 题回归集 `golden_set_*` + 30 题独立测试集 `test_set_*`，含简单/中等/复杂/对抗四类），标准答案自动比对（数字带 2% 容差）、工具调用约束、可选 LLM-as-judge 语义打分、失败案例自动落盘
- **工程配套**：15 个单元测试（工具层 10 + Agent 主流程 5）、样例数据生成脚本、`.gitignore` / `.dockerignore` / `.env.example`

## 架构

```
用户需求
   │
   ▼
┌──────────────────────────────────────────────────┐
│ core/agent.py                                    │
│  Planner → Executor → (反思/重规划) → Reporter     │
│  记忆注入(recall + short)    流式输出(stream_out)  │
└──────────────────────────────────────────────────┘
   │                    │                    │
   ▼                    ▼                    ▼
core/llm.py        core/registry.py     core/memory.py
LLMClient          工具注册表           短期+长期记忆
重试/用量/流式        │                    (JSONL 持久化)
                    ▼
              core/tools_csv.py / core/tools_sql.py（9 个工具）
                    │
                    ▼
              core/datasource.py（CSV / SQLite 统一接口）
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   data/sample_sales.csv   data/sample_ecommerce.db
   （单表主数据）             （三表业务库）

入口：
  ui/app.py          Streamlit 交互界面
  server/app.py      FastAPI 服务（/analyze、/analyze/stream）
  eval/run_eval.py   自动化评测（csv / sqlite 通用）
```

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖（Python 3.10+）
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux/macOS
pip install -r requirements.txt

# 2. 配置密钥（.env 已被 gitignore，不会入库）
copy .env.example .env           # Windows
# cp .env.example .env           # Linux/macOS
# 编辑 .env，填入 API_KEY / BASE_URL / MODEL

# 3. 生成样例数据（可选，已有内置样例）
python scripts/make_sample_data.py
python scripts/make_sample_db.py

# 4. 运行 Streamlit 界面
streamlit run ui/app.py

# 5. 运行 FastAPI 服务（可选）
uvicorn server.app:app --host 0.0.0.0 --port 8000
# 健康检查: GET http://localhost:8000/health
# 同步分析: POST /analyze   {"query": "哪个地区销售额最高", "kind": "csv"}
# 流式分析: POST /analyze/stream（SSE）
```

### Docker 部署

```bash
docker build -t data-agent .
docker run -p 8000:8000 -v $(pwd)/data:/app/data data-agent
```

## 评测

```bash
# 回归集（开发期快速验证）
python eval/run_eval.py --golden eval/golden_set_sql_biz.jsonl --data data/sample_ecommerce.db
python eval/run_eval.py --golden eval/golden_set_biz.jsonl --data data/sample_sales.csv

# 独立测试集（发版前全量验证，与回归集分离防过拟合）
python eval/run_eval.py --golden eval/test_set_sql.jsonl --data data/sample_ecommerce.db --out eval/test_sql_final3.csv
python eval/run_eval.py --golden eval/test_set_biz.jsonl --data data/sample_sales.csv --out eval/test_csv_final2.csv

# 只跑前 N 条（快速验证）
python eval/run_eval.py --limit 5 ...

# 启用 LLM-as-judge 语义打分（额外消耗 tokens）
python eval/run_eval.py --judge ...
```

**评测指标**：
- `ok`：有标准答案时以答案命中为准（数字 2% 容差）；无标准答案时 must_call（必调工具）+ must_contain（必含信息）双判据
- `tool_success_rate`：工具调用成功率
- `avg_compression_ratio`：工具结果压缩比（原始 token / 压缩后 token）
- `judge_score`：LLM 裁判对报告质量的 0-5 分
- 失败用例自动生成 `eval/failure_report.md`（含调用链与报告节选）

**实测结果**（真实 DeepSeek API，judge 关闭）：
- 回归集：SQL 16/16 通过，CSV 13/14（92.9%）
- 独立测试集：28/30（93.3%），其中对抗/安全类 16/16（100%）正确拒绝危险 SQL、越权工具与提示注入
- 典型失败：报告省略关键数字、全量长跑中规划器退化——已作为已知稳定性边界记录

## 目录结构

```
agent/
├── core/                 # Agent 核心
│   ├── agent.py          # 主流程：规划/执行/反思/重规划/报告
│   ├── config.py         # 配置（LLM / Agent 参数）
│   ├── context.py        # 上下文压缩与历史裁剪
│   ├── datasource.py     # CSV / SQLite 数据源抽象
│   ├── llm.py            # LLMClient（重试/用量/流式）
│   ├── memory.py         # 短期 + 长期记忆
│   ├── refine.py         # 反思与动态重规划
│   ├── registry.py       # 工具注册表与 Function Calling schema
│   ├── tools_csv.py      # CSV 工具实现
│   └── tools_sql.py      # SQL 工具实现
├── data/                 # 样例数据（脚本可重建）
├── eval/                 # golden set + 评测脚本 + 结果
├── scripts/              # 数据生成、MCP 示例
├── server/               # FastAPI 服务
├── tests/                # 15 个单元测试（工具层 10 + Agent 5）
├── ui/                   # Streamlit 界面
├── .env.example          # 密钥模板（.env 不入库）
├── .gitignore / .dockerignore
├── Dockerfile
└── requirements.txt
```

## 常见问题

- **Q: 不配置 API Key 能跑吗？** 单元测试可以（mock LLM），运行界面/评测需要真实 Key。
- **Q: 换了数据集怎么适配？** CSV 单表直接上传即可；SQLite 需要表结构清晰，工具会自动列表现名。记忆检索基于关键词，换库后建议清空 `memory_store.jsonl`。
- **Q: 为什么报告里说"未能生成图表"？** 系统对图表状态做了全局校验：只有真正生成图表才会让模型描述，防止幻觉。如果数据集不适合某种图（如无数值列），模型会诚实说明。

## 已知边界（诚实说明）

- 记忆检索为关键词重叠（2-gram），非向量语义检索；数据量大时建议升级为向量库（如 chromadb/faiss）。
- 评测 must_contain 为字符串包含判据，语义质量依赖可选 LLM-as-judge。
- 示例数据为脚本生成的模拟业务数据，用于演示流程；替换真实数据集即可用于真实分析。
