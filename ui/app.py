# ui/app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, base64, tempfile
import streamlit as st
import pandas as pd

from core.config import LLMConfig, AgentConfig
from core.llm import LLMClient
from core.datasource import CSVDataSource, SQLiteDataSource
from core.agent import run_agent
from core.memory import Memory

st.set_page_config(page_title="数据分析 Agent", page_icon="📊", layout="wide")
st.title("📊 数据分析 Agent")

# ---- state ----
for k, v in {"ds": None, "run": None, "trace": []}.items(): 
    if k not in st.session_state:
        st.session_state[k] = v


def _read_csv_any(file) -> pd.DataFrame:  
    for enc in ("utf-8", "gbk", "gb2312"):
        try:
            file.seek(0)  
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    raise ValueError("无法识别编码")  



# ---- sidebar ----
with st.sidebar:  #“侧边栏”
    st.subheader("数据源")
    mode = st.radio("类型", ["CSV 上传", "SQLite 示例库"], horizontal=False)
    if mode == "CSV 上传":
        f = st.file_uploader("上传 CSV", type=["csv"])
        if f:
            try:
                st.session_state.ds = CSVDataSource(_read_csv_any(f))
                st.session_state.run = None
                st.success(f"{len(st.session_state.ds.get_dataframe())} 行")
            except Exception as e:
                st.error(f"读取失败: {e}")
    else:
        default_db = "data/sample_ecommerce.db"
        #表示“默认示例数据库路径”
        if st.button("加载示例库") and os.path.exists(default_db):
            st.session_state.ds = SQLiteDataSource(default_db)
            st.session_state.run = None
            st.success("已连接 SQLite")
        up = st.file_uploader("或上传 .db", type=["db", "sqlite"])
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            tmp.write(up.read()); tmp.close()
            st.session_state.ds = SQLiteDataSource(tmp.name)
            st.session_state.run = None
            st.success("已连接上传库")

# ---- main ----
query = st.text_area("分析需求", height=90,
    placeholder="例如：分析各地区的销售额排名，找出最好和最差地区并给出建议")

use_mem = st.checkbox("启用记忆（会话 + 跨会话长期记忆）", value=True)

if st.button("🚀 开始分析", type="primary") and query.strip() and st.session_state.ds:

    llm = LLMClient(LLMConfig())
    cfg = AgentConfig()
    trace_lines: list[str] = []


    def _trace(msg: str):
        trace_lines.append(msg)

    with st.spinner("Agent 分析中..."):
        run = run_agent(query, st.session_state.ds, llm, cfg, trace=_trace,
                        memory=Memory() if use_mem else None)

    st.session_state.run = run
    st.session_state.trace = trace_lines

# ---- 结果渲染 ----
if st.session_state.run:
    run = st.session_state.run
    if run.artifacts:
        st.header("📈 图表")
        for a in run.artifacts:
            st.subheader(a["title"])  
            st.image(base64.b64decode(a["image"]))


    if run.report:
        st.header("📝 报告")
        st.markdown(run.report)
        st.download_button("下载 Markdown", run.report,
            file_name="report.md", mime="text/markdown")

    with st.expander("🔍 执行轨迹", expanded=False): 

        st.markdown("\n".join(f"- {t}" for t in st.session_state.trace))
        st.dataframe(pd.DataFrame(run.steps))

        st.caption(f"LLM 调用 {run.usage.get('calls')} 次 | "
                   f"输入 {run.usage.get('prompt_tokens')} tokens | "
                   f"输出 {run.usage.get('completion_tokens')} tokens")
