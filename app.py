import json
from pathlib import Path
import pandas as pd
import streamlit as st

from core.database import EvalDB
from core.providers import ModelConfig
from core.runner import BenchmarkRunner

st.set_page_config(page_title="LLM Eval Harness & Benchmark", page_icon="⚖️", layout="wide")
db = EvalDB()

st.title("⚖️ LLM Evaluation Harness & Prompt Benchmarking Tool")
st.write("Compare two models or prompt templates on latency, token cost, and LLM-as-a-Judge accuracy.")

tab_run, tab_history = st.tabs(["Run Benchmark", "Run History"])

with tab_run:
    # --- Dataset Selection ---
    st.subheader("1. Evaluation Dataset")
    uploaded = st.file_uploader("Upload JSON dataset (or leave empty to use built-in 50-prompt set)", type=["json"])
    if uploaded:
        dataset = json.load(uploaded)
    else:
        default_path = Path("data/benchmark_50.json")
        if default_path.exists():
            dataset = json.loads(default_path.read_text(encoding="utf-8"))
        else:
            st.warning("Run `python generate_dataset.py` first to create the 50-prompt dataset.")
            st.stop()

    sample_size = st.slider("Number of prompts to evaluate", min_value=1, max_value=len(dataset), value=min(10, len(dataset)))
    active_dataset = dataset[:sample_size]

    # --- Model Configurations ---
    st.subheader("2. Configure Models & Judge")
    c_a, c_b, c_j = st.columns(3)

    with c_a:
        st.markdown("**Model A (Baseline)**")
        a_provider = st.selectbox("Provider A", ["ollama", "openai_compat"], key="ap")
        a_model = st.text_input("Model ID A", value="llama3", key="am")
        a_url = st.text_input("Base URL A", value="http://localhost:11434" if a_provider == "ollama" else "https://api.openai.com/v1", key="au")
        a_key = st.text_input("API Key A", type="password", disabled=(a_provider == "ollama"), key="ak")
        a_sys = st.text_area("System Prompt A", value="You are a concise, accurate assistant.", height=70, key="as")

    with c_b:
        st.markdown("**Model B (Challenger)**")
        b_provider = st.selectbox("Provider B", ["ollama", "openai_compat"], key="bp")
        b_model = st.text_input("Model ID B", value="qwen2:0.5b", key="bm")
        b_url = st.text_input("Base URL B", value="http://localhost:11434" if b_provider == "ollama" else "https://api.openai.com/v1", key="bu")
        b_key = st.text_input("API Key B", type="password", disabled=(b_provider == "ollama"), key="bk")
        b_sys = st.text_area("System Prompt B", value="Answer directly in one short sentence.", height=70, key="bs")

    with c_j:
        st.markdown("**LLM-as-a-Judge**")
        j_provider = st.selectbox("Judge Provider", ["ollama", "openai_compat"], key="jp")
        j_model = st.text_input("Judge Model ID", value="llama3", key="jm")
        j_url = st.text_input("Judge Base URL", value="http://localhost:11434" if j_provider == "ollama" else "https://api.openai.com/v1", key="ju")
        j_key = st.text_input("Judge API Key", type="password", disabled=(j_provider == "ollama"), key="jk")

    if st.button("🚀 Start Benchmark", type="primary", use_container_width=True):
        cfg_a = ModelConfig(name=f"A: {a_model}", provider=a_provider, model_id=a_model, base_url=a_url, api_key=a_key, system_prompt=a_sys)
        cfg_b = ModelConfig(name=f"B: {b_model}", provider=b_provider, model_id=b_model, base_url=b_url, api_key=b_key, system_prompt=b_sys)
        cfg_j = ModelConfig(name=f"Judge: {j_model}", provider=j_provider, model_id=j_model, base_url=j_url, api_key=j_key)

        progress = st.progress(0)
        status_text = st.empty()

        def on_progress(curr: int, total: int, msg: str) -> None:
            progress.progress(curr / total)
            status_text.text(msg)

        runner = BenchmarkRunner(cfg_a, cfg_b, cfg_j)
        output = runner.run(active_dataset, progress_cb=on_progress)
        db.save_run(output["summary"], output["results"])
        status_text.success("Benchmark complete and saved to history!")

        s = output["summary"]
        df = pd.DataFrame(output["results"])

        # --- KPI Cards ---
        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Avg Score ({s['model_a']})", f"{s['avg_score_a']} / 5")
        k2.metric(f"Avg Score ({s['model_b']})", f"{s['avg_score_b']} / 5", delta=round(s["avg_score_b"] - s["avg_score_a"], 2))
        k3.metric("P95 Latency (A vs B)", f"{s['p95_latency_a_ms']}ms / {s['p95_latency_b_ms']}ms")
        k4.metric("Win Split (A / B / Tie)", f"{s['wins_a']} / {s['wins_b']} / {s['ties']}")

        # --- Charts ---
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Score Comparison per Prompt**")
            st.bar_chart(df[["score_a", "score_b"]])
        with ch2:
            st.markdown("**Latency Comparison (ms)**")
            st.line_chart(df[["latency_a_ms", "latency_b_ms"]])

        # --- Detailed Results Table & Export ---
        st.subheader("Detailed Prompt-Level Results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button("Download Results (CSV)", df.to_csv(index=False).encode("utf-8"), "benchmark_results.csv", "text/csv")
        d2.download_button("Download Report (JSON)", json.dumps(output, indent=2), "benchmark_report.json", "application/json")

with tab_history:
    runs = db.fetch_runs()
    if not runs:
        st.info("No benchmark runs saved yet.")
    else:
        st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)
