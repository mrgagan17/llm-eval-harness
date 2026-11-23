# ⚖️ LLM Evaluation Harness & Prompt Benchmarking Tool

A production-ready Python evaluation harness and Streamlit dashboard for benchmarking LLMs and prompt templates head-to-head on **latency**, **token cost**, and **LLM-as-a-Judge accuracy**.

## ✨ Key Features
- **Local & Cloud Model Support:** Compare local **Ollama** models (`llama3`, `qwen2:0.5b`, `mistral`) against each other or against any **OpenAI-compatible API** (OpenAI, Groq, OpenRouter).
- **LLM-as-a-Judge Rubric:** Automatically scores responses on a 1–5 scale for **Accuracy**, **Completeness**, and **Conciseness** with structured JSON output and a deterministic lexical fallback.
- **Built-in 50-Prompt Dataset:** Includes `data/benchmark_50.json` spanning Coding, Reasoning, Extraction, SQL, and Summarization, plus support for custom JSON uploads.
- **Executive Dashboard & Persistence:** Tracks P95 latency, win-rate splits, and estimated USD cost in SQLite, with one-click CSV and JSON exports.

## 🚀 Quick Start
```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## 🐳 Docker
```bash
docker build -t llm-eval-harness .
docker run -p 8501:8501 llm-eval-harness
```
