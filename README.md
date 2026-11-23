# ⚖️ LLM Evaluation Harness & Prompt Benchmarking Tool

A production-ready Python evaluation harness and Streamlit dashboard for benchmarking LLMs and prompt templates head-to-head on **latency**, **token cost**, and **LLM-as-a-Judge accuracy**.

## ✨ Key Features
- **Local & Cloud Model Support:** Compare local **Ollama** models (`llama3`, `qwen2:0.5b`, `mistral`) against each other or against any **OpenAI-compatible API** (OpenAI, Groq, OpenRouter).
- **LLM-as-a-Judge Rubric:** Automatically scores responses on a 1–5 scale for **Accuracy**, **Completeness**, and **Conciseness** with structured JSON output and a deterministic lexical fallback.
- **Built-in 50-Prompt Dataset:** Includes `data/benchmark_50.json` spanning Coding, Reasoning, Extraction, SQL, and Summarization, plus support for custom JSON uploads.
- **Executive Dashboard & Persistence:** Tracks P95 latency, win-rate splits, and estimated USD cost in SQLite, with one-click CSV and JSON exports.

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Model Providers** | Local Ollama REST API & OpenAI-Compatible APIs (`requests`) |
| **Evaluation Engine** | LLM-as-a-Judge (JSON Rubric) + Lexical Fallback |
| **Data & Metrics** | Pandas, NumPy |
| **Database** | SQLite (`sqlite3`) |
| **Deployment** | Docker |

---

## 📁 Project Structure

```text
llm-eval-harness/
├── app.py                   # Streamlit benchmarking dashboard
├── generate_dataset.py      # Generates the 50-prompt evaluation dataset
├── core/
│   ├── __init__.py
│   ├── providers.py         # Ollama + OpenAI-compatible API clients & cost tracking
│   ├── judge.py             # LLM-as-a-Judge 1-5 rubric scorer
│   ├── runner.py            # Benchmark orchestrator & summary aggregator
│   └── database.py          # SQLite persistence for runs & results
├── data/
│   └── benchmark_50.json    # 50-prompt test dataset
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md

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
