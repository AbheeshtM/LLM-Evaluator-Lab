# LLM Reliability Lab

A reproducible evaluation framework to measure **accuracy, confidence calibration, hallucinations, bias, and robustness** of Large Language Models (LLMs) on structured tabular data.

This project focuses on **evaluation, not fine-tuning**, and is designed to benchmark and compare multiple LLMs under identical conditions.

---

## 🚀 What This Project Does

Given a dataset and a fixed decision prompt, the system:

- Runs LLM inference on sampled data
- Parses predictions and self-reported confidence
- Measures:
  - Accuracy
  - High-confidence errors (hallucinations)
  - Bias across demographic groups
  - Confidence calibration
- Supports **resumable pipelines** via artifact-based caching
- Enables **fair comparison across multiple LLMs**

---

## 🧠 Key Concepts Evaluated

- **Accuracy** – correctness of predictions
- **Hallucinations** – confident but incorrect predictions
- **Bias** – performance differences across sensitive attributes (e.g. gender)
- **Calibration** – relationship between confidence and correctness
- **Reproducibility** – identical pipeline across models

---

## 📂 Project Structure

LLM Evaluation/
├── experiments/ # Sampling, baseline inference, bias, robustness
├── evaluation/ # Parsing, confidence extraction, metrics
├── llm/ # LLM client + rate limiting
├── prompts/ # Decision prompt
├── data/ # Raw and processed datasets
├── results/ # Evaluation artifacts (CSV outputs)
├── app/ # Streamlit dashboard (UI only)
├── run_all.py # Resumable pipeline orchestrator
├── README.md
└── LICENSE


---

## ⚙️ Setup

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
setx GROQ_API_KEY your_api_key_here
(Optional, for model comparison)
setx GROQ_MODEL llama-3.1-8b-instant
▶️ How to Run the Full Pipeline

From project root:

python run_all.py


The pipeline is resumable:

If a step already produced its output file, it will be skipped automatically.

If the pipeline fails, re-running run_all.py resumes from the last incomplete step.

🔁 Comparing Multiple LLMs

Run pipeline with Model A

Move results into a model-specific folder

Change model via environment variable

Run pipeline again

Compare results

Example:

setx GROQ_MODEL llama-3.1-8b-instant
python run_all.py

setx GROQ_MODEL qwen/qwen3-32b
python run_all.py

python experiments/compare_models.py

📊 Dashboard (UI Only)

The Streamlit app visualizes precomputed results only.

streamlit run app/dashboard.py


No live LLM calls are made in the UI.

📌 Design Principles

Same data + same prompt = fair comparison

Evaluation is separated from visualization

Confidence parsing is centralized

Pipelines are resumable and reproducible

LLMs are evaluated, not trusted blindly

🔮 Future Improvements

Multi-dataset support via config files

Robustness comparison across models

Cost and latency benchmarking

Streamlit-based model comparison view
