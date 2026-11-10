# Confidence-Aware RAG Pipeline with Hallucination Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CUDA 12.8](https://img.shields.io/badge/PyTorch-CUDA%2012.8%20%7C%20sm__120-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-92%20passed%20(100%25)-success.svg)](tests/)
[![Streamlit Studio](https://img.shields.io/badge/UI-10%2F10%20Enterprise%20Studio-4F46E5.svg)](app.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-grade, selective retrieval-augmented generation (RAG) system that **knows when it does not know**. Rather than blindly producing fluent hallucinations when retrieval misses or when facts conflict, this architecture places two inline verification gates around generation and selectively abstains whenever supporting evidence is insufficient.

> **GitHub Repository:** [https://github.com/horix1540/raag](https://github.com/horix1540/raag)  
> **Evaluation Artifacts & Audited Reports:** `reports/audit_m5.json` · `reports/calibration.json` · `reports/PROJECT_REVIEW.md`

---

## Key System Architecture

Conventional RAG pipelines evaluate output offline or rely on uncalibrated LLM-as-a-judge prompts. This system operates **inline and pre-generation**, dividing verification into two distinct checkpoints calibrated on SQuAD 2.0 and HotpotQA benchmarks:

```
Query
  │
  ├─► 01. RETRIEVE · Bi-encoder dense passage retrieval (all-MiniLM-L6-v2)
  │
  ├─► 02. GATE A · Pre-Generation Sufficiency (ms-marco-MiniLM-L-6-v2 cross-encoder)
  │         Threshold: ≥ 0.45 (mean top-2 sigmoid relevance)
  │         Fail ────────────────────────► ABSTAIN (0 LLM tokens spent; ~15 ms)
  │
  ├─► 03. GENERATE · Context-restricted synthesis with citation tags [1], [2]
  │         Model self-reports INSUFFICIENT_CONTEXT ──► ABSTAIN
  │
  ├─► 04. GATE B v2 · Post-Generation Claim NLI Groundedness (DeBERTa-v3-base)
  │         Threshold: ≥ 0.25 (mean entailment across atomic claims)
  │         Premise: Passages + overlapping 3-sentence windows + multi-hop top-2 pairs
  │         Hard Contradiction Veto: Any Evident Conflict (p_contra ≥ 0.50) caps score at 0.15
  │         Fail ────────────────────────► ABSTAIN (draft answer suppressed)
  │
  └─► 05. DECIDE · Weighted Harmonic Composite: 1 / (0.4 / A + 0.6 / B)
            Below warning threshold (0.33) ──► ANSWER SHOWN (Flagged Low Confidence)
            Both gates pass (≥ 0.33) ────────► ANSWER SHOWN (Verified & Grounded)
```

### Why Two Gates?
- **Gate A (Pre-Generation):** Evaluates whether the retrieved passages actually contain sufficient information *before* calling the generator. On unanswerable queries, Gate A rejects in ~15 ms, saving 3–6 seconds of LLM compute and preventing the LLM from hallucinating answers from parametric memory.
- **Gate B v2 (Post-Generation):** Faithfulness cannot be evaluated before an answer exists. Once drafted, the answer is split into atomic claims and tested against passages and 3-sentence sliding windows using natural language inference (NLI).

### Why Weighted Harmonic Mean?
An arithmetic mean between high retrieval relevance (e.g. 0.95) and catastrophic hallucination (e.g. 0.05) produces an acceptable-looking score of `0.4·(0.95) + 0.6·(0.05) = 0.41` (which would clear a 0.33 threshold). The **weighted harmonic mean** collapses immediately if *either* gate approaches zero:
$$\text{Composite} = \frac{1}{\frac{0.4}{A} + \frac{0.6}{B}} = \frac{1}{\frac{0.4}{0.95} + \frac{0.6}{0.05}} = \frac{1}{0.421 + 12.0} \approx 0.080 \quad (\text{Flagged / Withheld})$$

---

## 3 Experimental Arms (Ablation Study)

To address **Project Review Logical Hole H3** (isolating prompt refusal behavior from gate enforcement), the system provides 3 distinct execution arms:

1. **Guarded Arm (2-Gate Safety):** Active 2-gate pipeline. Gate A short-circuits out-of-scope queries; Gate B suppresses ungrounded or contradicted answers.
2. **Prompt-Only Arm (H3 Ablation):** The generator runs with the strict system prompt (`INSUFFICIENT_CONTEXT` self-refusal instructions and inline citation format), and both Gate A and Gate B are measured, but the gates are **not enforced**. This isolates the prompt's contribution from the gates' mathematical contribution.
3. **Baseline Control Arm (Unguarded):** Conventional RAG baseline without gates or abstention instructions. Answers 100% of queries, demonstrating the uncalibrated hallucination rate.
4. **Side-by-Side Mode:** Executes Guarded and Baseline simultaneously in the UI to visualize hallucination prevention in real time.

---

## Audited Milestone 1–5 Benchmark Results

From the audited benchmark runs on SQuAD 2.0 ($n = 150$, 75 answerable / 75 unanswerable) and HotpotQA ($n = 150$ multi-hop) recorded in [`reports/audit_m5.json`](file:///d:/projects/confidence-aware-rag-main/reports/audit_m5.json):

### 1. SQuAD 2.0 Benchmark ($n = 150$)

| Configuration / Arm | Hallucination Rate (Unanswerable) | Over-Abstention (Answerable) | Answered (Answerable) | Selective Accuracy (Gold in Answered) | Corrected F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Control (Unguarded)** | 100.0% (75/75) | 0.0% (0/75) | 75 / 75 | 90.7% (68/75) | 0.251 |
| **Guarded M2 (Uncalibrated)** | 4.0% (3/75) | 84.0% (63/75) | 12 / 75 | 91.7% (11/12) | 0.511 |
| **Guarded M3 (Gate B v1 · Full Passage)** | 10.7% (8/75) | 72.0% (54/75) | 21 / 75 | 95.2% (20/21) | 0.505 |
| **Guarded M5 (Gate B Window-Only)** | 14.7% (11/75) | 50.7% (38/75) | 37 / 75 | 97.3% (36/37) | 0.543 |
| **Guarded M5 Final (Gate B v2 · Passages + 3-Sent Windows)** | **14.7% (11/75)** | **48.0% (36/75)** | **39 / 75** | **97.4% (38/39)** | **0.553** |

> **Key Milestone 5 Breakthrough:** Switching Gate B from full-paragraph premises (v1) to passages + overlapping 3-sentence windows (v2) cut over-abstention from **70.7% to 48.0%** (paired McNemar $p < 0.001$) while raising selective accuracy to **97.4%** (38/39 correct).

### 2. Local DeBERTa NLI vs. LLM-as-a-Judge (`reports/judge_agreement.json`)
- **Latency:** Local DeBERTa NLI Gate B ran in **261.7 ms** median vs. **224,372.8 ms** for local DeepEval LLM judge (~860× faster).
- **Agreement:** Cohen's kappa was near-zero ($\kappa = -0.037$), demonstrating that small local LLM judges emit inconsistent yes/no labels and malformed JSON, whereas local cross-encoders provide deterministic, continuous probability distributions.

---

## 10/10 Enterprise UI Studio Features

The Streamlit interface (`app.py`, `pages/1_How_to_use.py`, `pages/2_Monitoring_Dashboard.py`) includes:

- **Enterprise Design System:** Custom theme with Indigo/Slate styling, Plus Jakarta Sans, Inter, and JetBrains Mono typography.
- **5-Stage Pipeline Telemetry Stepper:** Displays millisecond latency, chunk counts, and live stage status badges (`PASSED`, `FAILED`, `SKIPPED`, `UNFORCED`).
- **Real-Time KPI Cards & Gauge Bars:** Gate A relevance, Gate B faithfulness, composite harmonic mean, and end-to-end wall time with calibrated threshold lines.
- **Claim-Level NLI Breakdown:** Categorizes answer sentences under the **RAGTruth taxonomy** (*Supported*, *Baseless Information*, *Evident Conflict*).
- **Inline Citation Pills & Passage Highlighting:** Converts raw `[1]`, `[2]` citation markers into styled `Ref 1` badges and dynamically highlights supporting sentences in retrieved passages.
- **1-Click "Test LLM Connection":** Validates generator connectivity (`src.llm.health_check()`) without crashing queries.
- **Live Corpus Ingestion Drawer:** Drag-and-drop `.pdf`, `.docx`, `.md`, or `.txt` files in the sidebar with 1-click ChromaDB chunking and embedding.
- **Gate B v2 Advanced Settings:** Live switching between premise modes (`both`, `multi_hop`, `window`, `passage`) and hypothesis modes (`claim`, `claim_clean`, `qa_short`, `qa`).
- **Monitoring & Audit Dashboard:** Live SQLite query log (`data/runs.sqlite3`), verdict distribution charts, confidence trajectories, and the M1–M5 benchmark audit data.

---

## Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/horix1540/raag.git
cd raag

# Create Python 3.12 virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows; source .venv/bin/activate on Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

> **GPU Support:** For NVIDIA RTX 40/50 series GPUs (Blackwell/Ada Lovelace), install PyTorch with CUDA 12.8:  
> `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`

### 2. Environment Configuration

Copy the example environment file and configure your API keys (optional if using local Ollama):

```bash
cp .env.example .env
```

| Key | Default | Description |
|:---|:---|:---|
| `RAG_PROVIDER` | `ollama` | LLM generator provider: `ollama`, `groq`, or `gemini` |
| `RAG_SCORER` | `nli` | Confidence scorer backend: `nli` (local cross-encoders), `llm`, or `both` |
| `GOOGLE_API_KEY` | — | Google Gemini API key (if `RAG_PROVIDER=gemini`) |
| `GROQ_API_KEY` | — | Groq API key (if `RAG_PROVIDER=groq`) |

### 3. Build Knowledge Base & Run Smoke Tests

```bash
# Index default course notes into ChromaDB
python scripts/build_index.py --reset

# Run model-free component smoke test (24 checks, no LLM needed)
python scripts/smoke_test.py
```

### 4. Run Unit Tests (100% Passing)

```bash
python -m pytest tests/ -q -p no:cacheprovider
# 92 passed in ~5.4s
```

### 5. Launch the Enterprise Studio

```bash
streamlit run app.py
```

The web studio will open at `http://localhost:8501`.

---

## Repository Structure

```
├── app.py                          # 10/10 RAG Verification Studio (Streamlit frontend)
├── .streamlit/
│   └── config.toml                 # Enterprise Indigo/Slate theme configuration
├── pages/
│   ├── 1_How_to_use.py             # Architecture, Interactive Viva Lab & Defense Guide
│   └── 2_Monitoring_Dashboard.py   # Real-time SQLite telemetry & M1–M5 Benchmark Audit
├── src/
│   ├── config.py                   # Central settings: thresholds, weights, model configs
│   ├── embeddings.py               # Bi-encoder embeddings with automatic CUDA/CPU fallback
│   ├── ingest.py                   # Document loading, markdown header splitting, ChromaDB indexing
│   ├── retriever.py                # Bi-encoder dense vector retrieval with cosine conversion
│   ├── generator.py                # Context-restricted generation prompt & unguarded baseline
│   ├── scorers.py                  # Gate A (ms-marco cross-encoder) & Gate B v2 (DeBERTa NLI)
│   ├── confidence.py               # Weighted harmonic mean, decision matrix, RAGTruth taxonomy
│   ├── pipeline.py                 # Pipeline orchestration: Guarded, Prompt-Only, & Baseline arms
│   ├── store.py                    # SQLite query telemetry database (data/runs.sqlite3)
│   ├── ui.py                       # Enterprise design system, telemetry stepper, gauge bars, styling
│   └── llm.py                      # Multi-provider client (Ollama, Groq, Gemini) with health_check
├── scripts/
│   ├── build_index.py              # ChromaDB index builder
│   ├── smoke_test.py               # Fast 24-step integration test
│   ├── run_benchmark.py            # Resumable evaluation harness over SQuAD 2.0 / HotpotQA
│   ├── eval_benchmark.py           # Selective accuracy, EM/F1, over-abstention evaluation
│   ├── calibrate.py                # Fits Youden's J Gate A and F1 Gate B thresholds
│   └── risk_coverage.py            # Generates AURC curves and risk-coverage plots
├── tests/                          # 92 comprehensive unit tests covering all math & pipeline logic
└── reports/
    ├── audit_m5.json               # Full audited M1-M5 benchmark metrics
    ├── calibration.json            # Calibrated thresholds and ROC/PR data
    └── PROJECT_REVIEW.md           # Formal academic review & defense against logical holes
```

---

## Citation & Academic License

This project was developed as a research and engineering showcase for reliable AI systems and selective prediction in retrieval-augmented architectures. Released under the [MIT License](LICENSE).
