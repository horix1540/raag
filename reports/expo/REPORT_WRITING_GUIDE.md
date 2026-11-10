# SDMS project report: writing guide

A field-by-field plan for the report form at SDMS (Project 4358 → Reports).
Each field lists **what to cover**, **where it already exists in your own
milestone reports** (`reports/Submitted/`), the **facts to use**, and an
**image** where the form allows one. Write the sentences yourself.

## How to pass the plagiarism check honestly

- SDMS's check is a **similarity check against other submitted project
  reports** (must be under 20%). It is not an AI detector.
- Your own project, described in your own words, is naturally unique.
- **Don't paste milestone paragraphs verbatim.** If your milestone reports
  were also uploaded to SDMS, the checker may match them. Re-explain each
  point in fresh sentences. You understand the project well enough now to do
  this.
- Keep standard definitions short and cite them in section 8. A
  textbook-style definition of "RAG" copied from a website is the most
  likely thing to be flagged.
- **Run Plagiarism Check before submitting**, and rewrite whatever it
  highlights.
- If your college requires you to disclose AI assistance, say how you used
  it (e.g. "code review and result auditing"). Check the policy.

---

## 1. Project Overview

### 1.1 Abstract (≤ 250 words, aim for ~200)
Cover, in this order: the problem → the two-gate idea → how you tested it
→ the key numbers → the main limitation.
- **Facts:**
  - 150 SQuAD 2.0 questions (75 answerable, 75 unanswerable)
  - plain RAG answered 75 of 75 unanswerable questions; the guarded pipeline
    answered 11
  - 38 of the 39 answers it gave were correct
  - Gate B v2 cut over-abstention from 70.7% to 48.0% (paired test,
    p < 0.001)
  - HotpotQA chunking raised full supporting-fact recall from 30.7% to 66.0%
- **Limitation in one line:** the NLI model still rejects some correct
  answers.
- **Source:** M5 §1, rewritten with the corrected numbers.

### 1.2 Problem Statement (≤ 250)
Covered by M1 §2.
- RAG reduces hallucination but doesn't remove it, and the tone is the same
  whether or not the answer is supported.
- The three failure modes:
  - a question outside the corpus
  - invented details
  - a contradiction of the source
- Why a wrong confident answer is worse than abstaining.
- **Image:** none needed.

### 1.3 Objectives (≤ 250)
Numbered list:
1. Score retrieval relevance before generating (Gate A).
2. Check each generated sentence for entailment by the passages (Gate B).
3. Combine both scores into one confidence and abstain, with reasons.
4. Log every decision for monitoring.
5. Evaluate against a plain RAG baseline on SQuAD 2.0 and HotpotQA.
6. Calibrate the thresholds on labelled data.

### 1.4 Scope & Feasibility (≤ 250)
- **In scope:**
  - English text question answering over your own documents (PDF, DOCX,
    Markdown)
  - single-hop questions are the main target
  - multi-hop questions were tested
- **Out of scope:** multilingual, images and tables, fine-tuning any model.
- **Feasibility:**
  - everything runs locally on a laptop GPU (RTX 4060, 8 GB) with free,
    open models
  - no paid APIs needed (Ollama); Groq or Gemini are optional
  - the confidence layer adds well under a second against about 3 s of
    generation

---

## 2. Literature Review & Background

### 2.1 Background & Survey (≤ 400)
Short explanations, each with a citation:
- **RAG** (Lewis et al., 2020)
- **Hallucination** (Ji et al., 2023 survey)
- **NLI** as a faithfulness check: premise → hypothesis, labelled
  entailment / neutral / contradiction
- **Cross-encoder rerankers** (MS MARCO; Bajaj et al., 2016)
- **Selective prediction:** answer-or-abstain and risk–coverage (Geifman &
  El-Yaniv, 2017)

### 2.2 Existing Systems & Limitations (≤ 400)
- **Self-RAG** (Asai et al., 2023): the model reflects on relevance and
  support, but needs a specially trained model.
- **CRAG** (Yan et al., 2024): a retrieval evaluator triggers corrective
  actions, but it is a pre-generation check only.
- **SelfCheckGPT** (Manakul et al., 2023): consistency across several
  samples, which is costly (several LLM calls).
- **SummaC** (Laban et al., 2022) and **AlignScore** (Zha et al., 2023):
  NLI/alignment scoring for factual consistency.
- **RAGAS / DeepEval / LLM-as-judge:** slow, and in your own study they
  agreed with NLI only at chance level.
- **Common gap:** most work does either a pre-check or a post-check, and few
  report the coverage cost honestly against a baseline.

### 2.3 Comparative Study & Proposed Work (≤ 400)
- **Table** comparing the approaches above against yours. Columns:
  - checks before generating?
  - checks after?
  - extra LLM calls?
  - runs locally?
  - explains the abstention?
- **Your position:** an integrated, measured system, combining both checks,
  all-local cheap models, sentence-level reasons, and an honest
  baseline-controlled evaluation. *Say "empirical study", not "new method".*

---

## 3. Requirements Analysis & Specification

### 3.1 Functional Requirements (≤ 400)
Numbered **FR-1 … FR-n**:
- ingest documents and build the index
- retrieve the top-k passages
- Gate A score and decision
- generate with citations, or refuse
- Gate B claim-level scoring
- composite score and verdict (four verdicts)
- show the reasons and the withheld answer
- log telemetry
- dashboard
- baseline mode
- adjustable thresholds

### 3.2 Non-Functional Requirements (≤ 400)
- **Latency:** confidence layer under 1 s.
- **Privacy:** fully local option.
- **Reproducibility:** temperature 0, fixed seeds, scripts.
- **Testability:** 92 unit tests that need no model.
- **Explainability:** per-claim labels.
- **Robustness:** an LLM outage doesn't crash the app; telemetry failures
  never break a query.
- **Usability:** in-app guide.

### 3.3 Hardware & Software Prerequisites (≤ 400)
- **Hardware:** NVIDIA GPU with ≥ 8 GB (tested on an RTX 4060 laptop), 16 GB
  RAM, ~15 GB disk for models.
- **Software:**
  - Python 3.13 with a venv
  - Ollama with `llama3.1:8b`
  - `requirements.txt`: LangChain, ChromaDB, sentence-transformers,
    Streamlit, scikit-learn
  - Windows 11 tested
- **Image:** none, or a small table.

---

## 4. System Design & Architecture

### 4.1 Architecture Description (≤ 400)
- **Image:** the pipeline flow from the poster (crop
  `reports/expo/Poster_A3_preview.png`) or `assets/thumbnail.png`.
- Walk through it: Retrieve → Gate A → Generate → Gate B → Decide.
- Why two different models: ranking vs entailment (M1 §3.1).
- Why the harmonic mean (M1 §3.3): 0.95 and 0.05 → about 0.08, not 0.50.

### 4.2 System Modules & Workflow (≤ 400)
Module table (M1 §4):
| Module | Role |
|---|---|
| `ingest.py`, `embeddings.py`, `retriever.py` | data layer |
| `scorers.py` | reranker and NLI |
| `confidence.py` | the gates, composite, verdicts |
| `generator.py`, `llm.py` | generation |
| `pipeline.py` | orchestration |
| `store.py` | SQLite telemetry |
| `app.py`, `pages/` | UI |

Then the four verdicts and when each happens (M1 §3.4 table).

### 4.3 Database & Data Flow Design (≤ 400)
- **ChromaDB vector store:**
  - one collection
  - chunks of ~800 characters with 120 overlap
  - IDs based on the chunk content, so re-indexing creates no duplicates
- **SQLite `queries` table:** one row per question, about 36 columns,
  including scores, verdict, per-claim JSON, latencies, run tag, arm and the
  thresholds used.
- **Data-flow image:** question → retrieval → scores → verdict → row →
  dashboard.

---

## 5. Technology Stack & Implementation

### 5.1 Technology Stack & Tools (≤ 400)
Table: component → tool → why.
| Component | Tool |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (384-dim) |
| Vector store | ChromaDB |
| Gate A | `ms-marco-MiniLM-L-6-v2` reranker (sigmoid; mean of top 2) |
| Gate B | `nli-deberta-v3-base` |
| LLM | `llama3.1:8b` via Ollama |
| LLM judge | DeepEval with gemma2 |
| UI | Streamlit |
| Telemetry | SQLite |
| Quality | pytest, ruff |
| Version control | Git and GitHub |

### 5.2 Implementation Details & Modules (≤ 400)
The interesting decisions:
- **Claim splitting:** sentence-level.
- **Contradiction veto:** absolute cap of 0.15, decoupled from the
  threshold (M2 §5.1).
- **Contradiction attribution:** only the passage that best supports a claim
  can veto it.
- **Generator self-refusal marker:** `INSUFFICIENT_CONTEXT`.
- **Gate B v2:** each claim is scored against whole passages *and*
  3-sentence windows, and the best support wins.
- **Images:** app screenshots from `reports/submission_screenshots/` (the
  grounded answer and the dashboard), or new screenshots of the redesigned
  app.

---

## 6. Testing & Results

### 6.1 Testing Methodology & Test Cases (≤ 400)
- **Unit tests:** 92, no models needed (scoring maths, decision table,
  metrics, statistics).
- **Smoke test:** 24 checks (indexing → telemetry).
- **Benchmarks:**
  - SQuAD 2.0: 150 questions, gold paragraph + 4 distractors, one fresh
    index per question
  - HotpotQA: 150 questions, sentence vs chunk indexing
- **Two arms:** guarded vs plain baseline.
- **Metrics, defined:**
  - hallucination rate = answered an unanswerable question
  - over-abstention = refused an answerable question
  - selective accuracy
  - AUROC
- **Sample test cases** (3–4 rows: input, expected, actual):
  - the five-stages question → answered
  - the Amazon rainfall question → Gate A refuses
  - a fabricated answer → contradiction caught

### 6.2 Results & Outputs (≤ 400)
- **Main table** (fresh runs, 150 SQuAD questions):
  | | Plain RAG | Gate B v1 | Gate B v2 |
  |---|---|---|---|
  | Correct answerable | 68 | 21 | 38 |
  | Hallucinations (of 75) | 75 | 9 | 11 |
  | Over-abstention | 0% | 70.7% | 48.0% |
- **Selective accuracy:** 38/39 = 97%.
- **Where refusals of unanswerable questions came from** (M3 run): Gate A 33,
  generator self-refusal 27, Gate B 7.
- **HotpotQA:**
  - full supporting-fact recall 30.7% → 66.0%
  - baseline accuracy 43% → 59%
  - guarded error rate among answered questions 41% → 24%, at 22% coverage
- **Image:** the results chart from the poster.

### 6.3 Performance Analysis (≤ 400)
- **Latency:**
  - Gate A ≈ 16 ms
  - Gate B v1 ≈ 94–118 ms; Gate B v2 ≈ 400 ms
  - generation ≈ 3 s
- **Statistics:**
  - Wilson intervals (v2 hallucination 8.4–24.4%)
  - McNemar: v1 → v2 coverage gain p < 0.001; hallucination change p = 0.5
- **Calibration:**
  - Gate A AUROC 0.65
  - threshold stable (0.45 chosen in 185 of 200 held-out splits)
- **Judge study:** κ ≈ 0, with the local scorer about 860× faster.
- **Honest note:** the earlier citation-scoring bug and how you found it.
  Examiners value that.

---

## 7. Conclusion & Future Scope

### 7.1 Key Achievements (≤ 250)
- A working two-gate system with app, dashboard and tests.
- A 7× reduction in hallucinated answers versus plain RAG.
- Gate B v2 recovered 17 correct answers that v1 wrongly refused (p < 0.001).
- An audit that found and fixed the evaluation bugs.

### 7.2 Limitations & Challenges (≤ 250)
- The NLI model rejects some correct answers (the "False alarm" demo
  question).
- The prompt and the gates are not separated in the comparison.
- 150-question samples.
- Calibration used a proxy label.
- Multi-hop answers are hard for single-premise NLI.
- An 8 GB GPU is tight.

### 7.3 Future Scope (≤ 250)
- A stronger NLI model (DeBERTa-large is being tested).
- A prompt-only control arm.
- 500+ questions with held-out calibration.
- Multi-passage entailment and claim decomposition.
- A larger demo corpus.

### 7.4 Conclusion (≤ 200)
Three or four sentences: what you set out to do, what the evidence shows,
and what it costs.

---

## 8. References (≤ 400)
Use one citation style throughout (e.g. IEEE). These are real papers; check
each on Google Scholar before you list it.
- Lewis et al., 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Rajpurkar, Jia, Liang, 2018. Know What You Don't Know: Unanswerable Questions for SQuAD. ACL.
- Yang et al., 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP.
- Niu et al., 2024. RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. ACL.
- Asai et al., 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR.
- Yan et al., 2024. Corrective Retrieval Augmented Generation. arXiv:2401.15884.
- Manakul, Liusie, Gales, 2023. SelfCheckGPT. EMNLP.
- Laban et al., 2022. SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization. TACL.
- Zha et al., 2023. AlignScore: Evaluating Factual Consistency with a Unified Alignment Function. ACL.
- Ji et al., 2023. Survey of Hallucination in Natural Language Generation. ACM Computing Surveys.
- Geifman & El-Yaniv, 2017. Selective Classification for Deep Neural Networks. NeurIPS.
- He et al., 2021. DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR.
- Reimers & Gurevych, 2019. Sentence-BERT. EMNLP.
- Bajaj et al., 2016. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset (arXiv:1611.09268).

## 9. Annexure (up to 5 links)
1. **GitHub repository:** github.com/Sumwanyuno/confidence-aware-rag. It is
   currently **private**, so evaluators can't open it. Either make it public
   or add your faculty as collaborators.
2. **Demo video:** screen-record the 5-minute demo (Windows: Win+Alt+R with
   Xbox Game Bar), upload it as an unlisted YouTube video or to Google Drive,
   and link it.
3. **Poster (PDF)**, uploaded to Drive.
4. **Presentation (PPT)**, uploaded to Drive.
5. **Datasets:** the SQuAD 2.0 and HotpotQA official pages.
