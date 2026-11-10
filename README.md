# Confidence-Aware RAG Pipeline with Hallucination Detection

A retrieval-augmented generation system that **knows when it does not know**. Instead of
always producing a fluent answer, it scores retrieval reliability before generating and
answer groundedness after, and abstains when either check fails.

> Semester project (Jul–Dec 2026) · Artificial Intelligence & Machine Learning Systems

---

## The problem

A conventional RAG pipeline always answers. Vector search returns its top-k nearest
neighbours whether or not any of them are relevant, and a language model handed weak
context will still produce a confident, well-formed, wrong answer. Nearest is not the
same as sufficient.

## The approach: two gates, not one score

Most RAG evaluation happens offline, after the fact. This pipeline puts the check
*inline*, and splits it in two because the two questions are answerable at different
times:

```
question
   │
   ├─► retrieve top-k passages
   │
   ├─► GATE A · retrieval sufficiency         ← before generation
   │     cross-encoder relevance (ms-marco)
   │     fail ─────────────────────────────► abstain, no tokens spent
   │
   ├─► generate answer (context-restricted prompt)
   │
   ├─► GATE B · answer groundedness           ← after generation
   │     per-claim NLI entailment (DeBERTa)
   │     fail ─────────────────────────────► abstain, answer suppressed
   │
   └─► composite = weighted harmonic mean(A, B)
         below threshold ──────────────────► answer shown, flagged low-confidence
         otherwise ───────────────────────► answer shown
```

**Why two gates.** Faithfulness is undefined until an answer exists, so it cannot guard
generation. Retrieval relevance needs only the query and the passages, so it can — and
catching an unanswerable question at Gate A means the wrong answer is never produced at
all, rather than produced and then filtered.

**Why a harmonic mean.** Relevancy 0.95 with faithfulness 0.05 averages to a
respectable-looking 0.50. The weighted harmonic mean returns about 0.09. Both conditions
have to hold, the same way F1 refuses to reward precision without recall.

**Claim-level output.** Gate B splits the answer into claims and labels each one using
the RAGTruth taxonomy — *supported*, *baseless information* (the source is silent), or
*evident conflict* (the source disagrees). Any single contradiction fails the gate
outright, because averaging lets nine correct sentences hide one that conflicts.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
cp .env.example .env
```

Index the sample corpus and verify every component:

```bash
python scripts/build_index.py --reset
python scripts/smoke_test.py
```

`smoke_test.py` needs **no LLM** — retrieval and both confidence gates run entirely on
local cross-encoders. That is the point: the confidence layer is independently testable.

To enable generation, install [Ollama](https://ollama.com/download) and pull a model:

```bash
ollama pull llama3.1:8b
python scripts/smoke_test.py --with-llm
```

Run the benchmark (both arms over the labelled SQuAD 2.0 sample; resumable —
completed `(item_id, arm)` pairs are skipped on restart):

```bash
python scripts/run_benchmark.py                 # 150 items × guarded + baseline
python scripts/eval_benchmark.py                # EM/F1, hallucination rate, per arm
python scripts/calibrate.py                     # fit thresholds -> reports/calibration.json
python scripts/risk_coverage.py                  # AURC + figures in reports/figures/
python scripts/run_benchmark.py --tag squad2-judge --limit 50 --scorer both --arm guarded
python scripts/judge_agreement.py               # NLI vs LLM judge, correlation + kappa
python scripts/build_hotpot.py                  # prepare the HotpotQA sample
python scripts/run_retrieval_hotpot.py           # sentence vs. chunk retrieval
python scripts/run_generation_hotpot.py          # guarded vs. baseline generation
```

Launch the app:

```bash
streamlit run app.py
```

---

## Repository layout

| Path | What it does |
| --- | --- |
| `src/config.py` | Every tunable in one place: thresholds, weights, models, devices |
| `src/embeddings.py` | Single embedding entry point, with CUDA→CPU fallback |
| `src/ingest.py` | load → chunk → embed → persist to Chroma, with provenance metadata |
| `src/retriever.py` | Retrieval, plus explicit Chroma-distance → similarity conversion |
| `src/generator.py` | Context-restricted prompt; `generate_unguarded` is the control arm |
| `src/scorers.py` | Three scoring backends: reranker (Gate A), NLI (Gate B), DeepEval judge |
| `src/confidence.py` | Harmonic mean, gate thresholds, decision table, RAGTruth labels |
| `src/pipeline.py` | The orchestration the architecture diagram describes |
| `src/store.py` | SQLite telemetry backing the monitoring dashboard |
| `src/squad_metrics.py` | Official SQuAD 2.0 EM/F1 normalisation + no-answer convention |
| `src/evalstats.py` | Risk–coverage curves, AURC, judge-agreement statistics |
| `app.py` | Streamlit Q&A interface showing both gate scores and per-claim analysis |
| `pages/1_How_to_use.py` | In-app guide: quick start, reading a result, experiments, troubleshooting, glossary |
| `pages/2_Monitoring_Dashboard.py` | Confidence trends, verdict mix, claims, latency budget, arm comparison, per run |
| `src/ui.py` | Shared UI pieces: pipeline strip, verdict banner, threshold score bars |
| `scripts/build_index.py` | Ingest `data/raw/` into ChromaDB |
| `scripts/build_benchmark.py` | Sample the labelled SQuAD 2.0 dev set (gold + distractors) |
| `scripts/run_benchmark.py` | Resumable harness: both arms over the benchmark, telemetry-tagged |
| `scripts/eval_benchmark.py` | EM/F1, hallucination rate, over-abstention per arm |
| `scripts/calibrate.py` | Fit gate thresholds on labelled data → `reports/calibration.json` |
| `scripts/risk_coverage.py` | Risk–coverage curves, AURC, figures |
| `scripts/judge_agreement.py` | NLI-vs-LLM-judge correlation, Cohen's κ, latency cost |
| `scripts/smoke_test.py` | Staged component verification (24 checks, no LLM needed) |
| `scripts/bench_latency.py` | Warm per-stage latency of the confidence layer |
| `scripts/make_thumbnail.py` | Regenerates `assets/thumbnail.png` (architecture banner) |
| `scripts/make_milestone_report.py` | Regenerates the milestone report `.docx` |
| `scripts/make_milestone5_report.py` | Builds the consolidated Milestone 5 report from saved evaluation results |
| `tests/` | Unit tests for the scoring maths, gate decision table, and eval statistics |

---

## Configuration

Everything is driven by `.env` (see `.env.example`). The two settings that matter most:

| Variable | Options | Effect |
| --- | --- | --- |
| `RAG_PROVIDER` | `ollama` / `groq` / `gemini` | Which backend generates and judges |
| `RAG_SCORER` | `nli` / `llm` / `both` | Local cross-encoders, DeepEval judge, or both |

`RAG_SCORER=nli` is the default because it costs no API calls, which is what makes a
benchmark sweep over thousands of questions feasible on a laptop. `both` runs each and
logs the disagreement, which is the basis of the local-vs-LLM-judge agreement study.

---

## Status

**Milestones 1–4 complete; Milestone 5 consolidates the results and reproducibility evidence.** Environment, ingestion, retrieval, both gates,
composite scoring, telemetry, Streamlit app and monitoring dashboard (M1);
local generation end to end, the SQuAD 2.0 harness and the baseline control
arm (M2); threshold calibration, risk–coverage analysis and the
NLI-vs-LLM-judge agreement study (M3); HotpotQA multi-hop evaluation,
chunk-vs-sentence granularity ablation, and the Gate B architectural
diagnosis (M4). The submitted milestone reports are in `reports/Submitted/`; the generator scripts write drafts to `reports/`.

### Milestone 1 (27 Aug 2026)

| Check | Command | Result |
|---|---|---|
| Unit tests (model-free) | `python -m pytest tests/ -q` | 31 passed |
| Integration smoke test | `python scripts/smoke_test.py` | 24 passed, 0 failed, 0 skipped |
| Linter | `ruff check .` | no findings |
| Confidence-layer latency | `python scripts/bench_latency.py` | 134 ms median (Gate A 16 ms + Gate B 118 ms) |

Measured separation on the sample corpus: an in-corpus question scores 0.999 at Gate A
while an out-of-corpus question scores 0.000 and abstains before the generator is called;
a grounded answer scores 0.997 at Gate B while a fabricated one scores 0.000 and is
labelled *Evident Conflict*.

### Milestone 2 (local generation, SQuAD 2.0 harness, control arm)

150-item SQuAD 2.0 dev sample (75 answerable / 75 unanswerable, gold paragraph +
4 cross-article distractors each), run through both arms with `llama3.1:8b`
(calibrated thresholds, final scorer):

| Metric | Guarded (gates on) | Baseline (control) |
| --- | --- | --- |
| Hallucination rate (unanswerable) | **0.107** | 1.00 |
| Over-abstention rate (answerable) | 0.72 | 0.00 |
| Exact match | **0.447** | 0.127 |
| F1 | **0.499** | 0.249 |

The control arm answers every unanswerable question — a 100% hallucination rate —
because nothing ever tells it to stop. The guarded arm abstains on 89% of them
while answering more answerable questions correctly than the baseline answers in
total. The remaining over-abstention is dominated by Gate B refusing
llama3.1:8b's fragmentary extractive answers (see *Known limitations* below).

### Milestone 3 (calibration, risk–coverage, judge agreement)

Thresholds fitted on the labelled dev sample (`scripts/calibrate.py`, full sweep
tables in `reports/calibration.json`):

| Gate | Signal | AUC | Fitted threshold | Fitting criterion |
| --- | --- | --- | --- | --- |
| A (retrieval sufficiency) | ms-marco rerank relevancy | 0.652 | 0.45 | Youden's J (answerable recall 0.80) |
| B (groundedness) | DeBERTa NLI faithfulness | 0.594 | 0.25 | F1 on the grounded proxy, constrained above the contradiction cap |
| Composite | weighted harmonic mean | — | 0.33 | 10% above the both-marginal corner |

A logistic regression over all four Gate A features (reranker + three embedding
statistics) reaches only cv-AUC 0.673 vs 0.652 for the reranker alone — the cheap
single-signal design is within noise of the multivariate fit.

Risk–coverage (`scripts/risk_coverage.py`, figures in `reports/figures/`):

| Arm | AURC | Risk @ 50% coverage | Risk @ 80% coverage |
| --- | --- | --- | --- |
| Guarded | **0.635** | 0.480 | 0.550 |
| Baseline | 0.866 | 0.840 | 0.873 |

**Two Gate B defects were found and fixed during calibration** — both are the
kind of thing the measurement exists to surface:

1. *Spurious cross-context contradictions*: nli-deberta's contradiction head
   fires on unrelated premises (an etymology paragraph "contradicts" a
   steam-engine claim at p=0.92). The old worst-case-over-all-contexts rule
   false-flagged 14 of 37 fully-entailed claims. Contradiction is now read from
   the claim's best-entailing context; the max over contexts applies only when
   nothing entails the claim.
2. *Broken judge verdicts*: `qwen2.5:7b-instruct` answers "no" to every DeepEval
   faithfulness verdict regardless of the claim (deterministic at temperature 0),
   making its scores meaningless. The judge is now `gemma2:latest`, which follows
   the yes/no convention correctly; the judge adapter also repairs the malformed
   JSON small local models emit (missing commas, code fences).

The full sweep data and per-arm confusion tables are in `reports/benchmark_m2.json`,
`reports/calibration.json`, and `reports/risk_coverage.json`.

**Known limitations** (honest accounting, Milestone 4 targets): Gate B has no
grounded-answer ground truth for its threshold — the answerable/unanswerable
proxy is imperfect, and llama3.1:8b's fragmentary answers ("New Orleans [1]")
are not proper NLI hypotheses, which depresses Gate B scores on answerable
questions and drives most of the over-abstention. HotpotQA multi-hop items
will stress Gate A's top-2-mean aggregation; the RAGTruth taxonomy spot-check
will validate the claim labels against human judgement.

### Milestone 4 (HotpotQA multi-hop, granularity ablation)

150-item HotpotQA dev sample, chunk vs. sentence indexing, both arms with
`llama3.1:8b` (full data in `reports/retrieval_ablation_m4.json` and
`reports/generation_hotpot_m4.json`):

**Retrieval dynamics** — 800-character chunks act as a volume dragnet,
mechanically boosting full recall by +35.3 points over strict sentence
indexing (66.0% vs 30.7%). The chunk index averages 11.8 units per item
against 40.8 for sentences; retrieving 5 chunks simply covers more source
text.

| Metric | Sentence | Chunk | Δ |
| --- | --- | --- | --- |
| Supporting-fact recall | 58.5% | 82.3% | +23.8 pp |
| Full recall (all facts retrieved) | 30.7% | 66.0% | **+35.3 pp** |

**Generation performance** — baseline F1 rose from 0.188 (sentence) to
0.271 (chunk), a **44% improvement**. The LLM successfully synthesises
answers from the wider chunk context when no confidence gate intervenes.

| Metric | Sent. baseline | Chunk baseline | Sent. guarded | Chunk guarded |
| --- | --- | --- | --- | --- |
| F1 | 0.188 | **0.271** | 0.086 | 0.150 |
| Answer rate | 100% | 100% | 17% | 22% |
| Gate B contradiction rate | 75% | 71% | 24% | 39% |

**The confidence bottleneck** — Gate B marked 70–75% of baseline answers
as contradictions on this HotpotQA sample. This is evidence that the current
single-premise NLI gate struggles with multi-hop answers, but it does not by
itself prove that every such flag is false or establish a universal architectural
boundary. The retrieval artifact reports zero items with both supporting facts
co-located in a chunk, so this run does not test whether one retrieved premise
can contain both hops. Multi-premise entailment, reasoning decomposition, or an
LLM judge remain candidate follow-up designs; they were not compared in this
milestone.

Mean reranker relevancy is 0.84 for chunks, above the 0.45 threshold. This
average does not establish per-item Gate A accuracy or prove clean transfer;
it shows that low mean reranker scores do not explain the low guarded answer
rate in this run.

### Milestone 5 finalization

Reproduce the model-free checks and launch the project demo:

```bash
python -m pytest tests/ -q -p no:cacheprovider
ruff check .
python scripts/smoke_test.py
streamlit run app.py
```

The smoke test and app need the configured local embedding and NLI models.
Generation also needs the selected LLM provider (Ollama by default). The
benchmark result files under `reports/` are the recorded experiment outputs;
the two HotpotQA sweep commands above rerun those experiments and require the
model services configured in `.env`.

Generate the final report with:

```bash
python scripts/make_milestone5_report.py
```

## Running tests

```bash
python -m pytest tests/ -q
```
