"""Architecture, Interactive Viva Lab, and User Guide for Confidence-Aware RAG.

Ink + Signal visual design: dark-first, calm, high-contrast, Linear/Vercel discipline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ui  # noqa: E402
from src.confidence import weighted_harmonic_mean  # noqa: E402
from src.config import ConfidenceConfig  # noqa: E402

st.set_page_config(
    page_title="Architecture & Viva Guide · Confidence-Aware RAG",
    page_icon="🛡️",
    layout="wide",
)
ui.inject_css()
cfg = ConfidenceConfig()

with st.sidebar:
    st.html("<div class='sidebar-sec-label'>Navigation</div>")
    st.page_link("app.py", label="RAG Verification Studio", icon=":material/shield:")
    st.page_link("pages/1_How_to_use.py", label="Architecture & Viva Guide", icon=":material/menu_book:")
    st.page_link("pages/2_Monitoring_Dashboard.py", label="Telemetry & Audit Dashboard", icon=":material/bar_chart:")

ui.hero(
    "Architecture, Interactive Lab & Viva Defense Guide",
    "Comprehensive guide to the two-gate selective prediction architecture, live harmonic mean simulator, "
    "and audited Milestone 1–5 findings for viva defense.",
    [
        "Gate A Pre-LLM Reranker",
        "Gate B v2 DeBERTa NLI",
        "Harmonic Mean Composite",
    ],
)

quick, read, try_, viva, own, trouble, terms = st.tabs(
    [
        "Quick Start",
        "Reading Telemetry",
        "Interactive Lab",
        "Viva & M1–M5 Defense",
        "Custom Documents",
        "Troubleshooting",
        "Glossary",
    ]
)

with quick:
    st.markdown("### Why Two Gates Instead of One Post-Generation Score?")
    st.write(
        "A conventional RAG pipeline always answers—even when vector search returns irrelevant "
        "nearest neighbours or when the LLM fabricates details from parametric memory. This system "
        "places two inline verification gates around generation and abstains whenever evidence is insufficient."
    )
    ui.pipeline_strip()

    st.markdown("### Three Core Diagnostic Scenarios")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card(
            "1 · Grounded In-Scope Answer",
            "Run 'Lists' on the Studio page. Gate A passes (0.50), Gate B verifies every sentence (0.997), "
            "and the harmonic composite reaches 0.714 -> Verified & Answered.",
        )
    with c2:
        ui.card(
            "2 · Pre-Generation Refusal (Gate A)",
            "Run 'Out of scope' (Amazon basin rainfall). Gate A scores ~0.00 (< 0.45) and halts the pipeline "
            "in ~15 ms before the LLM is ever invoked (0 tokens spent).",
        )
    with c3:
        ui.card(
            "3 · Gate B v1 vs v2 Premise Windowing",
            "Run 'False alarm' and switch NLI Premise between 'passage' (v1) and 'both' (v2: passages + 3-sentence windows) "
            "to observe how v2 resolves long-premise neutrality without hallucination.",
        )

with read:
    st.markdown("### 1. Pipeline Stage Status Badges")
    st.html(
        f"<div style='display:flex;gap:0.6rem;flex-wrap:wrap;margin:0.5rem 0 1.25rem 0'>"
        f"{ui.chip('PASSED · Threshold cleared', ui.PASS)}"
        f"{ui.chip('FAILED · Gate triggered abstention', ui.FAIL)}"
        f"{ui.chip('SKIPPED · Short-circuited saving tokens', ui.SKIP)}"
        f"{ui.chip('FLAGGED · Self-refused or unforced in baseline', ui.WARN)}"
        f"</div>"
    )

    st.markdown("### 2. Verdict Decision Matrix")
    st.dataframe(
        {
            "Verdict State": [
                "Verified & Answered",
                "Answered · Low Confidence Warning",
                "Abstained · Insufficient Context (Gate A)",
                "Abstained · Ungrounded Answer (Gate B)",
            ],
            "Gate A (>= 0.45)": ["Pass", "Pass", "Fail (or LLM self-refusal)", "Pass"],
            "Gate B (>= 0.25)": ["Pass", "Pass", "Not Run (0 tokens)", "Fail (or Contradiction Cap 0.15)"],
            "Composite (>= 0.33)": ["≥ 0.33", "< 0.33", "N/A", "N/A"],
            "User-Visible Action": [
                "Displays verified answer with inline [Ref] citations.",
                "Displays answer with an amber low-confidence warning banner.",
                "Refuses before generation; saves ~3–6 seconds of LLM compute.",
                "Suppresses drafted answer; allows inspection in withheld-answer drawer.",
            ],
        },
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 3. Interactive Simulator: Why Weighted Harmonic Mean?")
    st.write(
        "Adjust the sliders below to compare a standard weighted arithmetic average against the "
        "pipeline's weighted harmonic mean (`w_A = 0.4`, `w_B = 0.6`). Notice how an arithmetic mean "
        "lets high retrieval relevance disguise an unfaithful, hallucinated answer."
    )
    h1, h2 = st.columns(2)
    ra = h1.slider("Gate A Score (Retrieval Relevance)", 0.0, 1.0, 0.95, 0.01)
    rb = h2.slider("Gate B Score (Answer Faithfulness)", 0.0, 1.0, 0.05, 0.01)
    arith = cfg.weights[0] * ra + cfg.weights[1] * rb
    harm = weighted_harmonic_mean([ra, rb], cfg.weights)
    m1, m2 = st.columns(2)
    with m1:
        ui.score_bar(
            "Weighted Arithmetic Mean (0.4·A + 0.6·B)",
            arith,
            cfg.composite_threshold,
            "False sense of security: passes 0.33 even when faithfulness is near zero.",
        )
    with m2:
        ui.score_bar(
            "Weighted Harmonic Mean (Used by Pipeline)",
            harm,
            cfg.composite_threshold,
            "Collapses immediately when either gate approaches zero, enforcing both conditions.",
        )

with try_:
    st.markdown("### Guided Diagnostic Experiments")
    exps = [
        (
            "Experiment 1 · Precision vs. Coverage Trade-off",
            "Open 'Calibrated Gate Thresholds' in the sidebar and raise Gate B from 0.25 to 0.85. "
            "Run the Reasoning sample to observe selective prediction trading coverage for stricter verification.",
        ),
        (
            "Experiment 2 · Isolating Gate B when Gate A is Disabled",
            "Lower Gate A threshold to 0.00 and run the Amazon rainfall out-of-scope query. "
            "Watch whether the generator self-refuses (`INSUFFICIENT_CONTEXT`) or Gate B catches the ungrounded response.",
        ),
        (
            "Experiment 3 · Guarded vs. Prompt-Only vs. Baseline Control Arm (H3 Ablation)",
            "Select 'Prompt-Only (No Gates · H3 Ablation)' in the sidebar. The system prompt contains the abstention "
            "instruction and passage citations, but gates are unforced. Compare this against 'Baseline Control' and 'Guarded' "
            "to measure how much hallucination prevention comes from the prompt alone vs. the two confidence gates.",
        ),
        (
            "Experiment 4 · Gate B v1 (`passage`) vs. Gate B v2 (`both` / `multi_hop`)",
            "Open 'NLI Granularity & Hypotheses (M5)' in the sidebar. Compare `passage` mode against `both` "
            "(passages + 3-sentence windows) or `multi_hop` (which also pairs top-2 passages). Windowing prevents DeBERTa-v3 "
            "from going neutral on long paragraphs and properly entails synthesized facts.",
        ),
    ]
    for title, body in exps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)

with viva:
    st.markdown("### Milestone 1–5 Audit & Viva Defense Reference")
    qa = [
        (
            "Q1: Why use two separate gates instead of one post-generation score?",
            "Faithfulness is undefined until an answer exists, whereas retrieval relevance depends only on the "
            "query and retrieved passages. Gate A skipped LLM generation on 48 of 150 SQuAD 2.0 items (~16 ms vs ~3,000 ms) "
            "and carries the strongest hallucination-ranking signal on SQuAD 2.0 (AUROC 0.63).",
        ),
        (
            "Q2: Why did Milestone 3 have 72% over-abstention, and how did Milestone 5 fix it?",
            "Audit of M3 revealed that 35 of the 36 answerable answers suppressed by Gate B were actually correct. "
            "DeBERTa-v3-base (trained on 1–2 sentence premises) went neutral when handed whole 100–350 word paragraphs. "
            "Gate B v2 (`nli_premise='both'`) scores each claim against both full passages and overlapping 3-sentence "
            "windows, raising Gate B AUROC from 0.56 to 0.65 and cutting over-abstention from 71% to 48% (McNemar p < 0.001) "
            "while maintaining 97.4% selective accuracy (38/39).",
        ),
        (
            "Q3: How much of unanswerable refusal comes from the prompt vs. the gates?",
            "On the 75 unanswerable SQuAD 2.0 items in M3, 67 were refused in total: 33 by Gate A, 7 by Gate B "
            "(40 from the gates), and 27 from the generator's own `INSUFFICIENT_CONTEXT` self-refusal.",
        ),
        (
            "Q4: Why use local DeBERTa NLI instead of an LLM-as-a-Judge in production?",
            "In `reports/judge_agreement.json`, local NLI ran Gate B in 261.7 ms median vs 224,372.8 ms (~860× faster) "
            "for the DeepEval LLM judge, with near-zero Cohen's kappa (-0.037), showing small local LLM judges are "
            "neither fast enough for inline gating nor reliable ground-truth calibrators.",
        ),
    ]
    for q, a in qa:
        with st.expander(q, expanded=True):
            st.write(a)

with own:
    st.markdown("### Indexing Custom Documents")
    st.markdown(
        "You can ingest custom `.pdf`, `.docx`, `.md`, or `.txt` files in two ways:\n"
        "1. **Directly in the UI**: Open **Corpus & Document Upload** in the Studio sidebar, upload your files, "
        "and click **Ingest & Rebuild Index**.\n"
        "2. **Via CLI**: Place files in `data/raw/` and run:"
    )
    st.code("python scripts/build_index.py --reset", language="bash")

with trouble:
    st.markdown("### Troubleshooting Common Issues")
    issues = [
        (
            "Cold start takes 20–60 seconds on the first query",
            "Expected: `all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`, and `nli-deberta-v3-base` load into memory on first use.",
        ),
        (
            "ConnectionError / Ollama pipeline failure",
            "Start the Ollama daemon (`ollama serve`) and ensure `ollama pull llama3.1:8b` has completed.",
        ),
        (
            "Empty index warning in sidebar",
            "Click 'Ingest & Rebuild Index' in the sidebar or run `python scripts/build_index.py --reset`.",
        ),
    ]
    for title, body in issues:
        with st.expander(title):
            st.write(body)

with terms:
    st.markdown("### Technical Glossary")
    glossary = {
        "Gate A (Reranker)": "Pre-generation cross-encoder (`ms-marco-MiniLM-L-6-v2`) scoring query-passage relevance.",
        "Gate B v2 (NLI)": "Post-generation DeBERTa-v3 cross-encoder testing claim entailment over passages + 3-sentence windows.",
        "RAGTruth Taxonomy": "Categorises claims into Supported, Baseless Information (absent), or Evident Conflict (contradicted).",
        "Weighted Harmonic Mean": "Composite formula `1 / (0.4/A + 0.6/B)` that collapses if either gate is near zero.",
        "Selective Accuracy": "Proportion of answered questions that are correct (97.4% on SQuAD 2.0 answerable items under Gate B v2).",
        "AURC": "Area Under the Risk–Coverage Curve (lower is better; 0.460 for Gate A vs 0.547 random ranking).",
    }
    for term, meaning in glossary.items():
        st.markdown(f"- **{term}**: {meaning}")
