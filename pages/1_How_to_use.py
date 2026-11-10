"""How to use the app: a guided tour for first-time users and examiners.

Everything on this page is static explanation except the harmonic-mean demo,
which reuses the pipeline's own ``weighted_harmonic_mean`` so the numbers it
shows are exactly what the confidence layer computes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ui  # noqa: E402
from src.confidence import weighted_harmonic_mean  # noqa: E402
from src.config import ConfidenceConfig  # noqa: E402

st.set_page_config(page_title="Guide · Confidence-Aware RAG", page_icon="📖", layout="wide")
ui.inject_css()
cfg = ConfidenceConfig()

ui.hero("📖 How to use the app",
        "A five-minute tour: what the app does, how to read a result, and experiments worth trying.")
st.page_link("app.py", label="Back to the app", icon="🛡️")

quick, read, try_, dash, own, trouble, terms = st.tabs([
    "Quick start", "Reading a result", "Experiments", "Dashboard",
    "Your own documents", "Troubleshooting", "Glossary",
])

# --- Quick start ---------------------------------------------------------------
with quick:
    st.subheader("What this app does")
    st.write(
        "It answers questions from a set of documents (the project's RAG course notes by default), "
        "like any retrieval-augmented chatbot. The difference is that it **checks itself twice** and "
        "**refuses to answer** when it cannot show that the documents support the answer. A wrong "
        "answer delivered confidently is worse than an honest \"I don't know\"."
    )
    ui.pipeline_strip()
    st.subheader("Your first three questions")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card("1 · A question it can answer",
                "Click \"Ask this\" on the Lists sample. Expect a green \"Answered\" banner, "
                "every pipeline stage passed, and every claim marked Supported.")
    with c2:
        ui.card("2 · A question it should refuse",
                "Click the Out of scope sample (Amazon rainfall). Gate A fails, the Generate "
                "stage shows \"not run\": the model never even sees the question.")
    with c3:
        ui.card("3 · The same question without the checks",
                "Turn on Baseline mode in the sidebar and ask the rainfall question again. An "
                "ordinary RAG system answers anyway. This is the failure the project prevents.")
    st.info("The first question after starting the app takes 20 to 60 seconds while the models "
            "load onto the GPU. After that, answers take a few seconds.")

# --- Reading a result ----------------------------------------------------------
with read:
    st.subheader("1. The pipeline strip")
    st.write("The five boxes under the question box light up for every query:")
    st.markdown(
        f"- {ui.chip('passed', ui.PASS)} the stage ran and its check passed\n"
        f"- {ui.chip('failed', ui.FAIL)} the check failed, and the pipeline abstains\n"
        f"- {ui.chip('skipped', ui.SKIP)} the stage was not needed (for example, no generation "
        "after Gate A fails)\n"
        f"- {ui.chip('flagged', ui.WARN)} the model declined by itself, or a gate failed in baseline mode",
        unsafe_allow_html=True,
    )

    st.subheader("2. The verdict banner")
    st.dataframe(
        {
            "Banner": ["Answered", "Answered, low confidence", "Abstained: not enough relevant context",
                       "Abstained: answer not supported by the sources"],
            "What happened": [
                "Both gates passed and the combined score is above the flag line.",
                "Both gates passed, but the combined score is low. The answer is shown with a warning.",
                "Gate A failed, or the model itself said the passages were insufficient. Nothing was generated or shown.",
                "Gate B failed: the answer was generated but is not backed by the passages. "
                "You can still open it under \"Show the answer that was withheld\".",
            ],
        },
        hide_index=True, use_container_width=True,
    )

    st.subheader("3. The tabs under the answer")
    st.markdown(
        "- **Confidence scores** shows each gate's score as a bar. The thin vertical line is the "
        "threshold: green bars clear it, red bars fall short.\n"
        "- **Claim check** shows the answer split into sentences, each labelled **Supported**, "
        "**Not found in sources** (baseless information) or **Contradicted** (evident conflict), "
        "with the passage that supports it best.\n"
        "- **Sources** shows the passages that were retrieved, with their relevance scores.\n"
        "- **Timing & details** shows how long each stage took and which models ran."
    )

    st.subheader("4. Why the combined score uses a harmonic mean")
    st.write("Move the sliders. An ordinary average lets strong retrieval hide a fabricated answer; "
             "the harmonic mean the pipeline uses does not.")
    h1, h2 = st.columns(2)
    ra = h1.slider("Gate A score (retrieval)", 0.0, 1.0, 0.95, 0.01)
    rb = h2.slider("Gate B score (support)", 0.0, 1.0, 0.05, 0.01)
    arith = cfg.weights[0] * ra + cfg.weights[1] * rb
    harm = weighted_harmonic_mean([ra, rb], cfg.weights)
    m1, m2 = st.columns(2)
    with m1:
        ui.score_bar("Weighted average", arith, cfg.composite_threshold,
                     "Looks acceptable even when the answer is unsupported.")
    with m2:
        ui.score_bar("Weighted harmonic mean (what the app uses)", harm, cfg.composite_threshold,
                     "Collapses when either score is low.")

# --- Experiments -----------------------------------------------------------------
with try_:
    st.subheader("Experiments that show how it works")
    exps = [
        ("Make it stricter",
         "Open Thresholds (advanced) in the sidebar and raise Gate B to 0.95. Ask the Reasoning "
         "sample. Answers that were accepted may now be withheld: higher safety, lower coverage."),
        ("Make it reckless",
         "Lower Gate A to 0.00 and ask the Out of scope question. Generation now runs, and you can "
         "watch Gate B (or the model itself) catch what Gate A let through."),
        ("Compare with ordinary RAG",
         "Toggle Baseline mode and repeat any question. The scores are still computed, so you can "
         "see what the gates would have decided."),
        ("Ask a trick question",
         "Ask something the notes nearly cover, e.g. \"Who invented RAG and in what year?\". Watch "
         "which claims are marked Not found in sources."),
        ("Catch a false alarm",
         "Ask the False alarm sample. The answer is correct, yet Gate B labels one sentence "
         "Contradicted (contradiction 1.00) and withholds it. Open \"Show the answer that was "
         "withheld\" and the Claim check tab to see exactly which sentence and why. The NLI model "
         "(DeBERTa-base) struggles with long definition-style sentences; this is the main "
         "limitation of the project, and why a stronger NLI model is the recommended next step."),
        ("Reset",
         "\"Reset to calibrated values\" in the sidebar puts every threshold back to the values "
         "fitted on SQuAD 2.0."),
    ]
    for title, body in exps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)
    st.caption("Threshold changes only affect your session and take effect on the next question.")

# --- Dashboard -------------------------------------------------------------------
with dash:
    st.subheader("The Monitoring Dashboard")
    st.write(
        "Every question asked in the app is logged, along with every benchmark run. The dashboard "
        "(left menu) turns that log into charts: how often the system answers or abstains, which "
        "claims fail, how confident it is over time, and where the time goes."
    )
    st.markdown(
        "- **Data shown** picks what to analyse: your own questions from the app, or a specific "
        "benchmark run such as `squad2-v6` (Gate B v2) or `squad2-v5-v1` (Gate B v1).\n"
        "- A falling Gate A trend means people are asking about things the documents do not "
        "cover: a sign the knowledge base needs updating.\n"
        "- **Download CSV** exports the log for your own analysis."
    )
    st.page_link("pages/2_Monitoring_Dashboard.py", label="Open the dashboard", icon="📊")

# --- Own documents -----------------------------------------------------------------
with own:
    st.subheader("Asking about your own documents")
    st.markdown(
        "1. Put PDF, DOCX, Markdown or text files in `data/raw/`.\n"
        "2. Rebuild the index from the project folder:"
    )
    st.code("python scripts/build_index.py --reset", language="bash")
    st.markdown("3. Restart the app (or press **R** in it). The sidebar shows how many passages "
                "are indexed.")
    st.caption("Documents are split into ~800-character passages with overlap, so a sentence that "
               "spans a boundary is not lost.")

# --- Troubleshooting ---------------------------------------------------------------
with trouble:
    st.subheader("Troubleshooting")
    issues = [
        ("The first answer takes a long time",
         "Normal: the embedding, reranker, NLI and language models load on first use. On an 8 GB "
         "GPU everything fits only just, so close other GPU-heavy apps if it stays slow."),
        ("\"The pipeline failed\" / connection error",
         "Ollama is not running or the model is missing. Start Ollama, then run "
         "`ollama pull llama3.1:8b` once."),
        ("The sidebar says the index is empty",
         "Run `python scripts/build_index.py --reset` from the project folder."),
        ("Groq or Gemini does not work",
         "They need an API key in the `.env` file (see `.env.example`). Ollama needs none."),
        ("Everything abstains",
         "Check the thresholds in the sidebar: a very high Gate A or Gate B refuses almost "
         "everything. Use \"Reset to calibrated values\"."),
    ]
    for title, body in issues:
        with st.expander(title):
            st.write(body)

# --- Glossary ------------------------------------------------------------------------
with terms:
    st.subheader("Glossary")
    glossary = {
        "RAG": "Retrieval-Augmented Generation: retrieve relevant passages, then have a language "
               "model answer from them.",
        "Hallucination": "A fluent answer that the sources do not support, or that contradicts them.",
        "Abstain": "Decline to answer. Here it is a feature: an honest refusal instead of a guess.",
        "Gate A (reranker)": "A cross-encoder that reads the question and each passage together and "
                             "scores relevance. Runs before generation.",
        "Gate B (NLI)": "A natural language inference model that checks whether a passage entails "
                        "each sentence of the answer. Runs after generation.",
        "Entailment / contradiction": "NLI's verdicts: the passage implies the sentence, or the "
                                      "passage says the opposite.",
        "Threshold": "The minimum score a gate needs to pass. Calibrated on SQuAD 2.0.",
        "Composite": "The weighted harmonic mean of the two gate scores (0.4 relevance, "
                     "0.6 support).",
        "Baseline mode": "The control condition: same pipeline, checks measured but not enforced.",
    }
    for term, meaning in glossary.items():
        st.markdown(f"**{term}**: {meaning}")
