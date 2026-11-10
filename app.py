"""Streamlit front end for the confidence-aware RAG pipeline.

Run with:  streamlit run app.py

The interface is built around making the confidence decision *visible*. A demo
that just prints an answer proves nothing; this one shows which stage of the
pipeline passed or failed, both gate scores against their thresholds, the
per-claim entailment breakdown, and why the pipeline chose to answer or abstain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import ui  # noqa: E402
from src.config import ConfidenceConfig, RetrievalConfig, settings  # noqa: E402

st.set_page_config(
    page_title="Confidence-Aware RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The calibrated values for "Reset". Read from fresh config objects, not from
# ``settings``: the sliders write into ``settings``, which outlives a rerun.
_fresh_conf, _fresh_ret = ConfidenceConfig(), RetrievalConfig()
CALIBRATED = {
    "gate_a": _fresh_conf.gate_a_threshold,
    "gate_b": _fresh_conf.gate_b_threshold,
    "composite": _fresh_conf.composite_threshold,
    "top_k": _fresh_ret.top_k,
}

# Each sample demonstrates a different behaviour, and the label says which.
SAMPLES = [
    ("Lists", "What are the five stages of a standard RAG pipeline?",
     "In the notes: should be answered."),
    ("False alarm", "What is the difference between Baseless Information and Evident Conflict?",
     "In the notes, yet usually withheld: the NLI checker misreads a correct sentence as a "
     "contradiction. A known limitation, explained in the guide."),
    ("Reasoning", "Why can faithfulness only be computed after generation?",
     "In the notes, but needs explanation rather than lookup."),
    ("Out of scope", "What is the average annual rainfall in the Amazon basin?",
     "Not in the notes: Gate A should refuse before the LLM runs."),
]

CLAIM_BADGE = {
    "supported": ("✅", "Supported", ui.PASS),
    "evident_conflict": ("❌", "Contradicted", ui.FAIL),
    "baseless_information": ("⚠️", "Not found in sources", ui.WARN),
}


@st.cache_resource(show_spinner="Loading the document index…")
def _store(collection: str):
    from src.retriever import get_store

    return get_store(collection)


def _reset_thresholds() -> None:
    for key in ("gate_a", "gate_b", "composite", "top_k"):
        st.session_state[f"cfg_{key}"] = CALIBRATED[key]


def _sidebar() -> dict:
    for key, value in CALIBRATED.items():
        st.session_state.setdefault(f"cfg_{key}", value)
    with st.sidebar:
        st.page_link("pages/1_How_to_use.py", label="New here? Read the guide", icon="📖")
        st.divider()

        st.subheader("Mode")
        baseline = st.toggle(
            "Baseline mode",
            value=False,
            help="Turns the safety checks off: the gates are still measured and shown, but the "
                 "answer is always displayed. Use it to see what an ordinary RAG system would say.",
        )

        st.subheader("Models")
        provider = st.selectbox(
            "Answer model provider",
            ["ollama", "groq", "gemini"],
            index=["ollama", "groq", "gemini"].index(settings.llm.provider),
            help="Ollama runs locally with no rate limits. Groq and Gemini need an API key in .env.",
        )
        settings.llm.provider = provider  # type: ignore[assignment]
        scorer = st.selectbox(
            "Confidence checker",
            ["nli", "llm", "both"],
            index=["nli", "llm", "both"].index(settings.confidence.scorer),
            help="nli = fast local models (milliseconds). llm = an LLM judge, much slower. "
                 "both = run each and compare.",
        )
        settings.confidence.scorer = scorer  # type: ignore[assignment]

        with st.expander("Thresholds (advanced)", expanded=False):
            st.caption("Calibrated on SQuAD 2.0. Changes apply to this session only, "
                       "on your next question.")
            settings.confidence.gate_a_threshold = st.slider(
                "Gate A · relevance needed", 0.0, 1.0, step=0.01, key="cfg_gate_a",
                help="Higher = refuses more questions before generating.")
            settings.confidence.gate_b_threshold = st.slider(
                "Gate B · support needed", 0.0, 1.0, step=0.01, key="cfg_gate_b",
                help="Higher = withholds more answers that are only partly backed by the sources.")
            settings.confidence.composite_threshold = st.slider(
                "Low-confidence flag below", 0.0, 1.0, step=0.01, key="cfg_composite",
                help="Answers whose combined score falls below this are shown with a warning.")
            top_k = st.slider("Passages to retrieve", 1, 12, key="cfg_top_k")
            st.button("Reset to calibrated values", on_click=_reset_thresholds, use_container_width=True)

        st.divider()
        index_slot = st.empty()
    return {"top_k": top_k, "baseline": baseline, "index_slot": index_slot}


def _index_status(slot) -> None:
    """Filled in after the page is drawn: loading the index takes seconds on a cold start."""
    try:
        count = _store(settings.retrieval.collection_name)._collection.count()
        if count:
            slot.caption(f"📚 {count} passages indexed in '{settings.retrieval.collection_name}'")
        else:
            slot.warning("The index is empty. Add files to data/raw/ and run "
                         "`python scripts/build_index.py --reset`.")
    except Exception as exc:  # noqa: BLE001
        slot.error(f"Index unavailable: {exc}")


def _ask_form() -> str | None:
    st.markdown("##### Try a sample question")
    for row in (SAMPLES[:2], SAMPLES[2:]):
        for col, (tag, q, why) in zip(st.columns(2), row, strict=True):
            with col.container(border=True):
                st.markdown(f"**{q}**")
                st.caption(f"{tag} · {why}")
                if st.button("Ask this →", key=f"sample_{tag}"):
                    st.session_state["question"] = q
                    st.session_state["submitted"] = q

    with st.form("ask", clear_on_submit=False, border=False):
        q_col, b_col = st.columns([5, 1], vertical_alignment="bottom")
        question = q_col.text_input(
            "Or ask your own question about the indexed documents",
            key="question",
            placeholder="e.g. Why does RAG still hallucinate?",
        )
        if b_col.form_submit_button("Ask", type="primary", use_container_width=True):
            st.session_state["submitted"] = question.strip()
    return st.session_state.get("submitted") or None


def _run(question: str, opts: dict):
    """Run the pipeline once per question+settings; the result survives reruns."""
    from src.pipeline import answer_question

    sig = (question, opts["top_k"], opts["baseline"], settings.confidence.gate_a_threshold,
           settings.confidence.gate_b_threshold, settings.confidence.composite_threshold,
           settings.llm.provider, settings.confidence.scorer)
    cached = st.session_state.get("last")
    if cached and cached[0] == sig:
        return cached[1]

    with st.status("Working on it…", expanded=True) as status:
        st.write("Retrieving passages, checking relevance (Gate A), generating, "
                 "checking support (Gate B)…")
        st.caption("The first question after starting the app is slow while the models load.")
        try:
            result = answer_question(question, top_k=opts["top_k"], baseline=opts["baseline"],
                                     store=_store(settings.retrieval.collection_name))
        except Exception as exc:  # noqa: BLE001
            status.update(label="The pipeline failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            st.info("Is Ollama running? Start it, then ask again. See the guide's "
                    "Troubleshooting section.")
            return None
        status.update(label=f"Done in {result.total_ms / 1000:.1f} s", state="complete", expanded=False)
    st.session_state["last"] = (sig, result)
    return result


def _render_answer(result, baseline: bool) -> None:
    rep = result.confidence
    if baseline:
        ui.verdict_banner("answer", "Baseline mode: the gates were measured but not enforced.")
    else:
        ui.verdict_banner(rep.verdict, rep.explanation)

    with st.container(border=True):
        st.markdown("**Answer**")
        st.write(result.displayed_answer)
        if rep.abstained and result.answer and not baseline:
            with st.expander("Show the answer that was withheld"):
                st.caption("The model did produce this, but the checks judged it unsafe to show.")
                st.write(result.answer)


def _tab_scores(rep) -> None:
    a, b = rep.gate_a, rep.gate_b
    ui.score_bar("Gate A · how relevant are the retrieved passages?", a.score,
                 settings.confidence.gate_a_threshold,
                 "Cross-encoder reranker, average of the top two passages. Checked before generating.")
    ui.score_bar("Gate B · how well do the sources support the answer?", b.score if b else None,
                 settings.confidence.gate_b_threshold,
                 "NLI model, average support across the answer's sentences. A contradicted "
                 "sentence caps this at 0.15." if b else "Not run: no answer was generated.")
    ui.score_bar("Combined confidence (weighted harmonic mean)", rep.composite,
                 settings.confidence.composite_threshold,
                 "Pulled down hard by whichever gate is weaker, so good retrieval cannot hide a bad answer.")
    with st.expander("Raw retrieval signals"):
        st.dataframe(
            {
                "signal": ["reranker relevance (drives Gate A)", "top embedding similarity",
                           "mean embedding similarity", "similarity margin (top 1 - top 2)"],
                "value": [round(a.rerank_relevancy, 4), round(a.top_similarity, 4),
                          round(a.mean_similarity, 4), round(a.similarity_margin, 4)],
            },
            hide_index=True, use_container_width=True,
        )
        st.caption("Embedding similarity is logged for analysis but does not drive the gate: "
                   "MiniLM cosine values are not calibrated probabilities.")


def _tab_claims(rep) -> None:
    b = rep.gate_b
    if b is None or not b.claims:
        st.info("No claims to check: the pipeline stopped before an answer was generated.")
        return
    counts = b.taxonomy_counts
    c1, c2, c3 = st.columns(3)
    c1.metric("Supported", counts["supported"])
    c2.metric("Not found in sources", counts["baseless_information"])
    c3.metric("Contradicted", counts["evident_conflict"])
    st.caption("Each sentence of the answer is checked against the passages "
               "(RAGTruth labels: supported, baseless information, evident conflict).")
    for i, c in enumerate(b.claims, 1):
        icon, label, colour = CLAIM_BADGE[c.taxonomy_label]
        with st.container(border=True):
            st.html(f"{ui.chip(label, colour)} <span style='opacity:.7;font-size:.8rem'>"
                    f"sentence {i} · best support from passage {c.best_context_index + 1}</span>")
            st.markdown(f"{icon} {c.claim}")
            st.progress(min(max(c.entailment, 0.0), 1.0),
                        text=f"support {c.entailment:.2f} · neutral {c.neutral:.2f} · "
                             f"contradiction {c.contradiction:.2f}")


def _tab_sources(result) -> None:
    a = result.confidence.gate_a
    for i, chunk in enumerate(result.retrieval.chunks):
        rel = a.per_chunk_relevance[i] if i < len(a.per_chunk_relevance) else None
        title = f"Passage {i + 1} · {Path(chunk.source).name}"
        if rel is not None:
            title += f" · relevance {rel:.2f}"
        with st.expander(title, expanded=i == 0):
            st.write(chunk.text)
            st.caption(f"embedding similarity {chunk.similarity:.3f}")


def _tab_details(result) -> None:
    rep = result.confidence
    rows = [("Retrieval", result.retrieval_ms), ("Gate A", rep.gate_a.latency_ms)]
    if result.generation:
        rows.append(("Generation", result.generation.latency_ms))
    if rep.gate_b:
        rows.append(("Gate B", rep.gate_b.latency_ms))
    rows.append(("Total", result.total_ms))
    st.dataframe({"stage": [r[0] for r in rows], "milliseconds": [round(r[1]) for r in rows]},
                 hide_index=True, use_container_width=True)
    st.caption(f"Provider {settings.llm.provider} · model {settings.llm.generator_model} · "
               f"checker {settings.confidence.scorer} · logged to the Monitoring Dashboard.")
    a, b = rep.gate_a, rep.gate_b
    if a.llm_reason or (b and b.llm_reason):
        st.markdown("**LLM judge reasoning**")
        if a.llm_relevancy is not None:
            st.write(f"Contextual relevancy {a.llm_relevancy:.3f}: {a.llm_reason}")
        if b is not None and b.llm_faithfulness is not None:
            st.write(f"Faithfulness {b.llm_faithfulness:.3f}: {b.llm_reason}")


def _intro() -> None:
    st.markdown("##### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card("1 · Check before answering",
                "Gate A asks whether the retrieved passages are relevant at all. If not, the "
                "system says so instead of letting the model guess.")
    with c2:
        ui.card("2 · Check after answering",
                "Gate B tests every sentence of the answer against the passages and flags "
                "anything unsupported or contradicted.")
    with c3:
        ui.card("3 · Answer or abstain",
                "Only answers that pass both checks are shown. Everything else is withheld, "
                "with the reason and the evidence.")


def main() -> None:
    ui.inject_css()
    opts = _sidebar()

    ui.hero("🛡️ Confidence-Aware RAG",
            "Ask a question about your documents. The system answers only when it can show "
            "the sources support it, and tells you why when it can't.")

    question = _ask_form()
    result = _run(question, opts) if question else None
    ui.pipeline_strip(result, baseline=opts["baseline"])

    if result is None:
        _intro()
        _index_status(opts["index_slot"])
        return

    _render_answer(result, opts["baseline"])
    t1, t2, t3, t4 = st.tabs(["Confidence scores", "Claim check", "Sources", "Timing & details"])
    with t1:
        _tab_scores(result.confidence)
    with t2:
        _tab_claims(result.confidence)
    with t3:
        _tab_sources(result)
    with t4:
        _tab_details(result)
    _index_status(opts["index_slot"])


if __name__ == "__main__":
    main()
