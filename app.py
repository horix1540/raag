"""Streamlit front end for the Confidence-Aware RAG Studio.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import ui  # noqa: E402
from src.config import RAW_DIR, ConfidenceConfig, RetrievalConfig, settings  # noqa: E402

st.set_page_config(
    page_title="Confidence-Aware RAG Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_fresh_conf, _fresh_ret = ConfidenceConfig(), RetrievalConfig()
CALIBRATED = {
    "gate_a": _fresh_conf.gate_a_threshold,
    "gate_b": _fresh_conf.gate_b_threshold,
    "composite": _fresh_conf.composite_threshold,
    "top_k": _fresh_ret.top_k,
    "nli_premise": _fresh_conf.nli_premise,
    "nli_hypothesis": _fresh_conf.nli_hypothesis,
}

SAMPLES = [
    (
        "Lists · Grounded Pass",
        "What are the five stages of a standard RAG pipeline?",
        "In the notes: passes Gate A (0.50) & Gate B (0.997) -> Answered.",
        ui.PASS,
    ),
    (
        "False Alarm · NLI Edge Case",
        "What is the difference between Baseless Information and Evident Conflict?",
        "In the notes: compare Gate B v1 (passage) vs v2 (3-sentence windows).",
        ui.WARN,
    ),
    (
        "Reasoning · Multi-Claim",
        "Why can faithfulness only be computed after generation?",
        "In the notes: tests multi-sentence explanation & claim verification.",
        ui.BRAND,
    ),
    (
        "Out of Scope · Gate A Refusal",
        "What is the average annual rainfall in the Amazon basin?",
        "Not in the notes: Gate A refuses in ~15 ms before the LLM ever runs.",
        ui.FAIL,
    ),
]

CLAIM_BADGE = {
    "supported": ("✅", "Supported (Entailed)", ui.PASS),
    "evident_conflict": ("❌", "Evident Conflict (Contradicted)", ui.FAIL),
    "baseless_information": ("⚠️", "Baseless Info (Not in Sources)", ui.WARN),
}


@st.cache_resource(show_spinner="Loading ChromaDB vector store…")
def _store(collection: str):
    from src.retriever import get_store

    return get_store(collection)


def _reset_thresholds() -> None:
    for key in ("gate_a", "gate_b", "composite", "top_k", "nli_premise", "nli_hypothesis"):
        st.session_state[f"cfg_{key}"] = CALIBRATED[key]


def _sidebar() -> dict:
    for key, value in CALIBRATED.items():
        st.session_state.setdefault(f"cfg_{key}", value)
    with st.sidebar:
        st.markdown("### 🛡️ Navigation")
        st.page_link("app.py", label="RAG Verification Studio", icon="🛡️")
        st.page_link("pages/1_How_to_use.py", label="Architecture & Viva Guide", icon="📖")
        st.page_link("pages/2_Monitoring_Dashboard.py", label="Telemetry & Audit Dashboard", icon="📊")
        st.divider()

        st.markdown("### ⚙️ Execution Arm & Mode")
        exec_mode = st.radio(
            "Pipeline execution arm",
            [
                "Guarded (2-Gate Safety)",
                "Prompt-Only (No Gates · H3 Ablation)",
                "Baseline Control (Unguarded)",
                "Side-by-Side (Guarded vs. Baseline)",
            ],
            index=0,
            help="Compare Guarded mode against Prompt-Only (isolates prompt vs. gates per Project Review H3), "
            "Baseline Control, or run Guarded and Baseline side-by-side.",
        )
        baseline = exec_mode == "Baseline Control (Unguarded)"
        prompt_only = exec_mode == "Prompt-Only (No Gates · H3 Ablation)"
        compare_both = exec_mode == "Side-by-Side (Guarded vs. Baseline)"

        st.markdown("### 🧠 Model & Gate Backends")
        provider = st.selectbox(
            "Generator LLM provider",
            ["ollama", "groq", "gemini"],
            index=["ollama", "groq", "gemini"].index(settings.llm.provider),
            help="Ollama runs locally (llama3.1:8b). Groq and Gemini use .env API keys.",
        )
        settings.llm.provider = provider  # type: ignore[assignment]

        scorer = st.selectbox(
            "Confidence verifier backend",
            ["nli", "llm", "both"],
            index=["nli", "llm", "both"].index(settings.confidence.scorer),
            help="nli = fast local cross-encoders (~340 ms). llm = DeepEval judge. both = run & compare both.",
        )
        settings.confidence.scorer = scorer  # type: ignore[assignment]

        if st.button("🔌 Test LLM Connection", use_container_width=True):
            from src.llm import health_check

            ok, msg = health_check()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        with st.expander("🔬 Gate B v2 NLI Settings (M5)", expanded=False):
            st.caption(
                "Gate B v2 scores claims against whole passages, 3-sentence windows, or multi-hop top-2 pairs."
            )
            premise_opts = ["both", "multi_hop", "window", "passage"]
            if st.session_state.get("cfg_nli_premise") not in premise_opts:
                st.session_state["cfg_nli_premise"] = "both"
            premise = st.selectbox(
                "NLI premise granularity",
                premise_opts,
                key="cfg_nli_premise",
                help="both = Gate B v2 (passages + 3-sentence windows). multi_hop = adds top-2 passage pair. passage = v1.",
            )
            settings.confidence.nli_premise = premise

            hyp_opts = ["claim", "claim_clean", "qa_short", "qa"]
            if st.session_state.get("cfg_nli_hypothesis") not in hyp_opts:
                st.session_state["cfg_nli_hypothesis"] = "claim"
            hyp = st.selectbox(
                "NLI hypothesis construction",
                hyp_opts,
                key="cfg_nli_hypothesis",
                help="claim = v1 raw sentence. claim_clean = strips [1] citations. qa_short = prefixes question on short claims.",
            )
            settings.confidence.nli_hypothesis = hyp

        with st.expander("🎚️ Calibrated Thresholds", expanded=False):
            st.caption("Fitted on SQuAD 2.0 dev set (Youden's J for Gate A, F1 above contradiction cap for Gate B).")
            settings.confidence.gate_a_threshold = st.slider(
                "Gate A · retrieval relevance",
                0.0,
                1.0,
                step=0.01,
                key="cfg_gate_a",
            )
            settings.confidence.gate_b_threshold = st.slider(
                "Gate B · NLI support needed",
                0.0,
                1.0,
                step=0.01,
                key="cfg_gate_b",
            )
            settings.confidence.composite_threshold = st.slider(
                "Composite low-confidence warning",
                0.0,
                1.0,
                step=0.01,
                key="cfg_composite",
            )
            top_k = st.slider("Top-K passages to retrieve", 1, 12, key="cfg_top_k")
            st.button(
                "Reset to calibrated defaults",
                on_click=_reset_thresholds,
                use_container_width=True,
            )

        with st.expander("📂 Corpus & Document Upload", expanded=False):
            st.caption("Upload PDF, DOCX, MD, or TXT files to `data/raw/` and re-index live.")
            uploaded = st.file_uploader(
                "Add documents to index",
                type=["pdf", "docx", "md", "txt"],
                accept_multiple_files=True,
            )
            if uploaded and st.button("Ingest & Rebuild Index", type="primary", use_container_width=True):
                from src.ingest import build_index

                for uf in uploaded:
                    (RAW_DIR / uf.name).write_bytes(uf.getvalue())
                with st.spinner("Chunking & embedding documents into ChromaDB…"):
                    build_index(RAW_DIR, reset=True)
                    _store.clear()
                st.success(f"Indexed {len(uploaded)} new file(s) into '{settings.retrieval.collection_name}'.")

        st.divider()
        index_slot = st.empty()
    return {
        "top_k": top_k,
        "baseline": baseline,
        "prompt_only": prompt_only,
        "compare_both": compare_both,
        "index_slot": index_slot,
    }


def _index_status(slot) -> None:
    try:
        count = _store(settings.retrieval.collection_name)._collection.count()
        if count:
            slot.caption(
                f"🟢 **{count} passages indexed** in `{settings.retrieval.collection_name}` "
                f"· Premise `{settings.confidence.nli_premise}`"
            )
        else:
            slot.warning(
                "The index is empty. Upload files in the sidebar or run "
                "`python scripts/build_index.py --reset`."
            )
    except Exception as exc:  # noqa: BLE001
        slot.error(f"Index unavailable: {exc}")


def _ask_form() -> str | None:
    st.markdown("##### ⚡ Diagnostic Sample Queries")
    for row in (SAMPLES[:2], SAMPLES[2:]):
        cols = st.columns(2)
        for col, (tag, q, why, colour) in zip(cols, row, strict=True):
            with col:
                ui.sample_card(tag, q, why, colour)
                if st.button(
                    f"Run diagnostic · {tag.split('·')[0].strip()} →",
                    key=f"sample_{tag}",
                    use_container_width=True,
                ):
                    st.session_state["question"] = q
                    st.session_state["submitted"] = q

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    with st.form("ask", clear_on_submit=False, border=True):
        q_col, b_col = st.columns([5, 1], vertical_alignment="bottom")
        question = q_col.text_input(
            "Ask a question against the indexed knowledge base",
            key="question",
            placeholder="e.g. Why does RAG still hallucinate even when relevant passages are retrieved?",
        )
        if b_col.form_submit_button("Verify & Ask", type="primary", use_container_width=True):
            st.session_state["submitted"] = question.strip()
    return st.session_state.get("submitted") or None


def _run_single(question: str, top_k: int, baseline: bool, prompt_only: bool):
    from src.pipeline import answer_question

    return answer_question(
        question,
        top_k=top_k,
        baseline=baseline,
        prompt_only=prompt_only,
        store=_store(settings.retrieval.collection_name),
    )


def _run(question: str, opts: dict):
    sig = (
        question,
        opts["top_k"],
        opts["baseline"],
        opts["prompt_only"],
        opts["compare_both"],
        settings.confidence.gate_a_threshold,
        settings.confidence.gate_b_threshold,
        settings.confidence.composite_threshold,
        settings.llm.provider,
        settings.confidence.scorer,
        settings.confidence.nli_premise,
        settings.confidence.nli_hypothesis,
    )
    cached = st.session_state.get("last")
    if cached and cached[0] == sig:
        return cached[1]

    with st.status("Executing confidence-aware pipeline…", expanded=True) as status:
        st.write(
            "1. Retrieving top-k chunks via MiniLM bi-encoder\n"
            "2. Evaluating Gate A cross-encoder relevance (`ms-marco-MiniLM-L-6-v2`)\n"
            "3. Generating context-constrained answer & running Gate B DeBERTa NLI verification…"
        )
        try:
            if opts["compare_both"]:
                guarded_res = _run_single(question, opts["top_k"], baseline=False, prompt_only=False)
                base_res = _run_single(question, opts["top_k"], baseline=True, prompt_only=False)
                result = (guarded_res, base_res)
                status.update(
                    label=f"Completed Side-by-Side Comparison · Guarded: {guarded_res.verdict}",
                    state="complete",
                    expanded=False,
                )
            else:
                result = _run_single(
                    question,
                    opts["top_k"],
                    baseline=opts["baseline"],
                    prompt_only=opts["prompt_only"],
                )
                status.update(
                    label=f"Completed in {result.total_ms / 1000:.2f} s · Verdict: {result.verdict}",
                    state="complete",
                    expanded=False,
                )
        except Exception as exc:  # noqa: BLE001
            status.update(label="Pipeline execution failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            st.info(
                "Ensure Ollama is running (`ollama serve` and `ollama pull llama3.1:8b`), "
                "or switch provider to Groq/Gemini in the sidebar."
            )
            return None
    st.session_state["last"] = (sig, result)
    return result


def _render_answer(result, baseline: bool, prompt_only: bool) -> None:
    rep = result.confidence
    if baseline:
        ui.verdict_banner(
            "answer",
            f"Baseline control arm active: gates were measured (Composite {rep.composite:.2f}) but not enforced.",
        )
    elif prompt_only:
        ui.verdict_banner(
            rep.verdict,
            f"Prompt-only arm (H3 ablation): abstention prompt active; gates measured (Composite {rep.composite:.2f}) "
            "but not enforced.",
        )
    else:
        ui.verdict_banner(rep.verdict, rep.explanation)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.kpi_card(
            "Gate A · Relevance",
            f"{rep.gate_a.score:.3f}",
            f"Threshold ≥ {settings.confidence.gate_a_threshold:.2f}",
            ui.PASS if rep.gate_a.passed else ui.FAIL,
        )
    with k2:
        gb_val = f"{rep.gate_b.score:.3f}" if rep.gate_b else "Skipped"
        gb_col = (ui.PASS if rep.gate_b.passed else ui.FAIL) if rep.gate_b else ui.SKIP
        ui.kpi_card(
            "Gate B · Faithfulness",
            gb_val,
            f"Threshold ≥ {settings.confidence.gate_b_threshold:.2f} ({settings.confidence.nli_premise})",
            gb_col,
        )
    with k3:
        comp_col = (
            ui.PASS
            if not rep.abstained and rep.composite >= settings.confidence.composite_threshold
            else (ui.WARN if not rep.abstained else ui.FAIL)
        )
        ui.kpi_card(
            "Harmonic Composite",
            f"{rep.composite:.3f}",
            f"Weights 0.4 A / 0.6 B (≥ {settings.confidence.composite_threshold:.2f})",
            comp_col,
        )
    with k4:
        ui.kpi_card(
            "End-to-End Latency",
            f"{result.total_ms / 1000:.2f} s",
            f"Gates: {(rep.gate_a.latency_ms + (rep.gate_b.latency_ms if rep.gate_b else 0.0)):.0f} ms total",
            ui.BRAND,
        )

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    formatted_ans = ui.format_answer_html(result.displayed_answer)
    st.html(
        f"<div class='cr-answer-box'>"
        f"<div class='cr-answer-hdr'><span>System Response</span>"
        f"<span>Model: {settings.llm.provider}:{settings.llm.generator_model}</span></div>"
        f"<div class='cr-answer-body'>{formatted_ans}</div>"
        f"</div>"
    )
    if rep.abstained and result.answer and not (baseline or prompt_only):
        with st.expander("🔍 Inspect withheld LLM draft (suppressed by Gate B)"):
            st.caption(
                "The generator produced this draft, but Gate B detected unsupported or contradicted "
                "claims and blocked it from reaching the user."
            )
            st.write(result.answer)


def _render_side_by_side(guarded_res, base_res) -> None:
    st.markdown("##### ⚖️ Live Side-by-Side Comparison: Guarded Pipeline vs. Unguarded Baseline")
    col_g, col_b = st.columns(2)
    with col_g:
        st.markdown("#### 🛡️ Guarded Arm (2-Gate Safety Active)")
        ui.verdict_banner(guarded_res.confidence.verdict, guarded_res.confidence.explanation)
        st.html(
            f"<div class='cr-answer-box'>"
            f"<div class='cr-answer-hdr'><span>Guarded Output</span>"
            f"<span>Composite: {guarded_res.confidence.composite:.3f}</span></div>"
            f"<div class='cr-answer-body'>{ui.format_answer_html(guarded_res.displayed_answer)}</div>"
            f"</div>"
        )
    with col_b:
        st.markdown("#### ⚠️ Baseline Control Arm (Gates Unforced)")
        ui.verdict_banner("answer", "Baseline control arm: always answers regardless of Gate A or Gate B failure.")
        st.html(
            f"<div class='cr-answer-box'>"
            f"<div class='cr-answer-hdr'><span>Baseline Output</span>"
            f"<span>Composite: {base_res.confidence.composite:.3f}</span></div>"
            f"<div class='cr-answer-body'>{ui.format_answer_html(base_res.displayed_answer)}</div>"
            f"</div>"
        )


def _tab_scores(rep) -> None:
    a, b = rep.gate_a, rep.gate_b
    ui.score_bar(
        "Gate A · Pre-Generation Retrieval Sufficiency (ms-marco-MiniLM-L-6-v2)",
        a.score,
        settings.confidence.gate_a_threshold,
        "Mean sigmoid relevance of top-2 retrieved passages. Evaluated before calling the LLM.",
    )
    ui.score_bar(
        f"Gate B · Post-Generation Claim Groundedness (DeBERTa-v3 NLI · premise='{settings.confidence.nli_premise}')",
        b.score if b else None,
        settings.confidence.gate_b_threshold,
        "Mean entailment probability across extracted answer sentences. Any Evident Conflict caps this at 0.15."
        if b
        else "Not run: generation was skipped at Gate A or the model self-refused.",
    )
    ui.score_bar(
        "Composite Confidence · Weighted Harmonic Mean (0.4 Gate A + 0.6 Gate B)",
        rep.composite,
        settings.confidence.composite_threshold,
        "Dominated by whichever gate is weaker so strong retrieval cannot mask an ungrounded generation.",
    )
    with st.expander("📊 Raw Retrieval & Embedding Diagnostic Signals", expanded=True):
        st.dataframe(
            {
                "Diagnostic Signal": [
                    "Cross-encoder reranker relevance (drives Gate A)",
                    "Top-1 passage cosine similarity (all-MiniLM-L6-v2)",
                    "Mean top-k cosine similarity",
                    "Similarity margin (Top 1 − Top 2)",
                ],
                "Measured Value": [
                    round(a.rerank_relevancy, 4),
                    round(a.top_similarity, 4),
                    round(a.mean_similarity, 4),
                    round(a.similarity_margin, 4),
                ],
                "Role in Pipeline": [
                    "Primary calibrated decision signal (AUC 0.652)",
                    "Diagnostic telemetry (uncalibrated cosine scale)",
                    "Corpus density indicator",
                    "Hit concentration indicator",
                ],
            },
            hide_index=True,
            use_container_width=True,
        )


def _tab_claims(rep) -> None:
    b = rep.gate_b
    if b is None or not b.claims:
        st.info("No claims were evaluated because the pipeline abstained before generation.")
        return
    counts = b.taxonomy_counts
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.kpi_card("Supported Claims", str(counts["supported"]), "Entailment ≥ 0.50", ui.PASS)
    with c2:
        ui.kpi_card("Baseless Information", str(counts["baseless_information"]), "Entailment < 0.50", ui.WARN)
    with c3:
        ui.kpi_card(
            "Evident Conflict",
            str(counts["evident_conflict"]),
            "Contradiction ≥ 0.50 (Caps Gate B at 0.15)",
            ui.FAIL,
        )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.caption(
        "Sentence-level verification following the RAGTruth hallucination taxonomy "
        f"(scored with `nli_premise='{settings.confidence.nli_premise}'`, "
        f"`nli_hypothesis='{settings.confidence.nli_hypothesis}'`)."
    )
    for i, c in enumerate(b.claims, 1):
        icon, label, colour = CLAIM_BADGE[c.taxonomy_label]
        with st.container(border=True):
            st.html(
                f"{ui.chip(label, colour)} "
                f"<span style='opacity:.75;font-size:.8rem;font-family:JetBrains Mono,monospace'>"
                f"Claim #{i} · Best match: Passage #{c.best_context_index + 1}</span>"
            )
            st.markdown(f"**{icon} {c.claim}**")
            st.progress(
                min(max(c.entailment, 0.0), 1.0),
                text=(
                    f"Entailment (Support): {c.entailment:.3f}  |  "
                    f"Neutral: {c.neutral:.3f}  |  "
                    f"Contradiction: {c.contradiction:.3f}"
                ),
            )


def _tab_sources(result) -> None:
    a = result.confidence.gate_a
    b = result.confidence.gate_b
    supported_claims = [c.claim for c in (b.claims if b else []) if c.is_supported]
    st.caption("Sentences matching verified claims are highlighted in green.")
    for i, chunk in enumerate(result.retrieval.chunks):
        rel = a.per_chunk_relevance[i] if i < len(a.per_chunk_relevance) else None
        rel_badge = f" · Reranker {rel:.3f}" if rel is not None else ""
        src_name = Path(chunk.source).name if chunk.source != "unknown" else "rag_course_notes.md"
        title = f"Passage #{i + 1} · {src_name}{rel_badge} · Cosine {chunk.similarity:.3f}"
        with st.expander(title, expanded=i < 2):
            if rel is not None:
                col = ui.PASS if rel >= settings.confidence.gate_a_threshold else ui.WARN
                st.html(
                    f"{ui.chip(f'Reranker Relevance: {rel:.3f}', col)} "
                    f"{ui.chip(f'Bi-Encoder Cosine: {chunk.similarity:.3f}', ui.BRAND)}"
                )
            st.html(
                f"<div style='font-size:0.93rem;line-height:1.65'>"
                f"{ui.highlight_passage_html(chunk.text, supported_claims)}</div>"
            )


def _tab_details(result) -> None:
    rep = result.confidence
    rows = [
        ("1 · Vector Retrieval (Chroma + MiniLM)", result.retrieval_ms),
        ("2 · Gate A Reranker", rep.gate_a.latency_ms),
    ]
    if result.generation:
        rows.append(("3 · LLM Generation", result.generation.latency_ms))
    if rep.gate_b:
        rows.append(("4 · Gate B DeBERTa NLI", rep.gate_b.latency_ms))
    rows.append(("Total Pipeline Wall Time", result.total_ms))
    st.dataframe(
        {
            "Pipeline Stage": [r[0] for r in rows],
            "Latency (ms)": [round(r[1], 1) for r in rows],
            "Share (%)": [round((r[1] / max(result.total_ms, 0.1)) * 100, 1) for r in rows],
        },
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Provider `{settings.llm.provider}` · Model `{settings.llm.generator_model}` · "
        f"Verifier `{settings.confidence.scorer}` · Premise `{settings.confidence.nli_premise}` · "
        f"Hypothesis `{settings.confidence.nli_hypothesis}` · Persisted to SQLite (`data/runs.sqlite3`)."
    )
    a, b = rep.gate_a, rep.gate_b
    if a.llm_reason or (b and b.llm_reason):
        st.markdown("##### DeepEval LLM Judge Reasoning")
        if a.llm_relevancy is not None:
            st.write(f"**Contextual Relevancy ({a.llm_relevancy:.3f}):** {a.llm_reason}")
        if b is not None and b.llm_faithfulness is not None:
            st.write(f"**Faithfulness ({b.llm_faithfulness:.3f}):** {b.llm_reason}")


def _intro() -> None:
    st.markdown("##### 🏗️ Two-Gate Selective Prediction Architecture")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card(
            "1 · Pre-Generation Gate A (Sufficiency)",
            "Scores query-passage relevance via ms-marco-MiniLM-L-6-v2 before calling the LLM. "
            "Out-of-scope queries abstain in ~15 ms with zero LLM token cost.",
        )
    with c2:
        ui.card(
            "2 · Post-Generation Gate B v2 (Groundedness)",
            "Splits the generated answer into atomic sentences and tests NLI entailment against "
            "retrieved passages + 3-sentence windows using DeBERTa-v3.",
        )
    with c3:
        ui.card(
            "3 · Harmonic Composite & RAGTruth Audit",
            "Combines Gate A (0.4) and Gate B (0.6) via a weighted harmonic mean and applies a "
            "hard contradiction veto (0.15 cap) on any Evident Conflict.",
        )


def main() -> None:
    ui.inject_css()
    opts = _sidebar()

    ui.hero(
        "Confidence-Aware RAG Studio",
        "Selective retrieval-augmented generation with inline pre-generation relevance gating, "
        "post-generation DeBERTa NLI claim verification, and SQLite telemetry.",
    )

    question = _ask_form()
    raw_res = _run(question, opts) if question else None

    if raw_res is None:
        ui.pipeline_strip(None, baseline=opts["baseline"], prompt_only=opts["prompt_only"])
        _intro()
        _index_status(opts["index_slot"])
        return

    if opts["compare_both"] and isinstance(raw_res, tuple):
        guarded_res, base_res = raw_res
        ui.pipeline_strip(guarded_res, baseline=False, prompt_only=False)
        _render_side_by_side(guarded_res, base_res)
        primary = guarded_res
    else:
        primary = raw_res
        ui.pipeline_strip(primary, baseline=opts["baseline"], prompt_only=opts["prompt_only"])
        _render_answer(primary, opts["baseline"], opts["prompt_only"])

    t1, t2, t3, t4 = st.tabs(
        [
            "🎯 Confidence Gauges",
            "🔬 Claim-Level NLI Check",
            "📚 Retrieved Passages",
            "⏱️ Latency & Telemetry",
        ]
    )
    with t1:
        _tab_scores(primary.confidence)
    with t2:
        _tab_claims(primary.confidence)
    with t3:
        _tab_sources(primary)
    with t4:
        _tab_details(primary)
    _index_status(opts["index_slot"])


if __name__ == "__main__":
    main()
