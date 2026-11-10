"""Generate the Milestone 4 report as a .docx for the project portal.

Covers: HotpotQA multi-hop evaluation and chunk vs. sentence granularity
ablation.  Reads empirical data from:

    reports/retrieval_ablation_m4.json
    reports/generation_hotpot_m4.json

Usage:

    .venv/Scripts/python.exe scripts/make_milestone4_report.py
    -> reports/Milestone4_Report.docx
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "Milestone4_Report.docx"
LETTERHEAD = ROOT / "assets" / "mits_letterhead.png"
DIAGRAM = ROOT / "assets" / "thumbnail.png"

RETRIEVAL_JSON = ROOT / "reports" / "retrieval_ablation_m4.json"
GENERATION_JSON = ROOT / "reports" / "generation_hotpot_m4.json"

TITLE = "Design and Development of a Confidence-Aware RAG Pipeline with Hallucination Detection"
STUDENT = "Shivam Gupta"
ENROLMENT = "BTCD24O1060"
DUE = date(2026, 9, 17)

INK = RGBColor(0x11, 0x1B, 0x2E)
ACCENT = RGBColor(0x0E, 0x6E, 0x62)
MUTED = RGBColor(0x5A, 0x63, 0x74)


# ── helpers (shared with make_milestone_report.py) ──────────────────────

def style_document(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.0

    for name, size in (("Heading 1", 14), ("Heading 2", 11.5)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = ACCENT if name == "Heading 1" else INK
        st.paragraph_format.space_before = Pt(9 if name == "Heading 1" else 7)
        st.paragraph_format.space_after = Pt(3)


def para(doc: Document, text: str, *, size: float = 10.5, italic: bool = False,
         bold: bool = False, colour: RGBColor | None = None, align=None,
         space_after: float = 5) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    run.bold = bold
    if colour is not None:
        run.font.color.rgb = colour
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item).font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(3)


def table(doc: Document, headers: list[str], rows: list[list[str]],
          widths: list[float] | None = None) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(t.rows[0].cells, headers, strict=True):
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        for cell, text in zip(t.add_row().cells, row, strict=True):
            cell.text = ""
            cell.paragraphs[0].add_run(text).font.size = Pt(9)
    if widths:
        for row in t.rows:
            for cell, w in zip(row.cells, widths, strict=True):
                cell.width = Inches(w)
    t.rows[0].header = True
    for row in t.rows:
        row.cant_split = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ── load empirical data ─────────────────────────────────────────────────

def load_data() -> tuple[dict, dict]:
    with open(RETRIEVAL_JSON) as f:
        retrieval = json.load(f)
    with open(GENERATION_JSON) as f:
        generation = json.load(f)
    return retrieval, generation


# ── sections ─────────────────────────────────────────────────────────────

def cover(doc: Document) -> None:
    if LETTERHEAD.exists():
        doc.add_picture(str(LETTERHEAD), width=Inches(5.94))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_after = Pt(10)

    p = doc.add_paragraph()
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.3), WD_TAB_ALIGNMENT.RIGHT)
    left = p.add_run(f"{STUDENT.upper()} ({ENROLMENT})")
    left.bold = True
    left.font.size = Pt(10)
    right = p.add_run("\tCornerstone Project")
    right.font.size = Pt(10)
    right.font.color.rgb = MUTED
    p.paragraph_format.space_after = Pt(12)

    para(doc, "Milestone 4", size=20, colour=ACCENT, space_after=2)
    para(doc, "HotpotQA Multi-Hop Evaluation and Granularity Ablation",
         size=13, colour=INK, space_after=12)

    table(
        doc,
        ["Field", "Detail"],
        [
            ["Project Title", TITLE],
            ["Milestone", "Milestone 4: HotpotQA Multi-Hop Evaluation and "
                          "Granularity Ablation, due "
                          + DUE.strftime("%d %B %Y")],
            ["Prepared By", f"{STUDENT} ({ENROLMENT})"],
            ["Status", "Complete. Retrieval ablation, generation sweep, and "
                       "Gate B architectural diagnosis all finished and documented."],
        ],
        widths=[1.3, 5.0],
    )


def section_objective(doc: Document) -> None:
    doc.add_heading("1. Objective", level=1)
    para(doc,
         "Milestones 1 through 3 built and calibrated the confidence-aware pipeline on "
         "SQuAD 2.0, a single-hop factoid benchmark where the gold answer lives inside one "
         "paragraph. Milestone 4 asks two questions that SQuAD cannot answer. First, what "
         "happens to the pipeline when the question requires synthesising information across "
         "two or more passages — the multi-hop case that HotpotQA was designed for. Second, "
         "whether the granularity of the indexed text units matters: does chunking at ~800 "
         "characters outperform strict sentence-level indexing, and if so, what does the "
         "difference tell us about Gate B?")
    para(doc,
         "The expectation was that retrieval would improve with wider chunks because a single "
         "chunk is more likely to contain a complete supporting fact, and that Gate A would "
         "transfer well because the reranker sees each passage independently. The open "
         "question was Gate B: NLI entailment is defined over a single premise–hypothesis "
         "pair, and a multi-hop answer is by definition not entailed by any one passage alone.")


def section_retrieval(doc: Document, ret: dict) -> None:
    doc.add_heading("2. Retrieval Dynamics: Chunk vs. Sentence Indexing", level=1)

    sent = ret["granularities"]["sentence"]
    chunk = ret["granularities"]["chunk"]
    delta = ret["delta_chunk_minus_sentence"]

    para(doc,
         "The retrieval ablation indexed the same 150 HotpotQA items at two granularities "
         "— strict sentence splitting and the pipeline's default ~800-character boundary-aware "
         "chunking — and measured how many of HotpotQA's labelled supporting facts appeared in "
         f"the top-{ret['top_k']} retrieved results.")

    table(
        doc,
        ["Metric", "Sentence", "Chunk", "Δ (chunk − sentence)"],
        [
            ["Supporting-fact recall",
             f"{sent['supporting_recall']:.1%}",
             f"{chunk['supporting_recall']:.1%}",
             f"+{delta['supporting_recall']:.1%}"],
            ["Full recall (all facts retrieved)",
             f"{sent['full_recall_rate']:.1%}",
             f"{chunk['full_recall_rate']:.1%}",
             f"+{delta['full_recall_rate']:.1%}"],
            ["Supporting-fact precision",
             f"{sent['supporting_prec']:.1%}",
             f"{chunk['supporting_prec']:.1%}",
             f"+{delta['supporting_prec']:.1%}"],
            ["Mean units indexed per item",
             f"{sent['mean_units_indexed']:.1f}",
             f"{chunk['mean_units_indexed']:.1f}",
             "—"],
            ["Median retrieval latency",
             f"{sent['median_retrieval_ms']:.1f} ms",
             f"{chunk['median_retrieval_ms']:.1f} ms",
             "—"],
        ],
        widths=[1.9, 1.1, 1.1, 1.5],
    )

    para(doc,
         f"The headline number is full recall: {chunk['full_recall_rate']:.0%} of chunk-indexed "
         f"items had every labelled supporting fact somewhere in the top-{ret['top_k']}, compared "
         f"with {sent['full_recall_rate']:.0%} for sentences — a +{delta['full_recall_rate']:.1f} "
         "point improvement. This is not because chunking finds better matches. It is a volume "
         "effect: an 800-character chunk mechanically contains several sentences, so retrieving "
         f"{ret['top_k']} chunks covers far more source text than retrieving {ret['top_k']} "
         f"sentences. The sentence index averaged {sent['mean_units_indexed']:.0f} units per "
         f"item; the chunk index averaged {chunk['mean_units_indexed']:.0f}. The chunk dragnet "
         "simply sweeps wider.")

    para(doc,
         "Supporting-fact precision is low for both granularities — most of the retrieved text "
         "is not a labelled supporting fact — but this is expected at top-5 retrieval over "
         "multi-article corpora and does not indicate a retrieval failure. The reranker (Gate A) "
         "is designed to filter downstream, not the embedding search itself.")


def section_generation(doc: Document, gen: dict) -> None:
    doc.add_heading("3. Generation Performance", level=1)

    sb = gen["sentence_baseline"]
    cb = gen["chunk_baseline"]
    sg = gen["sentence_guarded"]
    cg = gen["chunk_guarded"]

    f1_improvement = (cb["f1"] - sb["f1"]) / sb["f1"] * 100

    para(doc,
         "The generation sweep ran all 150 items through both arms (guarded and baseline) at "
         f"both granularities, using {gen['llm_model']} as the generator.")

    table(
        doc,
        ["Metric", "Sentence\nbaseline", "Chunk\nbaseline",
         "Sentence\nguarded", "Chunk\nguarded"],
        [
            ["F1",
             f"{sb['f1']:.4f}", f"{cb['f1']:.4f}",
             f"{sg['f1']:.4f}", f"{cg['f1']:.4f}"],
            ["Exact match",
             f"{sb['exact_match']:.4f}", f"{cb['exact_match']:.4f}",
             f"{sg['exact_match']:.4f}", f"{cg['exact_match']:.4f}"],
            ["Answer rate",
             f"{sb['answer_rate']:.0%}", f"{cb['answer_rate']:.0%}",
             f"{sg['answer_rate']:.0%}", f"{cg['answer_rate']:.0%}"],
            ["Hallucination rate",
             f"{sb['hallucination_rate']:.0%}", f"{cb['hallucination_rate']:.0%}",
             f"{sg['hallucination_rate']:.0%}", f"{cg['hallucination_rate']:.0%}"],
            ["Gate B contradiction rate",
             f"{sb['gate_b_contradiction_rate']:.0%}",
             f"{cb['gate_b_contradiction_rate']:.0%}",
             f"{sg['gate_b_contradiction_rate']:.0%}",
             f"{cg['gate_b_contradiction_rate']:.0%}"],
            ["Mean reranker relevancy",
             f"{sb['mean_rerank_relevancy']:.4f}",
             f"{cb['mean_rerank_relevancy']:.4f}",
             f"{sg['mean_rerank_relevancy']:.4f}",
             f"{cg['mean_rerank_relevancy']:.4f}"],
            ["Mean NLI faithfulness",
             f"{sb['mean_nli_faithfulness']:.4f}",
             f"{cb['mean_nli_faithfulness']:.4f}",
             f"{sg['mean_nli_faithfulness']:.4f}",
             f"{cg['mean_nli_faithfulness']:.4f}"],
        ],
        widths=[1.5, 1.1, 1.1, 1.1, 1.1],
    )

    para(doc,
         f"Baseline F1 rose from {sb['f1']:.4f} (sentence) to {cb['f1']:.4f} (chunk), a "
         f"{f1_improvement:.0f}% improvement. The LLM successfully synthesises answers from "
         "the wider chunk context when no confidence gate intervenes. This confirms that "
         "the retrieval improvement translates into better generation — the extra context "
         "is not wasted.")

    para(doc,
         "Mean reranker relevancy is "
         f"{cb['mean_rerank_relevancy']:.2f} for chunks, above the calibrated 0.45 threshold. "
         "This average does not establish per-item Gate A accuracy or prove that the gate transfers "
         "cleanly; it does show that low mean reranker scores are not the explanation for the "
         "low guarded answer rate in this run.")


def section_confidence_bottleneck(doc: Document, gen: dict, retrieval: dict) -> None:
    doc.add_heading("4. The Confidence Bottleneck: Gate B and Multi-Hop NLI", level=1)

    sb = gen["sentence_baseline"]
    cb = gen["chunk_baseline"]

    doc.add_heading("4.1 The observed contradiction rate", level=2)
    para(doc,
         f"Gate B's contradiction rate on the baseline arm is {sb['gate_b_contradiction_rate']:.0%} "
         f"for sentence indexing and {cb['gate_b_contradiction_rate']:.0%} for chunk indexing. "
         "These are not hallucination rates — the baseline arm always answers, so the gate scores "
         "are computed and logged but not enforced. DeBERTa flags 70–75% of generated answers as "
         f"contradicted, while the chunk baseline achieves F1 {cb['f1']:.4f}. These scores do not "
         "show whether each contradiction is correct: the answers were not manually labelled for "
         "claim-level contradiction, and F1 is not a groundedness label.")

    para(doc,
         "The central finding is that the current Gate B configuration often rejects answers on "
         "this multi-hop sample. The experiment does not establish that all of those flags are "
         "false contradictions or identify one cause for every rejection.")

    doc.add_heading("4.2 A possible single-premise NLI failure mode", level=2)
    para(doc,
         "Natural language inference, as DeBERTa was trained on it, takes one premise and one "
         "hypothesis and returns entailment, neutral, or contradiction. The premise is a single "
         "text passage. The hypothesis is the claim to be checked. For the claim to be entailed, "
         "the premise alone must be sufficient to conclude it.")

    para(doc,
         "A multi-hop answer, by definition, requires combining information from two or more "
         "passages. Consider a HotpotQA question like 'What nationality is the director of "
         "Film X?' The answer requires: (1) Passage A states the director of Film X is Person Y; "
         "(2) Passage B states Person Y is of nationality Z. The correct answer 'Z' is not "
         "entailed by Passage A alone (which never mentions Z) and is not entailed by Passage B "
         "alone (which never mentions Film X). Each premise, evaluated independently, returns "
         "neutral or contradiction even when the combined reasoning is sound. This is a plausible "
         "failure mode for the observed results, but this experiment did not label individual "
         "cases to establish that it caused every Gate B rejection.")

    para(doc,
         "Gate B checks each claim against each retrieved passage independently, keeping the "
         "best entailment score. If supporting facts are split across passages, no individual "
         "premise may entail the complete claim. In that case, the score may remain low and the "
         "contradiction head may fire. The available metrics are consistent with this explanation, "
         "but do not prove it for each answer.")

    doc.add_heading("4.3 Architectural implication", level=2)
    para(doc,
         "The results raise an architectural concern, but do not establish a universal boundary. "
         "Changing the Gate B threshold would trade coverage against rejection and needs a held-out, "
         "human-labelled evaluation. The current study does not show that threshold tuning cannot "
         "improve the operating point, nor does it validate every contradiction label.")

    para(doc,
         "Useful follow-up comparisons include: (a) a multi-premise entailment model "
         "that accepts a set of passages and reasons over their conjunction, (b) an explicit "
         "reasoning chain that reduces the multi-hop claim to single-hop sub-claims before NLI "
         "scoring, or (c) replacing the NLI gate entirely with an LLM-as-judge that can read "
         "all passages at once. These are candidate designs, not conclusions tested in this "
         "milestone. The next evaluation should compare them on the same held-out answers with "
         "human claim-level labels.")

    doc.add_heading("4.4 What this means for the project's claim", level=2)
    para(doc,
         "The project's thesis is that a RAG pipeline should estimate how reliable its own "
         "answer is and decline to answer when that estimate is low. On single-hop questions "
         "(SQuAD 2.0), the calibrated run substantially reduced hallucinations on unanswerable "
         "questions, with high over-abstention on answerable questions. On HotpotQA, the guarded "
         "answer rate was low and Gate B frequently flagged generated answers. Because the dataset "
         "contains no cases where both supporting facts were co-located in a chunk, the run did "
         "not test whether a unified premise resolves the issue. The results identify a limitation "
         "to investigate, not a universal limit of multi-hop RAG.")

    colocated = retrieval["granularities"]["chunk"]["by_colocation"]["colocated"]["n"]
    para(doc,
         f"The retrieval ablation found {colocated} HotpotQA items with both supporting facts "
         "co-located in one chunk. This means the data cannot compare single-premise NLI on a "
         "unified two-hop premise against NLI on split supporting passages.")


def section_status(doc: Document) -> None:
    doc.add_heading("5. Status and Next Steps", level=1)

    para(doc,
         "Milestone 4 is complete. The retrieval ablation, generation sweep, and Gate B "
        "evaluation are finished, and the qualified Gate B analysis is documented. The empirical data is in "
         "reports/retrieval_ablation_m4.json and reports/generation_hotpot_m4.json.")

    doc.add_heading("5.1 Summary of findings", level=2)
    bullets(doc, [
        "800-character chunks act as a volume dragnet, mechanically boosting full recall by "
        "+35.3 points over strict sentence indexing.",
        "The LLM successfully synthesises answers from the wider chunk context, producing a "
        "44% F1 improvement in the baseline arm.",
        "Mean chunk reranker relevancy remains above the fitted threshold, although the aggregate "
        "does not establish per-item Gate A accuracy.",
        "Gate B flagged 70–75% of baseline answers as contradicted, but these labels were not "
        "human-validated. The experiment shows a sample-level limitation and motivates a "
        "multi-premise comparison; it does not prove a universal architectural boundary.",
    ])

    doc.add_heading("5.2 Milestone 5 handoff", level=2)
    table(
        doc,
        ["Milestone", "Due", "Focus"],
        [
            ["5", "01 Nov 2026",
             "Consolidated report prepared; tests, integration smoke checks, and app startup verified"],
        ],
        widths=[0.8, 1.2, 4.3],
    )


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    retrieval, generation = load_data()

    doc = Document()
    style_document(doc)
    for section in doc.sections:
        section.top_margin = Inches(0.62)
        section.bottom_margin = Inches(0.62)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    cover(doc)
    section_objective(doc)
    section_retrieval(doc, retrieval)
    section_generation(doc, generation)
    section_confidence_bottleneck(doc, generation, retrieval)
    section_status(doc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
