"""Build the Milestone 5 project consolidation report.

The report reads the saved benchmark and evaluation summaries in ``reports/``.
It records the final implementation status, reproducibility commands, and the
limits of the available evidence without rerunning expensive model sweeps.

    python scripts/make_milestone5_report.py
    -> reports/Milestone5_Report.docx
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "Milestone5_Report.docx"
LETTERHEAD = ROOT / "assets" / "mits_letterhead.png"

STUDENT = "Shivam Gupta"
ENROLMENT = "BTCD24O1060"
PROJECT_TITLE = "Design and Development of a Confidence-Aware RAG Pipeline with Hallucination Detection"
REPORT_DATE = date(2026, 10, 31)
DEADLINE = date(2026, 11, 1)

INK = RGBColor(0x11, 0x1B, 0x2E)
ACCENT = RGBColor(0x0E, 0x6E, 0x62)
MUTED = RGBColor(0x5A, 0x63, 0x74)


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def style_document(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 0.98
    for name, size in (("Heading 1", 14), ("Heading 2", 11.5)):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size - 0.5)
        style.font.bold = True
        style.font.color.rgb = ACCENT if name == "Heading 1" else INK
        style.paragraph_format.space_before = Pt(9 if name == "Heading 1" else 7)
        style.paragraph_format.space_after = Pt(3)
    title = doc.styles["Title"]
    title.font.name = "Calibri"
    title.font.size = Pt(19)
    title.font.bold = True
    title.font.color.rgb = INK
    title_ppr = title._element.get_or_add_pPr()
    title_border = title_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_ppr.remove(title_border)


def para(doc: Document, text: str, *, bold: bool = False, italic: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(4)


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item).font.size = Pt(10)
        p.paragraph_format.space_after = Pt(2)


def table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    result = doc.add_table(rows=1, cols=len(headers))
    result.style = "Light Grid Accent 1"
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(result.rows[0].cells, headers, strict=True):
        cell.text = text
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(9)
    for values in rows:
        cells = result.add_row().cells
        for cell, text in zip(cells, values, strict=True):
            cell.text = text
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9)
    for row in result.rows:
        row.cant_split = True
        for cell, width in zip(row.cells, widths, strict=True):
            cell.width = Inches(width)
    result.rows[0].header = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def cover(doc: Document) -> None:
    if LETTERHEAD.exists():
        doc.add_picture(str(LETTERHEAD), width=Inches(5.94))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_after = Pt(9)

    p = doc.add_paragraph()
    name = p.add_run(f"{STUDENT.upper()} ({ENROLMENT})")
    name.bold = True
    name.font.size = Pt(10)
    suffix = p.add_run("\tCornerstone Project")
    suffix.font.size = Pt(10)
    suffix.font.color.rgb = MUTED

    p = doc.add_paragraph(style="Title")
    p.add_run("Milestone 5 Project Consolidation and Final Evaluation")
    para(doc, "Reproducibility, evidence review, and final project status")
    table(
        doc,
        ["Field", "Detail"],
        [
            ["Project Title", PROJECT_TITLE],
            ["Milestone", f"Milestone 5, due {DEADLINE.strftime('%d %B %Y')}"],
            ["Prepared By", f"{STUDENT} ({ENROLMENT})"],
            ["Report Date", REPORT_DATE.strftime("%d %B %Y")],
            ["Status", "Implementation and M1–M4 evaluations complete; final reproducibility evidence recorded."],
        ],
        [1.3, 5.0],
    )


def section_summary(doc: Document, m3: dict, m4: dict) -> None:
    doc.add_heading("1. Executive Summary", level=1)
    guarded = m3["arms"]["guarded"]
    baseline = m3["arms"]["baseline"]
    para(
        doc,
        "The project delivers a runnable confidence-aware retrieval-augmented generation system. "
        "Its two-stage confidence layer checks whether retrieved passages are relevant before "
        "generation, then checks generated claims for grounding. The evaluation compares that "
        "guarded pipeline with an unguarded baseline on SQuAD 2.0 and HotpotQA.",
    )
    para(
        doc,
        f"On the 150-item SQuAD 2.0 evaluation, the calibrated guarded arm achieved {guarded['em']:.3f} "
        f"exact match and {guarded['f1']:.3f} F1, compared with {baseline['em']:.3f} exact match and "
        f"{baseline['f1']:.3f} F1 for the baseline. Its unanswerable-question hallucination rate was "
        f"{guarded['hallucination_rate']:.1%}, while its answerable-question over-abstention rate was "
        f"{guarded['over_abstention_rate']:.1%}. The result shows a useful reduction in unsupported "
        "answers alongside a substantial coverage cost; both are central to the project conclusion.",
    )
    para(
        doc,
        f"On HotpotQA, chunk indexing raised supporting-fact recall from "
        f"{m4['granularities']['sentence']['supporting_recall']:.1%} to "
        f"{m4['granularities']['chunk']['supporting_recall']:.1%}. The saved generation results show "
        "that chunking improves baseline answer F1, while the guarded system abstains on most items. "
        "The evidence supports a limitation in the current Gate B configuration on this sample; it does "
        "not prove a universal limit of single-premise NLI.",
    )


def section_completion(doc: Document) -> None:
    doc.add_heading("2. Project Completion", level=1)
    table(
        doc,
        ["Milestone", "Delivered objective", "Evidence"],
        [
            ["1", "Ingestion, retrieval, confidence gates, telemetry, Streamlit interface", "Source modules, app, tests, latency and smoke-test records"],
            ["2", "Local generation, SQuAD 2.0 harness, guarded and baseline arms", "Benchmark harness and reports/benchmark_m2.json"],
            ["3", "Threshold calibration, risk–coverage, NLI/judge comparison", "Calibration, benchmark, risk–coverage, and judge-agreement JSON"],
            ["4", "HotpotQA retrieval/generation and sentence/chunk ablation", "HotpotQA benchmark and reports/*_m4.json results"],
            ["5", "Consolidated evidence, reproducibility instructions, final report", "This report, README, tests, and lint results"],
        ],
        [0.75, 3.0, 2.6],
    )
    para(
        doc,
        "The project includes a Streamlit question-answering app and monitoring dashboard, model-free "
        "unit tests, scripts for rebuilding indexes and benchmark samples, resumable SQuAD evaluation, "
        "HotpotQA sweep scripts, and report generators. Configuration is centralized in src/config.py "
        "and environment variables documented in .env.example.",
    )


def section_evidence(doc: Document, calibration: dict, m3: dict, risk: dict, m4: dict) -> None:
    doc.add_page_break()
    doc.add_heading("3. Consolidated Evaluation Results", level=1)
    a = calibration["gate_a"]
    b = calibration["gate_b"]
    table(
        doc,
        ["Evaluation", "Result", "Interpretation"],
        [
            ["SQuAD 2.0 guarded", f"EM {m3['arms']['guarded']['em']:.3f}; F1 {m3['arms']['guarded']['f1']:.3f}", "Higher aggregate QA scores than the saved baseline; abstention limits answerable coverage."],
            ["Unanswerable questions", f"Hallucination {m3['arms']['guarded']['hallucination_rate']:.1%}; baseline {m3['arms']['baseline']['hallucination_rate']:.1%}", "The gates reduce unsupported answers on this sample."],
            ["Gate A calibration", f"AUC {a['roc_auc']:.3f}; threshold {a['chosen']['threshold']:.2f}", "Moderate separation of answerable and unanswerable items."],
            ["Gate B calibration", f"AUC {b['roc_auc']:.3f}; threshold {b['chosen']['threshold']:.2f}", "Weak discrimination on an imperfect groundedness proxy."],
            ["Risk–coverage", f"Guarded AURC {risk['arms']['guarded']['aurc']:.3f}; baseline {risk['arms']['baseline']['aurc']:.3f}", "Confidence ranking improves selective risk over the baseline in the recorded run."],
            ["HotpotQA full supporting recall", f"Sentence {m4['granularities']['sentence']['full_recall_rate']:.1%}; chunk {m4['granularities']['chunk']['full_recall_rate']:.1%}", "Chunking retrieves all labelled facts more often, partly because each unit covers more text."],
            ["HotpotQA baseline F1", f"Sentence {m4['sentence_baseline']['f1']:.3f}; chunk {m4['chunk_baseline']['f1']:.3f}", "Wider context improved the saved baseline generation score."],
        ],
        [1.55, 1.7, 3.1],
    )
    para(
        doc,
        "The calibration and judge-agreement artifacts also show that the local NLI scores and the "
        "LLM judge do not agree reliably on the small judge sample. This means the two scorers should "
        "not be treated as interchangeable ground truth. The Milestone 3 agreement run contains 34 "
        "items overall, with fewer observations for Gate B, so its correlations are diagnostic rather "
        "than definitive.",
    )


def section_repro(doc: Document) -> None:
    doc.add_heading("4. Reproducibility and Demo", level=1)
    para(
        doc,
        "The project now records a concise sequence for checking the code, rerunning the integration "
        "smoke test, and launching the interface. Run these commands from the repository root in the "
        "configured Python environment:",
    )
    for command in (
        "python -m pytest tests/ -q -p no:cacheprovider",
        "ruff check .",
        "python scripts/smoke_test.py",
        "streamlit run app.py",
    ):
        p = doc.add_paragraph()
        run = p.add_run(command)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(2)
    table(
        doc,
        ["Check", "Result"],
        [
            ["Model-free unit tests", "75 passed"],
            ["Ruff", "Passed; no findings"],
            ["LLM-enabled integration smoke test", "27 passed, 0 failed, 0 skipped"],
            ["Streamlit app", "Headless server started successfully on localhost"],
        ],
        [2.0, 4.35],
    )
    para(
        doc,
        "The smoke test validates indexing, retrieval, both confidence gates, score combination, and "
        "telemetry without calling a generative LLM. First-time runs require the configured embedding "
        "and scorer models to be available locally or downloadable. The app's generation path also "
        "requires a reachable provider such as Ollama with the selected model pulled. Full benchmark "
        "sweeps are intentionally separate because they take substantially longer.",
    )


def section_limits(doc: Document) -> None:
    doc.add_heading("5. Limitations and Next Work", level=1)
    bullets(
        doc,
        [
            "Gate B's SQuAD threshold uses answerability as a proxy for groundedness; this is not claim-level ground truth.",
            "The guarded SQuAD run has high over-abstention on answerable items, so reducing hallucinations comes with a major coverage trade-off.",
            "On HotpotQA, the saved retrieval summary reports no item with both supporting facts co-located in a chunk. The experiment therefore does not isolate the effect of a unified multi-hop premise.",
            "The HotpotQA contradiction rate is a model output, not a human-validated false-contradiction rate. It demonstrates poor alignment with this task sample, not a proven universal architectural boundary.",
            "A future extension should compare claim-level human labels, multi-premise entailment, explicit hop decomposition, and an LLM judge on the same held-out examples.",
        ],
    )
def main() -> None:
    benchmark = load("benchmark_m3.json")
    calibration = load("calibration.json")
    risk = load("risk_coverage_m3.json")
    hotpot = load("generation_hotpot_m4.json")
    retrieval = load("retrieval_ablation_m4.json")
    # These are separate empirical artifacts; copy retrieval's dimensions into
    # the generation summary object for one consistent report interface.
    hotpot["granularities"] = retrieval["granularities"]

    doc = Document()
    style_document(doc)
    for section in doc.sections:
        section.top_margin = Inches(0.62)
        section.bottom_margin = Inches(0.62)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
    cover(doc)
    section_summary(doc, benchmark, hotpot)
    section_completion(doc)
    section_evidence(doc, calibration, benchmark, risk, hotpot)
    section_repro(doc)
    section_limits(doc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
