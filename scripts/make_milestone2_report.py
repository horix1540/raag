"""Generate the Milestone 2 report as a .docx for the project portal.

Milestone 2: SQuAD 2.0 benchmark harness built and run, baseline control arm
established, placeholder thresholds used to measure the behavior before
calibration. Numbers match reports/benchmark_m2.json and
data/runs.sqlite3 run_tag='squad2-dev'.

    .venv/Scripts/python.exe scripts/make_milestone2_report.py
    -> reports/Milestone2_Report.docx
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "Milestone2_Report.docx"
LETTERHEAD = ROOT / "assets" / "mits_letterhead.png"
BENCHMARK_JSON = ROOT / "reports" / "benchmark_m2.json"

TITLE = "Design and Development of a Confidence-Aware RAG Pipeline with Hallucination Detection"
STUDENT = "Shivam Gupta"
ENROLMENT = "BTCD24O1060"
DUE = date(2026, 9, 3)

INK = RGBColor(0x11, 0x1B, 0x2E)
ACCENT = RGBColor(0x0E, 0x6E, 0x62)
MUTED = RGBColor(0x5A, 0x63, 0x74)


def style_document(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1

    for name, size in (("Heading 1", 14), ("Heading 2", 11.5)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = ACCENT if name == "Heading 1" else INK
        st.paragraph_format.space_before = Pt(11 if name == "Heading 1" else 8)
        st.paragraph_format.space_after = Pt(4)


def para(
    doc: Document,
    text: str,
    *,
    size: float = 10.5,
    bold: bool = False,
    italic: bool = False,
    colour: RGBColor | None = None,
    align=None,
    space_after: float = 6,
) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
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


def table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
) -> None:
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


def cover(doc: Document) -> None:
    if LETTERHEAD.exists():
        doc.add_picture(str(LETTERHEAD), width=Inches(5.94))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_after = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    para(doc, TITLE, size=20, bold=True, colour=INK, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    para(
        doc,
        "Milestone 2: Benchmark Harness and Baseline Results",
        size=16,
        colour=ACCENT,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=2,
    )


def section_objective(doc: Document) -> None:
    doc.add_heading("1. Objective", level=1)
    para(
        doc,
        "This milestone was about building the evaluation infrastructure and establishing a "
        "baseline to measure against. The question Milestone 2 had to answer was whether the "
        "two-gate confidence layer does anything useful at all before calibration, because "
        "asserting that it does without running the comparison would be claiming a result rather "
        "than measuring one. Three things had to be built: a labelled benchmark sample from "
        "SQuAD 2.0 with answerable and unanswerable questions, a harness that runs every item "
        "through two arms of the pipeline, and the per-arm scoring that produces the numbers "
        "this report argues from.",
    )


def section_approach(doc: Document) -> None:
    doc.add_heading("2. Approach", level=1)

    doc.add_heading("2.1 Why SQuAD 2.0 and not 1.1", level=2)
    para(
        doc,
        "Version 2.0 deliberately includes questions its own context cannot answer, written by "
        "crowdworkers to look answerable so the model cannot shortcut the decision. That "
        "unanswerable label is exactly what this project needs. An unanswerable item is one where "
        "the correct behavior is abstention, which gives Gate A a ground truth it can be judged "
        "against instead of a threshold chosen by eye. Without that label the harness could only "
        "report how often the system refused, not whether those refusals were justified.",
    )

    doc.add_heading("2.2 The two arms", level=2)
    para(
        doc,
        "Every question is run twice, through two configurations of the same pipeline. The "
        "guarded arm is the real system: Gate A can refuse before generation, and Gate B can "
        "suppress the answer afterwards if it is ungrounded. The baseline arm is the control: "
        "gates are measured but never enforced, which is what a conventional RAG system does. "
        "The comparison isolates the confidence layer: same retrieval, same generator, same "
        "contexts indexed, differing only in whether the gates' decisions are honored. Measuring "
        "the baseline on the same split with the same LLM means the delta is the confidence "
        "layer's contribution, not a model upgrade or a corpus change.",
    )

    doc.add_heading("2.3 Distractor paragraphs", level=2)
    para(
        doc,
        "Each item is stored with its gold paragraph plus four distractor paragraphs drawn from "
        "different Wikipedia articles. The distractors matter: with only the gold paragraph in "
        "the index, retrieval has nothing to choose between, so top_similarity and "
        "similarity_margin degenerate and the harness would measure the reranker in isolation "
        "rather than the pipeline. Distractors are taken from other titles so none of them can "
        "accidentally contain the answer. The harness builds a throwaway per-item index for every "
        "question, runs both arms, logs the results, and discards the index.",
    )


def section_results(doc: Document, data: dict) -> None:
    doc.add_heading("3. Results", level=1)
    g = data["arms"]["guarded"]
    b = data["arms"]["baseline"]

    para(
        doc,
        f"150 questions (75 answerable, 75 unanswerable), run through both arms. Total {data['n_rows']} "
        f"telemetry rows. The guarded arm answered {g['answered']} and abstained {g['abstained']} times; "
        f"the baseline answered every question. Thresholds were the uncalibrated placeholders from "
        f"Milestone 1: Gate A 0.45, Gate B 0.60, composite 0.55.",
    )
    para(
        doc,
        "One point about how the control arm was obtained, since it affects how the comparison "
        "should be read. The baseline ignores both gates by construction, so its output cannot "
        "depend on any threshold. It was therefore generated once and the same 150 rows are "
        "referenced by every later threshold experiment rather than being regenerated for each. "
        "Re-running it would have changed nothing except the generator's own sampling noise, "
        "which would have made the guarded-versus-baseline delta partly a measure of that noise. "
        "The consequence to be aware of is that the baseline column carries no run-to-run "
        "variance estimate of its own.",
    )

    table(
        doc,
        ["Metric", "Guarded", "Baseline", "Difference"],
        [
            [
                "EM (exact match, aggregate)",
                f"{g['em']:.3f}",
                f"{b['em']:.3f}",
                f"{g['em'] - b['em']:+.3f} (see 3.1)",
            ],
            [
                "F1 (aggregate)",
                f"{g['f1']:.3f}",
                f"{b['f1']:.3f}",
                f"{g['f1'] - b['f1']:+.3f} (see 3.1)",
            ],
            [
                "F1 on answerable half",
                f"{g['confusion']['answerable']['mean_f1']:.3f}",
                f"{b['confusion']['answerable']['mean_f1']:.3f}",
                f"{g['confusion']['answerable']['mean_f1'] - b['confusion']['answerable']['mean_f1']:+.3f}",
            ],
            [
                "Abstention rate",
                f"{g['abstention_rate']:.1%}",
                f"{b['abstention_rate']:.1%}",
                "—",
            ],
            [
                "Hallucination rate\n(unanswerable answered)",
                f"{g['hallucination_rate']:.1%}",
                f"{b['hallucination_rate']:.1%}",
                f"{g['hallucination_rate'] - b['hallucination_rate']:+.1%}",
            ],
            [
                "Over-abstention rate\n(answerable refused)",
                f"{g['over_abstention_rate']:.1%}",
                f"{b['over_abstention_rate']:.1%}",
                "—",
            ],
        ],
        widths=[2.0, 1.2, 1.2, 1.35],
    )

    doc.add_heading("3.1 What the numbers mean", level=2)
    para(
        doc,
        f"The guarded arm's hallucination rate is {g['hallucination_rate']:.1%}, meaning it produced an "
        f"answer to {int(g['hallucination_rate']*75)} of the 75 unanswerable questions. The baseline answered "
        f"every unanswerable question (100% hallucination). The confidence layer suppressed {int((1-g['hallucination_rate'])*75)} "
        f"fabrications at the cost of refusing {int(g['over_abstention_rate']*75)} answerable questions "
        f"({g['over_abstention_rate']:.1%} over-abstention).",
    )
    para(
        doc,
        f"The aggregate F1 needs care, and this is the number most open to misreading. The guarded "
        f"arm's F1 is {g['f1']:.3f} against the baseline's {b['f1']:.3f}, which looks like the "
        f"confidence layer made the system more accurate. It did not. Under the official SQuAD 2.0 "
        f"scoring rule an unanswerable question's gold answer is the empty string, so abstaining on "
        f"an unanswerable item scores F1 = 1.0. The guarded arm abstains on "
        f"{g['confusion']['unanswerable']['abstained']} of the 75 unanswerable items and therefore "
        f"averages {g['confusion']['unanswerable']['mean_f1']:.3f} on that half, while the baseline "
        f"answers all of them and averages {b['confusion']['unanswerable']['mean_f1']:.3f}. The "
        f"aggregate gap is almost entirely that one effect.",
    )
    para(
        doc,
        f"On the answerable half the ranking reverses, which is the honest comparison: the guarded "
        f"arm averages F1 {g['confusion']['answerable']['mean_f1']:.3f} against the baseline's "
        f"{b['confusion']['answerable']['mean_f1']:.3f}, because it refuses "
        f"{g['confusion']['answerable']['abstained']} questions it was asked to answer. So the "
        f"aggregate should not be read as an accuracy win. It is the arithmetic of a system that "
        f"abstains on nine questions in ten being rewarded for the abstentions that happened to be "
        f"correct. The per-condition breakdown is the table to argue from:",
    )

    table(
        doc,
        ["Condition", "Guarded answered", "Guarded abstained", "Baseline answered"],
        [
            [
                "Answerable (75 items)",
                str(g["confusion"]["answerable"]["answered"]),
                str(g["confusion"]["answerable"]["abstained"]),
                str(b["confusion"]["answerable"]["answered"]),
            ],
            [
                "Unanswerable (75 items)",
                str(g["confusion"]["unanswerable"]["answered"]),
                str(g["confusion"]["unanswerable"]["abstained"]),
                str(b["confusion"]["unanswerable"]["answered"]),
            ],
        ],
        widths=[1.8, 1.5, 1.7, 1.5],
    )

    para(
        doc,
        f"Read that table on its own terms. Of the 75 unanswerable questions the guarded arm let "
        f"only {g['confusion']['unanswerable']['answered']} through, against all 75 for the "
        f"baseline, so the gates suppressed "
        f"{g['confusion']['unanswerable']['abstained']} fabrications that a conventional pipeline "
        f"would have presented as answers. That is the result this milestone set out to obtain. "
        f"The price is in the row above it: {g['confusion']['answerable']['abstained']} answerable "
        f"questions refused. Both facts follow from the same setting, and with thresholds nobody "
        f"fitted there is no reason to expect the balance between them to be sensible. Choosing "
        f"that balance from labelled data is Milestone 3.",
    )


def section_latency(doc: Document) -> None:
    doc.add_heading("4. Latency", level=1)
    para(
        doc,
        "Measured on this milestone's 150-item run, local Ollama llama3.1:8b on the RTX 4060. "
        "Medians rather than means throughout: a handful of rows carry gate latencies of several "
        "hours because the laptop suspended mid-sweep while the wall-clock timer kept counting. "
        "Those rows destroy a mean and leave a median intact, so the median is the honest summary.",
    )
    table(
        doc,
        ["Stage", "Guarded median", "Baseline median"],
        [
            ["Retrieval", "16.8 ms", "15.1 ms"],
            ["Gate A", "23.6 ms", "1,432 ms"],
            ["Generation", "3,194 ms (102 items)", "3,970 ms (150 items)"],
            ["Gate B", "1,543 ms (72 items)", "195 ms"],
            ["Total per query", "2,927 ms", "5,878 ms"],
        ],
        widths=[1.8, 1.8, 1.8],
    )
    para(
        doc,
        "The guarded arm's total is roughly half the baseline's, and the reason is the ordering of "
        "the gates rather than anything clever: Gate A refused 48 items outright, so the generator "
        "was never called on them, and Gate B only ran on the 72 items that passed Gate A and "
        "produced text. The baseline always generates, so its generation figure is the full-sweep "
        "median. Both arms used the same provider and model, so the difference in generation "
        "latency is token count: the unguarded prompt does not instruct the model to abstain, so "
        "its answers run longer on average.",
    )
    para(
        doc,
        "Two figures in that table need a caveat rather than a defence. The baseline's Gate A "
        "median of 1,432 ms and the guarded arm's Gate B median of 1,543 ms are not the cost of "
        "the cross-encoders; the Milestone 1 bench measured those at 16 ms and 118 ms on warm "
        "models. These rows include model loading and GPU contention during a long unattended "
        "sweep, so they measure the sweep's conditions, not the layer's steady-state cost. "
        "Milestone 4 should re-measure the confidence layer in isolation with the models pinned "
        "warm before any latency claim is made from benchmark telemetry.",
    )
    para(
        doc,
        "One further point of provenance, stated because the numbers make it visible: the baseline "
        "arm was executed once and its 150 rows are reused for comparison against every guarded "
        "configuration. The baseline ignores thresholds by construction, so re-running it would "
        "change nothing except the generator's sampling noise, and holding it fixed makes the "
        "guarded-versus-baseline delta attributable to the gates alone. It does mean the baseline "
        "column is identical in this report and in Milestone 3's.",
    )


def section_status(doc: Document) -> None:
    doc.add_heading("5. Status and Next Steps", level=1)
    para(
        doc,
        "Milestone 2 is complete: the harness runs, both arms produce results, and the telemetry "
        "records every field the calibration and agreement studies will need. What the numbers show "
        "is that the placeholder thresholds are too conservative. An 84% over-abstention rate means "
        "the system is refusing to answer nearly everything, which makes it useless in practice even "
        "though it does suppress hallucinations.",
    )
    bullets(
        doc,
        [
            "The thresholds are still placeholders. Milestone 3 fits them against the answerable/"
            "unanswerable labels using the full feature set (rerank score, cosine statistics, "
            "contradiction cap) rather than single cut-offs chosen by eye.",
            "Gate B's contradiction cap was decoupled from the threshold during this milestone, so "
            "calibration cannot weaken it. The self-abstention signal (generator refusal) is now "
            "persisted as a separate flag in telemetry, so the next analysis can separate what the "
            "confidence layer contributed from what the generator refused on its own.",
            "Claim-level entailment and the per-chunk relevance distribution are now logged, which "
            "the calibration script will use to fit a logistic model over all four retrieval features "
            "instead of thresholding one.",
        ],
    )
    para(
        doc,
        "The next step is threshold calibration and the LLM-judge agreement study, both of which "
        "depend on the telemetry this milestone produced.",
    )


def main() -> int:
    if not BENCHMARK_JSON.exists():
        print(f"benchmark file not found: {BENCHMARK_JSON}")
        print("run: .venv/Scripts/python.exe scripts/eval_benchmark.py --tag squad2-dev")
        return 1

    data = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    if data.get("run_tag") != "squad2-dev":
        print(
            f"warning: benchmark_m2.json run_tag is {data.get('run_tag')!r}, expected 'squad2-dev'"
        )

    doc = Document()
    style_document(doc)
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    cover(doc)
    section_objective(doc)
    section_approach(doc)
    section_results(doc, data)
    section_latency(doc)
    section_status(doc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
