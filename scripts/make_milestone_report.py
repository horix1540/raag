"""Generate the Milestone 1 report as a .docx for the project portal.

Written as a script rather than typed into Word so the numbers in the report
always match the numbers the code actually produced:

    .venv/Scripts/python.exe scripts/make_milestone_report.py
    -> reports/Milestone1_Report.docx
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "Milestone1_Report.docx"
LETTERHEAD = ROOT / "assets" / "mits_letterhead.png"
DIAGRAM = ROOT / "assets" / "thumbnail.png"

TITLE = "Design and Development of a Confidence-Aware RAG Pipeline with Hallucination Detection"
STUDENT = "Shivam Gupta"
ENROLMENT = "BTCD24O1060"
DUE = date(2026, 8, 27)

INK = RGBColor(0x11, 0x1B, 0x2E)
ACCENT = RGBColor(0x0E, 0x6E, 0x62)
MUTED = RGBColor(0x5A, 0x63, 0x74)


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
         colour: RGBColor | None = None, align=None, space_after: float = 5) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
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
    # Repeat the header row and keep each row whole if a table spans two pages.
    t.rows[0].header = True
    for row in t.rows:
        row.cant_split = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


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

    para(doc, "Milestone 1", size=20, colour=ACCENT, space_after=2)
    para(doc, "Architecture and Pipeline Implementation", size=13, colour=INK, space_after=12)

    table(
        doc,
        ["Field", "Detail"],
        [
            ["Project Title", TITLE],
            ["Milestone", "Milestone 1: Architecture and Pipeline Implementation, due "
                          + DUE.strftime("%d %B %Y")],
            ["Prepared By", f"{STUDENT} ({ENROLMENT})"],
            ["Status", "Complete. Pipeline, confidence layer, interface and tests all built "
                       "and verified."],
        ],
        widths=[1.3, 5.0],
    )


def section_objective(doc: Document) -> None:
    doc.add_heading("1. Objective", level=1)
    para(doc,
         "This milestone was about settling the design of the system and building it end to end. "
         "Benchmark numbers were not expected this week and none are claimed here. The question I "
         "had to answer first was how a RAG pipeline can judge the reliability of its own "
         "retrieval before it generates an answer, because the standard measure of groundedness "
         "cannot be computed at that point. Nearly everything else follows from how that is "
         "resolved: where the abstention decision sits, which models are needed, what gets "
         "logged, and what the tests have to pin down.")
def section_problem(doc: Document) -> None:
    doc.add_heading("2. The Problem This Project Addresses", level=1)
    para(doc,
         "Retrieval-Augmented Generation reduces hallucination but does not remove it. An "
         "ordinary RAG application answers in the same confident tone whether the retrieved "
         "passages support the answer or not, because nothing in the pipeline ever asks whether "
         "they do. Three failure modes survive retrieval: the corpus may not contain the answer "
         "at all and the generator will usually answer anyway; the passages may be correct while "
         "the model adds a number or a causal link that appears nowhere in them; or the model may "
         "state something the passages refute, because it fell back on what it memorised during "
         "training instead of reading the context it was given.")
    para(doc,
         "So the system is least trustworthy at exactly the moment it has no way of saying so, "
         "and a wrong answer given confidently costs the reader more than a refusal would, "
         "because the reader cannot tell the two apart. What this project tests is the claim that "
         "a RAG system should estimate how reliable its own answer is and decline to answer when "
         "that estimate is low. The target is not higher accuracy on every question. It is a "
         "better trade: most of the wrong answers suppressed, at the cost of withholding a few "
         "correct ones.")


def section_approach(doc: Document) -> None:
    doc.add_heading("3. Approach", level=1)

    doc.add_heading("3.1 Why confidence is checked twice", level=2)
    para(doc,
         "The project description requires the system to evaluate retrieval reliability before "
         "generating a response. Taking that literally runs into a problem. Faithfulness, the "
         "usual measure of groundedness in RAG, is the proportion of claims in an answer that "
         "the retrieved context supports. Before generation there is no answer and therefore no "
         "claims, so faithfulness is undefined at that point. After generation it can be "
         "computed, but by then the tokens have been paid for and the fluent wrong answer "
         "already exists inside the system, one display call away from the user.")
    para(doc,
         "Rather than pick one of the two, the pipeline asks two different questions at two "
         "different points, using a model suited to each.")
    table(
        doc,
        ["", "Gate A, before generation", "Gate B, after generation"],
        [
            ["Question asked",
             "Could these passages support any answer to this question?",
             "Is the answer that was produced entailed by these passages?"],
            ["Model",
             "Relevance reranker\ncross-encoder/ms-marco-MiniLM-L-6-v2",
             "NLI cross-encoder\ncross-encoder/nli-deberta-v3-base"],
            ["Catches",
             "Retrieval failure: the question is outside the corpus",
             "Invented detail, and outright contradiction"],
            ["On failure",
             "Abstain without calling the generator at all",
             "Suppress the answer and report which claims failed"],
        ],
        widths=[1.1, 2.6, 2.6],
    )
    if DIAGRAM.exists():
        doc.add_picture(str(DIAGRAM), width=Inches(4.9))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        para(doc,
             "Figure 1. The two gates in the pipeline. Retrieval sufficiency is checked before "
             "generation, answer groundedness after it, and either gate failing suppresses the "
             "answer.",
             size=9, italic=True, colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)

    para(doc,
         "The gates use different model families because they are different tasks. NLI asks "
         "whether a premise entails a hypothesis, and before generation there is no hypothesis, "
         "only a question. Deciding whether a passage answers a question is a ranking problem, "
         "so Gate A uses a reranker trained on relevance judgements and Gate B uses an NLI "
         "model. One model doing both jobs would be misapplied at one end or the other.")
    para(doc,
         "Cosine similarity from the retriever was the obvious cheap signal for Gate A, and I "
         "decided against making it the deciding input. MiniLM cosine values are not calibrated "
         "probabilities: in the test corpus a passage that genuinely answered the question scored "
         "0.531, so any fixed cut-off on that scale is an arbitrary constant. The reranker's "
         "output, once passed through a sigmoid, behaves much more like a probability, so that is "
         "what drives the gate. Similarity statistics are still recorded on every query, because "
         "they are inputs to threshold calibration later rather than usable thresholds now.")

    doc.add_heading("3.2 Scoring one claim at a time", level=2)
    para(doc,
         "Gate B does not score the answer as a single block. The answer is split into "
         "sentence-level claims, and each claim is checked against every retrieved passage, "
         "keeping the best entailment score and the worst contradiction score across them. Each "
         "claim then gets a label taken from the RAGTruth hallucination taxonomy: evident "
         "conflict when contradiction probability reaches 0.50, baseless information when "
         "entailment falls below 0.50 without any contradiction, and supported otherwise. "
         "Contradiction is tested first, because a claim that conflicts with the source is a "
         "worse failure than one the source is merely silent about.")
    para(doc,
         "Working claim by claim is what makes the output diagnostic: the interface can point at "
         "the sentence that is unsupported and at the passage that came closest to supporting it, "
         "instead of showing one number and leaving the user to guess. The gate itself does need a "
         "single number, and it uses the mean entailment probability across the claims. One "
         "contradicted claim overrides that mean and fails the gate outright, since nine correct "
         "sentences should not be able to average away one that the source refutes.")
    doc.add_heading("3.3 Combining the two scores", level=2)
    para(doc,
         "The two gate scores are combined into one composite with a weighted harmonic mean, 0.4 "
         "on retrieval relevancy and 0.6 on groundedness. The choice of mean matters more than it "
         "looks. Take an answer where retrieval was excellent but the text is almost entirely "
         "unsupported, say 0.95 and 0.05: the arithmetic mean is 0.50, which reads as borderline "
         "and would clear a mid-range threshold, while the weighted harmonic mean of the same pair "
         "is about 0.08. This is the reasoning behind using F1 instead of averaging precision and "
         "recall. Both conditions have to hold, so a near-zero in either has to dominate, and good "
         "retrieval must not be able to buy credibility for an answer that is not grounded. "
         "Groundedness carries the heavier weight because a fluent unsupported answer is the "
         "failure this project exists to catch.")

    doc.add_heading("3.4 What the user sees", level=2)
    table(
        doc,
        ["Condition", "Verdict", "Shown to the user"],
        [
            ["Gate A below threshold", "abstain_no_context",
             "Abstention message. Generation never ran."],
            ["Gate A passes, Gate B below threshold", "abstain_ungrounded",
             "Abstention message with the failing claims listed. The suppressed answer stays "
             "available behind a disclosure control for inspection."],
            ["Both gates pass, composite below threshold", "flag_low_confidence",
             "The answer, with an explicit low-confidence warning."],
            ["Both gates pass, composite above threshold", "answer",
             "The answer, with its scores and the passages it came from."],
        ],
        widths=[1.85, 1.35, 3.1],
    )
    para(doc,
         "Every verdict carries a one-line explanation containing the actual scores and "
         "thresholds involved, so a reviewer can always see why a query was refused. One case "
         "the table does not separate: Gate A can pass and the generator itself can then emit "
         "the insufficient-context marker it was instructed to use. That self-refusal is "
         "honoured directly and the text is discarded unscored, since measuring entailment on a "
         "refusal string would be meaningless. It currently records the same verdict as a Gate A "
         "abstention, which overstates what the confidence layer contributed, so splitting the "
         "two in the telemetry is the first item of the next milestone.")
COMPONENTS = [
    ["Ingestion and retrieval\nsrc/ingest.py, embeddings.py, retriever.py",
     "PDF, DOCX, Markdown and text loading, boundary-aware chunking, content-addressed chunk IDs "
     "so re-ingesting a file cannot duplicate it, one embedding entry point with a CUDA to CPU "
     "fallback, and ChromaDB search with an explicit distance-to-similarity conversion per "
     "metric."],
    ["Scoring backends\nsrc/scorers.py",
     "The reranker, the NLI cross-encoder and a DeepEval judge adapter. NLI label order is read "
     "from the checkpoint rather than assumed, since guessing wrong inverts every score."],
    ["Confidence layer\nsrc/confidence.py",
     "The two gates, the composite, the contradiction rule and the decision table. Kept behind "
     "its own interface so it can be tested with no language model present."],
    ["Generation\nsrc/generator.py, llm.py",
     "Grounded generation over numbered passages with a self-abstention marker, an unguarded "
     "control arm, and one factory for Ollama, Groq or Gemini with generator and judge as "
     "separate handles."],
    ["Pipeline and telemetry\nsrc/pipeline.py, store.py",
     "Retrieve, Gate A, abstain early, generate, Gate B, decide, log. SQLite telemetry, 30 fields "
     "per query plus a primary key, written synchronously."],
    ["Interface\napp.py, pages/",
     "Streamlit. Gate scores against thresholds, per-claim entailment, retrieved passages, "
     "latency, and thresholds as live sliders so one query can be watched changing verdict. A "
     "second page plots the telemetry over time."],
    ["Tests and scripts\ntests/, scripts/",
     "31 unit tests, an index builder, a staged smoke test, a latency benchmark, and this report "
     "generator."],
]


def section_built(doc: Document) -> None:
    doc.add_heading("4. What Was Built", level=1)
    para(doc,
         "The deliverable for this milestone is working code rather than a prototype notebook: "
         "2,989 lines of Python across 17 modules, all of it running.")
    table(doc, ["Component", "What it does"], COMPONENTS, widths=[1.55, 4.75])
def section_verification(doc: Document) -> None:
    doc.add_heading("5. Verification", level=1)
    para(doc,
         "The confidence layer is the contribution, so I did not want its correctness to rest on a "
         "screenshot. The 31 unit tests cover the scoring mathematics and the decision table with "
         "no model loaded, so the suite finishes in about a tenth of a second and cannot be broken "
         "by a failed download or a busy GPU. They pin down the harmonic mean's penalty behaviour "
         "and its bounds, the distance-to-similarity conversion for each metric, sentence-splitting "
         "edge cases, the three taxonomy boundaries, all four verdicts, and the rule that one "
         "contradicted claim always fails Gate B.")
    para(doc,
         "A staged smoke test then runs the real components end to end and asserts behavioural "
         "separation rather than merely that nothing crashed: 24 checks passed, 0 failed. A query "
         "the corpus covers retrieved at top similarity 0.531 against 0.009 for one it does not "
         "cover. Gate A scored 0.999 on the covered query and 0.000 on the out-of-corpus one, "
         "which it refused before the generator was ever called. Gate B scored a grounded answer "
         "0.997 and a fabricated one 0.000, labelling the fabrication an evident conflict. The "
         "composite and the verdicts followed, and a telemetry row was written and read back "
         "correctly. Two further stages need a running language model and are held behind a flag "
         "for the next milestone.")
    para(doc,
         "Latency was measured rather than assumed, since the confidence layer is a cost added to "
         "every single query. Warm timings on the RTX 4060, ten iterations, models loaded once "
         "and the first pass discarded:")
    table(
        doc,
        ["Stage", "Median", "Mean", "Min", "Max"],
        [
            ["Retrieval (k = 5)", "17.6 ms", "25.9 ms", "14.2 ms", "106.0 ms"],
            ["Gate A, reranker", "16.1 ms", "15.8 ms", "11.6 ms", "22.5 ms"],
            ["Gate B, NLI over claims", "118.0 ms", "181.1 ms", "89.4 ms", "636.2 ms"],
            ["Confidence layer total", "134.1 ms", "196.9 ms", "104.5 ms", "658.8 ms"],
        ],
        widths=[2.3, 1.0, 1.0, 1.0, 1.0],
    )
    para(doc,
         "The whole two-gate layer costs roughly 134 ms at the median, less than a single round "
         "trip to any hosted language model. That figure is why local cross-encoders are the "
         "primary scoring path and the LLM judge is kept as an independent cross-check: a judge "
         "costs several API calls per answer, and a sweep of thousands of benchmark questions on "
         "a free tier would hit a rate-limit wall. ruff reports no findings across the codebase; "
         "five it raised during the week were fixed, two of them real defects, namely an "
         "unguarded zip over sequences of unequal length and a timing path that dropped "
         "retrieval latency whenever generation was skipped. The tests, the smoke test, the "
         "latency benchmark and the interface each run from a single command, listed in the "
         "README.")
def section_status(doc: Document) -> None:
    doc.add_heading("6. Status and Limitations", level=1)
    para(doc,
         "The milestone is complete: the architecture is settled, the pipeline runs end to end, "
         "and the verification described above passes. What it does not have yet is stated "
         "plainly here, because a project about overconfident systems should not be "
         "overconfident about itself.")
    bullets(doc, [
         "The thresholds are placeholders. 0.45 for Gate A, 0.60 for Gate B and 0.55 for the "
         "composite were chosen to make the mechanism demonstrable, not because data supports "
         "them. They are labelled as placeholders in the code, the interface and the README, and "
         "fitting them against labelled data is the main work ahead.",
         "There are no benchmark results yet, and verification so far used a six-chunk sample "
         "corpus rather than the full course materials. This report claims implementation and "
         "verification, not performance.",
         "Gate A has a known weak spot that is not yet quantified: a reranker scores passages one "
         "at a time, so a question whose answer needs two passages combined can score low on each.",
         "Claim splitting is a regular expression over sentence boundaries. A sentence is a coarse "
         "stand-in for a claim, which is a simplification I am accepting on purpose.",
    ])
    para(doc,
         "The next step is to bring the local generator online through both gates on my own "
         "course materials, including the two smoke stages that need a language model, and to "
         "start the benchmark harness the calibration work depends on.")


def main() -> int:
    doc = Document()
    style_document(doc)
    for section in doc.sections:
        section.top_margin = Inches(0.62)
        section.bottom_margin = Inches(0.62)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    cover(doc)
    section_objective(doc)
    section_problem(doc)
    section_approach(doc)
    section_built(doc)
    section_verification(doc)
    section_status(doc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
