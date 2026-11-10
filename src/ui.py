"""Shared presentation helpers for the Confidence-Aware RAG Streamlit Studio.

Kept apart from the pipeline code on purpose: nothing here decides anything,
it only renders what ``ConfidenceReport`` and ``PipelineResult`` already contain.
"""

from __future__ import annotations

import re
from html import escape

import streamlit as st

# Semantic status palette (WCAG AA compliant on both light and dark surfaces)
PASS = "#10b981"   # Emerald 500
FAIL = "#ef4444"   # Rose/Red 500
SKIP = "#64748b"   # Slate 500
WARN = "#f59e0b"   # Amber 500
IDLE = "#94a3b8"   # Slate 400
BRAND = "#4f46e5"  # Indigo 600

VERDICTS = {
    "answer": (
        "Verified & Answered",
        "Both Gate A (retrieval relevance) and Gate B (claim-level NLI entailment) passed with high composite confidence.",
        PASS,
    ),
    "flag_low_confidence": (
        "Answered · Low Confidence Warning",
        "Both gates cleared their individual thresholds, but the weighted harmonic composite fell below the warning line.",
        WARN,
    ),
    "abstain_no_context": (
        "Abstained · Insufficient Retrieval Context (Gate A)",
        "Retrieved passages did not meet the Gate A cross-encoder relevance threshold. Generation was skipped (0 LLM tokens spent).",
        FAIL,
    ),
    "abstain_ungrounded": (
        "Abstained · Ungrounded or Contradicted Answer (Gate B)",
        "An answer was drafted by the LLM, but Gate B NLI verification detected unsupported or contradicted claims and withheld it.",
        FAIL,
    ),
}

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root {
    --cr-bg: #f8fafc;
    --cr-surface: #ffffff;
    --cr-border: #e2e8f0;
    --cr-text: #0f172a;
    --cr-muted: #475569;
    --cr-subtle: #64748b;
    --cr-brand: #4f46e5;
    --cr-brand-soft: rgba(79, 70, 229, 0.08);
    --cr-pass: #10b981;
    --cr-fail: #ef4444;
    --cr-warn: #f59e0b;
    --cr-skip: #64748b;
}

@media (prefers-color-scheme: dark) {
    :root {
        --cr-bg: #0b0f19;
        --cr-surface: #111827;
        --cr-border: #1e293b;
        --cr-text: #f8fafc;
        --cr-muted: #94a3b8;
        --cr-subtle: #64748b;
        --cr-brand: #6366f1;
        --cr-brand-soft: rgba(99, 102, 241, 0.14);
    }
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1260px !important;
}

/* Enterprise Hero Header */
.cr-hero {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    padding: 1.55rem 1.85rem;
    margin-bottom: 1.25rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 58%, #312e81 100%);
    color: #f8fafc;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 12px 32px -10px rgba(15, 23, 42, 0.35);
}
.cr-hero-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
}
.cr-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.22);
    border: 1px solid rgba(129, 140, 248, 0.4);
    color: #c7d2fe;
}
.cr-hero-badges {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
}
.cr-hero-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.71rem;
    padding: 0.22rem 0.58rem;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: #e2e8f0;
}
.cr-hero h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
    line-height: 1.2;
    color: #ffffff !important;
}
.cr-hero p {
    color: #cbd5e1;
    margin: 0;
    font-size: 0.96rem;
    max-width: 820px;
    line-height: 1.5;
}

/* 5-Stage Pipeline Strip */
.cr-pipe-wrap {
    margin: 0.95rem 0 1.15rem 0;
}
.cr-pipe-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--cr-subtle);
    margin-bottom: 0.45rem;
}
.cr-pipe {
    display: grid;
    grid-template-columns: repeat(5, minmax(140px, 1fr));
    gap: 0.65rem;
}
@media (max-width: 900px) {
    .cr-pipe {
        grid-template-columns: repeat(2, 1fr);
    }
}
.cr-step {
    position: relative;
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: rgba(148, 163, 184, 0.04);
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 94px;
}
.cr-step-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
    margin-bottom: 0.25rem;
}
.cr-step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.12rem 0.42rem;
    border-radius: 5px;
    background: rgba(148, 163, 184, 0.16);
    color: inherit;
}
.cr-step .t {
    font-weight: 700;
    font-size: 0.92rem;
    letter-spacing: -0.01em;
}
.cr-step .d {
    font-size: 0.77rem;
    opacity: 0.74;
    line-height: 1.35;
    margin-top: 0.12rem;
}
.cr-step .s {
    display: inline-flex;
    align-items: center;
    gap: 0.32rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 600;
    margin-top: 0.5rem;
    padding: 0.18rem 0.5rem;
    border-radius: 6px;
    width: fit-content;
}

/* Verdict Banner */
.cr-verdict {
    border-radius: 14px;
    padding: 1.05rem 1.25rem;
    margin: 0.5rem 0 1.1rem 0;
    border: 1px solid rgba(148, 163, 184, 0.22);
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}
.cr-verdict-main {
    flex: 1 1 420px;
}
.cr-verdict .h {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.16rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.cr-verdict .e {
    font-size: 0.91rem;
    opacity: 0.9;
    margin-top: 0.25rem;
    line-height: 1.45;
}
.cr-verdict .m {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.79rem;
    opacity: 0.75;
    margin-top: 0.35rem;
}

/* Answer Box & Formatted Citations */
.cr-answer-box {
    border-radius: 14px;
    padding: 1.15rem 1.35rem;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: rgba(148, 163, 184, 0.03);
    margin-bottom: 1rem;
}
.cr-answer-hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.72;
}
.cr-answer-body {
    font-size: 0.98rem;
    line-height: 1.68;
    white-space: pre-wrap;
}
.cr-cite {
    display: inline-flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    font-weight: 700;
    padding: 0.05rem 0.42rem;
    margin: 0 0.12rem;
    border-radius: 5px;
    background: rgba(79, 70, 229, 0.12);
    color: #4f46e5;
    border: 1px solid rgba(79, 70, 229, 0.3);
    vertical-align: baseline;
}

/* Threshold Score Gauge Bar */
.cr-bar {
    padding: 0.85rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(148, 163, 184, 0.03);
    margin: 0.35rem 0 0.85rem 0;
}
.cr-bar .lbl {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.45rem;
}
.cr-bar .val-group {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.cr-bar .track {
    position: relative;
    height: 12px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.2);
    margin: 0.35rem 0;
}
.cr-bar .fill {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    border-radius: 999px;
    transition: width 0.35s ease;
}
.cr-bar .mark {
    position: absolute;
    top: -5px;
    bottom: -5px;
    width: 3px;
    border-radius: 2px;
    background: #0f172a;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.8);
}
@media (prefers-color-scheme: dark) {
    .cr-bar .mark {
        background: #f8fafc;
        box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.8);
    }
}
.cr-bar .cap {
    font-size: 0.78rem;
    opacity: 0.75;
    margin-top: 0.35rem;
    line-height: 1.4;
}

/* Badges & Cards */
.cr-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.14rem 0.55rem;
    border-radius: 999px;
    margin-right: 0.35rem;
    border: 1px solid currentColor;
}
.cr-card {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 14px;
    padding: 1.05rem 1.15rem;
    background: rgba(148, 163, 184, 0.03);
    height: 100%;
    min-height: 138px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}
.cr-card .t {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 0.98rem;
    margin-bottom: 0.35rem;
}
.cr-card .d {
    font-size: 0.86rem;
    opacity: 0.82;
    line-height: 1.5;
}

/* KPI Metric Card */
.cr-kpi {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 14px;
    padding: 1rem 1.15rem;
    background: rgba(148, 163, 184, 0.03);
}
.cr-kpi-lbl {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.7;
}
.cr-kpi-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0.2rem 0;
    line-height: 1.15;
}
.cr-kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    opacity: 0.8;
}

/* Equal-height sample question card */
.cr-sample {
    border: 1px solid rgba(148, 163, 184, 0.26);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    background: rgba(148, 163, 184, 0.03);
    min-height: 118px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    margin-bottom: 0.35rem;
}
.cr-sample-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
    margin-bottom: 0.35rem;
}
.cr-sample-q {
    font-weight: 700;
    font-size: 0.94rem;
    line-height: 1.35;
    margin-bottom: 0.25rem;
}
.cr-sample-why {
    font-size: 0.79rem;
    opacity: 0.76;
    line-height: 1.38;
}
</style>
"""


def inject_css() -> None:
    st.html(_CSS)


def hero(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    default_badges = badges or [
        "Gate A · ms-marco-MiniLM",
        "Gate B v2 · DeBERTa-v3 NLI",
        "Harmonic Composite (0.4 / 0.6)",
    ]
    pills_html = "".join(f"<span class='cr-hero-pill'>{escape(b)}</span>" for b in default_badges)
    st.html(
        f"<div class='cr-hero'>"
        f"<div class='cr-hero-top'>"
        f"<span class='cr-eyebrow'>● AI Reliability & Hallucination Guardrail Studio</span>"
        f"<div class='cr-hero-badges'>{pills_html}</div>"
        f"</div>"
        f"<h1>{escape(title)}</h1>"
        f"<p>{escape(subtitle)}</p>"
        f"</div>"
    )


def _step(idx: int, title: str, desc: str, state: str | None, note: str = "") -> str:
    colour = {"pass": PASS, "fail": FAIL, "skip": SKIP, "warn": WARN}.get(state or "", IDLE)
    label = {"pass": "PASSED", "fail": "FAILED", "skip": "SKIPPED", "warn": "FLAGGED"}.get(
        state or "", "READY"
    )
    icon = {"pass": "✓", "fail": "✕", "skip": "—", "warn": "!"}.get(state or "", "•")
    text = note or label
    status = (
        f"<div class='s' style='color:{colour};background:{colour}18;border:1px solid {colour}40'>"
        f"<span>{icon}</span><span>{escape(text)}</span></div>"
    )
    border = (
        f"border-color:{colour};background:linear-gradient(180deg, {colour}0d 0%, transparent 100%);"
        if state
        else ""
    )
    clean_title = title.split("·", 1)[-1].strip() if "·" in title else title
    return (
        f"<div class='cr-step' style='{border}'>"
        f"<div>"
        f"<div class='cr-step-top'><span class='t'>{escape(clean_title)}</span>"
        f"<span class='cr-step-num'>0{idx}</span></div>"
        f"<div class='d'>{escape(desc)}</div>"
        f"</div>"
        f"{status}</div>"
    )


def pipeline_strip(result=None, baseline: bool = False) -> None:
    """The five pipeline stages, styled as an enterprise telemetry stepper."""
    steps = [
        ("1 · Retrieve", "Bi-encoder top-k vector search"),
        ("2 · Gate A", "Pre-LLM cross-encoder relevance"),
        ("3 · Generate", "Context-restricted LLM synthesis"),
        ("4 · Gate B", "Claim-level DeBERTa NLI check"),
        ("5 · Decide", "Harmonic composite verdict"),
    ]
    states: list[tuple[str | None, str]] = [(None, "awaiting query")] * 5
    if result is not None:
        rep = result.confidence
        a, b = rep.gate_a, rep.gate_b
        gen = result.generation
        self_refused = bool(gen and gen.self_reported_insufficient)

        def verdict(passed: bool, score: float) -> tuple[str, str]:
            return ("pass", f"{score:.2f} · passed") if passed else ("fail", f"{score:.2f} · failed")

        states = [
            ("pass", f"{len(result.retrieval.chunks)} passages · {result.retrieval_ms:.0f}ms"),
            verdict(a.passed, a.score),
            ("skip", "skipped (0 tokens)")
            if gen is None
            else (("warn", "model self-refused") if self_refused else ("pass", f"{gen.latency_ms:.0f}ms · generated")),
            ("skip", "not run") if b is None else verdict(b.passed, b.score),
            (
                "pass" if not rep.abstained else "fail",
                "shown (baseline)"
                if baseline
                else (f"{rep.composite:.2f} · answered" if not rep.abstained else "abstained"),
            ),
        ]
        if baseline:
            states[1] = ("pass" if a.passed else "warn", f"{a.score:.2f} (unforced)")
            if b is not None:
                states[3] = ("pass" if b.passed else "warn", f"{b.score:.2f} (unforced)")

    mode_badge = "BASELINE CONTROL ARM (GATES UNFORCED)" if baseline else "GUARDED 2-GATE PIPELINE ACTIVE"
    html = (
        f"<div class='cr-pipe-wrap'>"
        f"<div class='cr-pipe-header'><span>Execution Pipeline Telemetry</span>"
        f"<span>{escape(mode_badge)}</span></div>"
        f"<div class='cr-pipe'>"
    )
    for i, ((t, d), (s, note)) in enumerate(zip(steps, states, strict=True), 1):
        html += _step(i, t, d, s, note)
    st.html(html + "</div></div>")


def verdict_banner(verdict: str, explanation: str) -> None:
    title, plain, colour = VERDICTS.get(verdict, (verdict, "", SKIP))
    icon = {"answer": "🛡️", "flag_low_confidence": "⚠️"}.get(verdict, "🛑")
    st.html(
        f"<div class='cr-verdict' style='border-left:6px solid {colour};background:{colour}12'>"
        f"<div class='cr-verdict-main'>"
        f"<div class='h' style='color:{colour}'><span>{icon}</span><span>{escape(title)}</span></div>"
        f"<div class='e'>{escape(plain)}</div>"
        f"<div class='m'>{escape(explanation)}</div>"
        f"</div></div>"
    )


def format_answer_html(text: str) -> str:
    """Format answer text with inline citation badges and clean spacing."""
    cleaned = escape(text.strip())
    cleaned = re.sub(r"\n+\s*(\[(?:\d+(?:\s*,\s*\d+)*)\])\s*$", r" \1", cleaned)
    cleaned = _CITATION_RE.sub(r"<span class='cr-cite'>Ref \1</span>", cleaned)
    return cleaned


def score_bar(label: str, value: float | None, threshold: float | None, caption: str = "") -> None:
    """A 0-1 gauge bar with threshold marker and status pill."""
    if value is None:
        st.html(
            f"<div class='cr-bar'><div class='lbl'><span>{escape(label)}</span>"
            f"<span class='val-group'>{chip('SKIPPED', SKIP)}</span></div>"
            f"<div class='track'></div><div class='cap'>{escape(caption)}</div></div>"
        )
        return
    v = max(0.0, min(1.0, float(value)))
    ok = threshold is None or value >= threshold
    colour = PASS if ok else FAIL
    status_txt = "PASS" if ok else "BELOW THRESHOLD"
    mark = (
        f"<div class='mark' style='left:calc({threshold * 100:.1f}% - 1.5px)' "
        f"title='Threshold {threshold:.2f}'></div>"
        if threshold is not None
        else ""
    )
    thresh_txt = f" / threshold {threshold:.2f}" if threshold is not None else ""
    st.html(
        f"<div class='cr-bar'>"
        f"<div class='lbl'><span>{escape(label)}</span>"
        f"<span class='val-group'><span>{value:.3f}{escape(thresh_txt)}</span>"
        f"{chip(status_txt, colour)}</span></div>"
        f"<div class='track'><div class='fill' style='width:{v * 100:.1f}%;background:{colour}'></div>{mark}</div>"
        f"<div class='cap'>{escape(caption)}</div></div>"
    )


def chip(text: str, colour: str) -> str:
    return (
        f"<span class='cr-chip' style='color:{colour};background:{colour}15;"
        f"border-color:{colour}45'>{escape(text)}</span>"
    )


def card(title: str, body: str) -> None:
    st.html(f"<div class='cr-card'><div class='t'>{escape(title)}</div><div class='d'>{escape(body)}</div></div>")


def kpi_card(label: str, value: str, subtext: str = "", colour: str = BRAND) -> None:
    st.html(
        f"<div class='cr-kpi' style='border-top:3px solid {colour}'>"
        f"<div class='cr-kpi-lbl'>{escape(label)}</div>"
        f"<div class='cr-kpi-val'>{escape(value)}</div>"
        f"<div class='cr-kpi-sub' style='color:{colour}'>{escape(subtext)}</div>"
        f"</div>"
    )


def sample_card(tag: str, question: str, why: str, badge_colour: str = BRAND) -> None:
    st.html(
        f"<div class='cr-sample'>"
        f"<div>"
        f"<div class='cr-sample-top'>{chip(tag.upper(), badge_colour)}</div>"
        f"<div class='cr-sample-q'>{escape(question)}</div>"
        f"</div>"
        f"<div class='cr-sample-why'>{escape(why)}</div>"
        f"</div>"
    )
