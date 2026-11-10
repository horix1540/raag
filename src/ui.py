"""Shared presentation helpers for the Streamlit pages.

Kept apart from the pipeline code on purpose: nothing here decides anything,
it only draws what ``ConfidenceReport`` already contains. The pages import it
so the main app, the guide and the dashboard share one visual language.
"""

from __future__ import annotations

from html import escape

import streamlit as st

# One colour per outcome, used everywhere a pass/fail/skip is drawn.
PASS, FAIL, SKIP, WARN, IDLE = "#2ea36b", "#e5534b", "#7d8590", "#d29922", "#3d4450"

VERDICTS = {
    "answer": ("Answered", "Both gates passed and the combined confidence is high.", PASS),
    "flag_low_confidence": (
        "Answered, low confidence",
        "Both gates passed, but only just. Read the answer with care.",
        WARN,
    ),
    "abstain_no_context": (
        "Abstained: not enough relevant context",
        "The retrieved passages do not support an answer, so nothing was generated.",
        FAIL,
    ),
    "abstain_ungrounded": (
        "Abstained: answer not supported by the sources",
        "An answer was generated, but the passages do not back it up, so it was withheld.",
        FAIL,
    ),
}

_CSS = """
<style>
.block-container {padding-top: 3.6rem; max-width: 1180px;}
.cr-hero h1 {font-size: 2.0rem; margin: 0 0 .25rem 0; line-height: 1.2;}
.cr-hero p {opacity: .78; margin: 0; font-size: 1rem;}
.cr-pipe {display: flex; align-items: stretch; gap: .35rem; margin: .9rem 0 .4rem 0; flex-wrap: wrap;}
.cr-step {flex: 1 1 120px; border-radius: 10px; padding: .55rem .7rem; border: 1px solid rgba(127,127,127,.25);
          background: rgba(127,127,127,.06); min-width: 110px;}
.cr-step .t {font-weight: 600; font-size: .92rem;}
.cr-step .d {font-size: .78rem; opacity: .75; margin-top: .1rem;}
.cr-step .s {font-size: .78rem; font-weight: 600; margin-top: .3rem;}
.cr-arrow {align-self: center; opacity: .45; font-size: 1.1rem;}
.cr-verdict {border-radius: 12px; padding: .9rem 1.1rem; margin: .4rem 0 1rem 0;
             border: 1px solid rgba(127,127,127,.2);}
.cr-verdict .h {font-size: 1.15rem; font-weight: 700;}
.cr-verdict .e {font-size: .92rem; opacity: .85; margin-top: .2rem;}
.cr-bar {margin: .15rem 0 .9rem 0;}
.cr-bar .lbl {display: flex; justify-content: space-between; font-size: .88rem; margin-bottom: .25rem;}
.cr-bar .track {position: relative; height: 10px; border-radius: 6px; background: rgba(127,127,127,.18);}
.cr-bar .fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px;}
.cr-bar .mark {position: absolute; top: -4px; bottom: -4px; width: 2px; background: currentColor; opacity: .8;}
.cr-bar .cap {font-size: .76rem; opacity: .7; margin-top: .2rem;}
.cr-chip {display: inline-block; font-size: .72rem; font-weight: 600; padding: .05rem .45rem; border-radius: 999px;
          margin-right: .3rem; border: 1px solid currentColor;}
.cr-card {border: 1px solid rgba(127,127,127,.22); border-radius: 12px; padding: .8rem 1rem; height: 100%;}
.cr-card .t {font-weight: 650; margin-bottom: .2rem;}
.cr-card .d {font-size: .88rem; opacity: .8;}
</style>
"""


def inject_css() -> None:
    st.html(_CSS)


def hero(title: str, subtitle: str) -> None:
    st.html(f"<div class='cr-hero'><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>")


def _step(title: str, desc: str, state: str | None, note: str = "") -> str:
    colour = {"pass": PASS, "fail": FAIL, "skip": SKIP, "warn": WARN}.get(state or "", IDLE)
    label = {"pass": "passed", "fail": "failed", "skip": "skipped", "warn": "flagged"}.get(state or "", "")
    status = f"<div class='s' style='color:{colour}'>{escape(note or label)}</div>" if state else ""
    border = f"border-color:{colour};" if state else ""
    return (f"<div class='cr-step' style='{border}'><div class='t'>{escape(title)}</div>"
            f"<div class='d'>{escape(desc)}</div>{status}</div>")


def pipeline_strip(result=None, baseline: bool = False) -> None:
    """The five stages, coloured by what happened to this query (grey before any query)."""
    steps = [
        ("1 · Retrieve", "find the closest passages"),
        ("2 · Gate A", "are they relevant?"),
        ("3 · Generate", "LLM writes an answer"),
        ("4 · Gate B", "is it backed by the passages?"),
        ("5 · Decide", "answer or abstain"),
    ]
    states: list[tuple[str | None, str]] = [(None, "")] * 5
    if result is not None:
        rep = result.confidence
        a, b = rep.gate_a, rep.gate_b
        gen = result.generation
        self_refused = bool(gen and gen.self_reported_insufficient)
        def verdict(passed: bool, score: float) -> tuple[str, str]:
            return ("pass", f"{score:.2f} · passed") if passed else ("fail", f"{score:.2f} · failed")

        states = [
            ("pass", f"{len(result.retrieval.chunks)} passages"),
            verdict(a.passed, a.score),
            ("skip", "not run") if gen is None else
            (("warn", "model declined") if self_refused else ("pass", "done")),
            ("skip", "not run") if b is None else verdict(b.passed, b.score),
            ("pass" if not rep.abstained else "fail",
             "shown (baseline)" if baseline else ("answered" if not rep.abstained else "abstained")),
        ]
        if baseline:
            # Gates are measured but never enforced in the control arm.
            states[1] = ("pass" if a.passed else "warn", f"{a.score:.2f} (not enforced)")
            if b is not None:
                states[3] = ("pass" if b.passed else "warn", f"{b.score:.2f} (not enforced)")
    html = "<div class='cr-pipe'>"
    for i, ((t, d), (s, note)) in enumerate(zip(steps, states, strict=True)):
        html += _step(t, d, s, note)
        if i < len(steps) - 1:
            html += "<div class='cr-arrow'>→</div>"
    st.html(html + "</div>")


def verdict_banner(verdict: str, explanation: str) -> None:
    title, plain, colour = VERDICTS.get(verdict, (verdict, "", SKIP))
    st.html(
        f"<div class='cr-verdict' style='border-left:6px solid {colour};background:{colour}14'>"
        f"<div class='h' style='color:{colour}'>{escape(title)}</div>"
        f"<div class='e'>{escape(plain)}</div>"
        f"<div class='e' style='opacity:.65;font-size:.82rem'>{escape(explanation)}</div></div>"
    )


def score_bar(label: str, value: float | None, threshold: float | None, caption: str = "") -> None:
    """A 0-1 bar with the threshold drawn as a tick, so pass/fail is visible at a glance."""
    if value is None:
        st.html(f"<div class='cr-bar'><div class='lbl'><span>{escape(label)}</span><span>not run</span></div>"
                f"<div class='track'></div><div class='cap'>{escape(caption)}</div></div>")
        return
    v = max(0.0, min(1.0, float(value)))
    ok = threshold is None or value >= threshold
    colour = PASS if ok else FAIL
    mark = (f"<div class='mark' style='left:calc({threshold * 100:.1f}% - 1px)'></div>"
            if threshold is not None else "")
    right = f"{value:.3f}" + (f" · threshold {threshold:.2f}" if threshold is not None else "")
    st.html(
        f"<div class='cr-bar'><div class='lbl'><span>{escape(label)}</span><span>{right}</span></div>"
        f"<div class='track'><div class='fill' style='width:{v * 100:.1f}%;background:{colour}'></div>{mark}</div>"
        f"<div class='cap'>{escape(caption)}</div></div>"
    )


def chip(text: str, colour: str) -> str:
    return f"<span class='cr-chip' style='color:{colour}'>{escape(text)}</span>"


def card(title: str, body: str) -> None:
    st.html(f"<div class='cr-card'><div class='t'>{escape(title)}</div><div class='d'>{escape(body)}</div></div>")
