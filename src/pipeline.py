"""End-to-end pipeline: retrieve -> Gate A -> generate -> Gate B -> decide -> log.

This is the module the report's architecture diagram describes and the only one
the Streamlit app talks to. Keeping the control flow in one readable function is
deliberate: the ordering of the gates *is* the contribution, so it should be
legible at a glance rather than buried in a chain of callbacks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.confidence import ConfidenceReport, get_evaluator
from src.config import settings
from src.generator import Generation, generate, generate_unguarded
from src.retriever import RetrievalResult, retrieve
from src.store import log_query

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    question: str
    answer: str
    displayed_answer: str
    retrieval: RetrievalResult
    generation: Generation | None
    confidence: ConfidenceReport
    retrieval_ms: float
    total_ms: float
    row_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return self.confidence.verdict

    @property
    def sources(self) -> list[str]:
        seen, out = set(), []
        for c in self.retrieval.chunks:
            if c.source not in seen:
                seen.add(c.source)
                out.append(c.source)
        return out


def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    collection: str | None = None,
    store=None,
    persist: bool = True,
    run_tag: str | None = None,
    item_id: str | None = None,
    baseline: bool = False,
    prompt_only: bool = False,
) -> PipelineResult:
    """Run the confidence-aware pipeline for one question.

    ``baseline=True`` skips both gates and always answers. That is the control
    arm for the evaluation: it is what a conventional RAG system does, and the
    comparison against it is the entire point of the project.

    ``item_id`` is the benchmark question id when this call comes from the
    harness, so a telemetry row can be joined back to its dataset label.
    """
    t_start = time.perf_counter()
    cfg = settings.confidence
    evaluator = get_evaluator()

    arm_name = "baseline" if baseline else ("prompt_only" if prompt_only else "guarded")
    meta: dict[str, Any] = {"arm": arm_name}
    if item_id is not None:
        meta["item_id"] = item_id

    t0 = time.perf_counter()
    retrieval = retrieve(question, top_k=top_k, store=store, collection=collection)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if baseline:
        gen = generate_unguarded(question, retrieval.contexts)
        gate_a = evaluator.gate_a(retrieval)
        gate_b = evaluator.gate_b(question, gen.answer, retrieval.contexts)
        report = ConfidenceReport(
            gate_a=gate_a,
            gate_b=gate_b,
            composite=evaluator.combine(gate_a, gate_b),
            verdict="answer",
            explanation="Baseline arm: gates measured but not enforced.",
        )
        return _finalise(
            question, gen.answer, gen.answer, retrieval, gen, report,
            retrieval_ms, t_start, persist, run_tag, meta,
        )

    if prompt_only:
        gen = generate(question, retrieval.contexts)
        gate_a = evaluator.gate_a(retrieval)
        if gen.self_reported_insufficient:
            report = evaluator.decide(gate_a, None)
            report.verdict = "abstain_no_context"
            report.explanation = "Prompt-only arm: generator self-refused (INSUFFICIENT_CONTEXT)."
            return _finalise(
                question, gen.answer, cfg.abstain_message, retrieval, gen, report,
                retrieval_ms, t_start, persist, run_tag, {**meta, "self_abstain": True},
            )
        gate_b = evaluator.gate_b(question, gen.answer, retrieval.contexts)
        report = ConfidenceReport(
            gate_a=gate_a,
            gate_b=gate_b,
            composite=evaluator.combine(gate_a, gate_b),
            verdict="answer",
            explanation="Prompt-only arm: abstention prompt active; gates measured but not enforced.",
        )
        return _finalise(
            question, gen.answer, gen.answer, retrieval, gen, report,
            retrieval_ms, t_start, persist, run_tag, meta,
        )

    # --- Gate A: is there anything worth answering from? ---------------------
    gate_a = evaluator.gate_a(retrieval)
    if not gate_a.passed:
        report = evaluator.decide(gate_a, None)
        return _finalise(
            question, "", cfg.abstain_message, retrieval, None, report,
            retrieval_ms, t_start, persist, run_tag, meta,
        )

    # --- Generation ----------------------------------------------------------
    gen = generate(question, retrieval.contexts)

    # The generator's own refusal is honoured directly. Scoring the string
    # "INSUFFICIENT_CONTEXT" for faithfulness would be meaningless.
    if gen.self_reported_insufficient:
        report = evaluator.decide(gate_a, None)
        report.verdict = "abstain_no_context"
        report.explanation = (
            "Gate A passed but the generator itself reported insufficient context."
        )
        return _finalise(
            question, gen.answer, cfg.abstain_message, retrieval, gen, report,
            retrieval_ms, t_start, persist, run_tag, {**meta, "self_abstain": True},
        )

    # --- Gate B: is the answer actually grounded? ----------------------------
    gate_b = evaluator.gate_b(question, gen.answer, retrieval.contexts)
    report = evaluator.decide(gate_a, gate_b)

    displayed = cfg.abstain_message if report.abstained else gen.answer
    return _finalise(
        question, gen.answer, displayed, retrieval, gen, report,
        retrieval_ms, t_start, persist, run_tag, meta,
    )


def _finalise(
    question: str,
    raw_answer: str,
    displayed: str,
    retrieval: RetrievalResult,
    gen: Generation | None,
    report: ConfidenceReport,
    retrieval_ms: float,
    t_start: float,
    persist: bool,
    run_tag: str | None,
    meta: dict,
) -> PipelineResult:
    total_ms = (time.perf_counter() - t_start) * 1000
    result = PipelineResult(
        question=question,
        answer=raw_answer,
        displayed_answer=displayed,
        retrieval=retrieval,
        generation=gen,
        confidence=report,
        retrieval_ms=retrieval_ms,
        total_ms=total_ms,
        meta=meta,
    )

    if persist:
        try:
            result.row_id = log_query(_to_row(result, run_tag))
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a query
            log.error("Telemetry write failed: %s", exc)

    return result


def _to_row(result: PipelineResult, run_tag: str | None) -> dict:
    row = result.confidence.to_row()
    gb = result.confidence.gate_b
    cfg = settings.confidence
    row.update(
        {
            "question": result.question,
            "answer": result.displayed_answer,
            "retrieval_ms": round(result.retrieval_ms, 1),
            "generation_ms": round(result.generation.latency_ms, 1) if result.generation else None,
            "total_ms": round(result.total_ms, 1),
            "provider": settings.llm.provider,
            "generator_model": settings.llm.generator_model,
            "scorer": settings.confidence.scorer,
            "sources": result.sources,
            "claims_json": [asdict(c) for c in gb.claims] if gb else [],
            "run_tag": run_tag,
            # Which arm produced this row. Without it a guarded and a baseline
            # run over the same questions are indistinguishable in the table,
            # which would make the comparison the project is built on impossible
            # to recompute later.
            "arm": result.meta.get("arm"),
            "self_abstain": int(bool(result.meta.get("self_abstain"))),
            "item_id": result.meta.get("item_id"),
            # The thresholds in force when this row was written. Calibration
            # changes them, and rows scored under different thresholds must not
            # be silently pooled.
            "thresholds_json": json.dumps(
                {
                    "gate_a": cfg.gate_a_threshold,
                    "gate_b": cfg.gate_b_threshold,
                    "composite": cfg.composite_threshold,
                    "weights": list(cfg.weights),
                    "contradiction_cap": cfg.contradiction_cap,
                }
            ),
        }
    )
    return row


