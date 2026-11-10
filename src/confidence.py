"""The confidence layer: two gates, one composite score, one decision.

Why two gates rather than one score at the end?

Gate A runs *before* generation and asks whether the retrieved passages could
possibly support an answer. If they cannot, the pipeline abstains without
calling the generator at all. This is the cheap path, it is what the project
description means by "evaluates retrieval reliability before generating a
response", and it removes the possibility of a fluent wrong answer existing in
the first place.

Gate B runs *after* generation and asks whether the answer that was produced is
actually entailed by those passages. Retrieval can be perfect and the model can
still invent a detail, so Gate A alone is insufficient.

A single post-hoc score cannot do both jobs: faithfulness is undefined until an
answer exists, and by then the tokens are already spent.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from src.config import settings
from src.retriever import RetrievalResult
from src.scorers import ClaimScore, get_llm_judge, get_nli_scorer, get_rerank_scorer

log = logging.getLogger(__name__)

Verdict = Literal["answer", "flag_low_confidence", "abstain_no_context", "abstain_ungrounded"]


def weighted_harmonic_mean(
    values: list[float], weights: list[float] | tuple[float, ...], *, eps: float = 1e-6
) -> float:
    """Weighted harmonic mean, which punishes any single near-zero component.

    An arithmetic mean of relevancy 0.95 and faithfulness 0.05 is 0.50, which
    would read as "borderline" for an answer that is almost entirely
    unsupported. The harmonic mean returns roughly 0.09 for the same pair. That
    asymmetry is the whole point: both conditions must hold, in the same way
    the F1 score refuses to reward precision without recall.
    """
    if not values:
        return 0.0
    w = list(weights)[: len(values)] or [1.0] * len(values)
    if len(w) < len(values):
        w += [1.0] * (len(values) - len(w))
    total_w = sum(w)
    if total_w <= 0:
        return 0.0
    denom = sum(wi / max(vi, eps) for wi, vi in zip(w, values, strict=True))
    return float(total_w / denom) if denom > 0 else 0.0


@dataclass
class GateAResult:
    """Pre-generation retrieval sufficiency."""

    score: float
    passed: bool
    rerank_relevancy: float
    top_similarity: float
    mean_similarity: float
    similarity_margin: float
    per_chunk_relevance: list[float] = field(default_factory=list)
    llm_relevancy: float | None = None
    llm_reason: str = ""
    latency_ms: float = 0.0


@dataclass
class GateBResult:
    """Post-generation groundedness."""

    score: float
    passed: bool
    nli_faithfulness: float
    claims: list[ClaimScore] = field(default_factory=list)
    llm_faithfulness: float | None = None
    llm_reason: str = ""
    latency_ms: float = 0.0

    @property
    def conflicts(self) -> list[ClaimScore]:
        return [c for c in self.claims if c.taxonomy_label == "evident_conflict"]

    @property
    def baseless(self) -> list[ClaimScore]:
        return [c for c in self.claims if c.taxonomy_label == "baseless_information"]

    @property
    def taxonomy_counts(self) -> dict[str, int]:
        counts = {"supported": 0, "evident_conflict": 0, "baseless_information": 0}
        for c in self.claims:
            counts[c.taxonomy_label] += 1
        return counts


@dataclass
class ConfidenceReport:
    """Everything the UI, the logger and the benchmark harness need."""

    gate_a: GateAResult
    gate_b: GateBResult | None
    composite: float
    verdict: Verdict
    explanation: str

    @property
    def abstained(self) -> bool:
        return self.verdict.startswith("abstain")

    @property
    def generation_skipped(self) -> bool:
        return self.verdict == "abstain_no_context"

    def to_row(self) -> dict:
        """Flatten to a single record for SQLite / pandas.

        Every value is a scalar or a JSON string: the telemetry table is read
        straight into a DataFrame, and a nested list in a cell would break both
        the SQLite insert and the dashboard's aggregations.
        """
        b = self.gate_b
        return {
            "gate_a_score": round(self.gate_a.score, 4),
            "gate_a_passed": self.gate_a.passed,
            "rerank_relevancy": round(self.gate_a.rerank_relevancy, 4),
            "top_similarity": round(self.gate_a.top_similarity, 4),
            "mean_similarity": round(self.gate_a.mean_similarity, 4),
            "similarity_margin": round(self.gate_a.similarity_margin, 4),
            # Per-passage reranker scores, kept because calibration needs the
            # shape of the distribution and not just its top-2 mean.
            "per_chunk_relevance": json.dumps(
                [round(v, 4) for v in self.gate_a.per_chunk_relevance]
            ),
            "llm_relevancy": self.gate_a.llm_relevancy,
            "gate_b_score": round(b.score, 4) if b else None,
            "gate_b_passed": b.passed if b else None,
            "nli_faithfulness": round(b.nli_faithfulness, 4) if b else None,
            "llm_faithfulness": b.llm_faithfulness if b else None,
            "n_claims": len(b.claims) if b else 0,
            "n_conflicts": len(b.conflicts) if b else 0,
            "n_baseless": len(b.baseless) if b else 0,
            "composite": round(self.composite, 4),
            "verdict": self.verdict,
            "gate_a_ms": round(self.gate_a.latency_ms, 1),
            "gate_b_ms": round(b.latency_ms, 1) if b else None,
        }


class ConfidenceEvaluator:
    """Runs the two gates and combines them into one decision.

    On the choice of Gate A's primary signal: ``all-MiniLM-L6-v2`` cosine
    similarity is *not* a calibrated probability - genuinely relevant passages
    typically land between 0.4 and 0.7, so thresholding it directly bakes in an
    arbitrary scale. The ms-marco reranker was trained on relevance judgements
    and its sigmoid output behaves far more like a probability, so it drives the
    gate. The embedding statistics are still recorded on every query, because
    the calibration step fits a model over all four features against labelled
    answerable/unanswerable data instead of guessing weights here.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or settings.confidence

    def _use_nli(self) -> bool:
        return self.cfg.scorer in {"nli", "both"}

    def _use_llm(self) -> bool:
        return self.cfg.scorer in {"llm", "both"}

    def gate_a(self, retrieval: RetrievalResult) -> GateAResult:
        t0 = time.perf_counter()
        contexts = retrieval.contexts

        if not contexts:
            return GateAResult(
                score=0.0,
                passed=False,
                rerank_relevancy=0.0,
                top_similarity=0.0,
                mean_similarity=0.0,
                similarity_margin=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        rerank, per_chunk = 0.0, []
        if self._use_nli():
            rerank, per_chunk = get_rerank_scorer().contextual_relevancy(
                retrieval.query, contexts
            )

        llm_rel, llm_reason = None, ""
        if self._use_llm():
            try:
                llm_rel, llm_reason = get_llm_judge().contextual_relevancy(
                    retrieval.query, contexts
                )
            except Exception as exc:  # noqa: BLE001 - judge outage must not block the gate
                log.error("LLM contextual relevancy failed: %s", exc)
                llm_reason = f"judge error: {exc}"

        score = rerank if self._use_nli() else float(llm_rel or 0.0)
        return GateAResult(
            score=score,
            passed=score >= self.cfg.gate_a_threshold,
            rerank_relevancy=rerank,
            top_similarity=retrieval.top_similarity,
            mean_similarity=retrieval.mean_similarity,
            similarity_margin=retrieval.similarity_margin,
            per_chunk_relevance=per_chunk,
            llm_relevancy=llm_rel,
            llm_reason=llm_reason,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


    def gate_b(self, query: str, answer: str, contexts: list[str]) -> GateBResult:
        t0 = time.perf_counter()

        nli_score, claims = 0.0, []
        if self._use_nli():
            nli_score, claims = get_nli_scorer().faithfulness(answer, contexts, question=query)

        llm_faith, llm_reason = None, ""
        if self._use_llm():
            try:
                llm_faith, llm_reason = get_llm_judge().faithfulness(query, answer, contexts)
            except Exception as exc:  # noqa: BLE001
                log.error("LLM faithfulness failed: %s", exc)
                llm_reason = f"judge error: {exc}"

        score = nli_score if self._use_nli() else float(llm_faith or 0.0)

        # Any outright contradiction caps the gate regardless of the mean. One
        # sentence that conflicts with the source is a hard failure, and
        # averaging lets nine correct sentences hide it. The cap is an absolute
        # constant, not "threshold minus epsilon": tying the penalty to the
        # threshold would let calibration quietly weaken or invert it.
        if claims and any(c.is_contradicted for c in claims):
            score = min(score, self.cfg.contradiction_cap)

        return GateBResult(
            score=score,
            passed=score >= self.cfg.gate_b_threshold,
            nli_faithfulness=nli_score,
            claims=claims,
            llm_faithfulness=llm_faith,
            llm_reason=llm_reason,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def combine(self, gate_a: GateAResult, gate_b: GateBResult | None) -> float:
        if gate_b is None:
            return gate_a.score
        return weighted_harmonic_mean([gate_a.score, gate_b.score], self.cfg.weights)


    def decide(self, gate_a: GateAResult, gate_b: GateBResult | None) -> ConfidenceReport:
        composite = self.combine(gate_a, gate_b)

        if not gate_a.passed:
            verdict: Verdict = "abstain_no_context"
            why = (
                f"Retrieval sufficiency {gate_a.score:.2f} is below the Gate A threshold "
                f"{self.cfg.gate_a_threshold:.2f}; generation was skipped."
            )
        elif gate_b is not None and not gate_b.passed:
            conflicts = len(gate_b.conflicts)
            baseless = len(gate_b.baseless)
            verdict = "abstain_ungrounded"
            why = (
                f"Groundedness {gate_b.score:.2f} is below the Gate B threshold "
                f"{self.cfg.gate_b_threshold:.2f} "
                f"({conflicts} contradicted, {baseless} unsupported of {len(gate_b.claims)} claims)."
            )
        elif composite < self.cfg.composite_threshold:
            verdict = "flag_low_confidence"
            why = (
                f"Both gates passed individually but the composite score {composite:.2f} "
                f"is below {self.cfg.composite_threshold:.2f}; answer shown with a warning."
            )
        else:
            verdict = "answer"
            why = f"Composite confidence {composite:.2f}; both gates passed."

        return ConfidenceReport(
            gate_a=gate_a,
            gate_b=gate_b,
            composite=composite,
            verdict=verdict,
            explanation=why,
        )


def get_evaluator() -> ConfidenceEvaluator:
    return ConfidenceEvaluator()





