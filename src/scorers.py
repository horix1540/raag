"""Scoring backends for the confidence layer.

Three independent ways to score the same thing, so the report can compare them
rather than assert that one is correct:

* ``RerankScorer``  - Gate A. Query-passage relevance via an ms-marco
  cross-encoder. Answers "does the corpus cover this question at all", which is
  a ranking problem and is decided *before* any tokens are generated.
* ``NLIScorer``     - Gate B. Claim-level entailment via a DeBERTa NLI
  cross-encoder. Answers "is each sentence of the answer actually supported by
  a retrieved passage", and distinguishes contradiction from mere absence.
* ``LLMJudgeScorer`` - reference standard. DeepEval's Faithfulness and
  Contextual Relevancy metrics. Accurate but costs LLM calls per answer, which
  is why the local scorers exist at all.

The local scorers run on the GPU in milliseconds, which is what makes a
several-thousand-question benchmark sweep feasible on a laptop.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from dataclasses import dataclass

from src.config import settings

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


_LEAD_IN_PREFIXES = (
    "here are",
    "here is",
    "the following",
    "the five stages",
    "the stages",
    "the steps",
    "based on",
    "according to",
    "as stated in",
    "in summary",
    "in short",
)
_BULLET_PREFIX = re.compile(r"^(\d+[\.\)]|\*|\-|\+)\s*")


def split_claims(answer: str, *, min_chars: int = 12, min_words: int | None = None) -> list[str]:
    """Split an answer into atomic factual claims, cleaning formatting artifacts.

    Strips markdown bullet/numbered prefixes, inline citation tags, and drops
    non-factual lead-in scaffolding sentences (e.g. 'The five stages of a standard RAG pipeline are:').
    Never scores a fragment shorter than min_claim_words (default 4) as a claim.
    """
    if not answer or not answer.strip():
        return []

    min_words = min_words if min_words is not None else settings.confidence.min_claim_words

    # If newlines are present, split line by line first to preserve list item structure
    lines = [line.strip() for line in answer.split("\n") if line.strip()]

    raw_candidates: list[str] = []
    for line in lines:
        line_clean = line.strip()
        lower_line = line_clean.lower()
        if line_clean.endswith(":") and any(lower_line.startswith(p) for p in _LEAD_IN_PREFIXES):
            continue
        if line_clean.endswith(":") and len(line_clean.split()) <= 12 and not any(ch.isdigit() for ch in line_clean[:3]):
            continue

        cleaned = _BULLET_PREFIX.sub("", line_clean).strip()
        cleaned = re.sub(r"^\*\*(.*?)\*\*", r"\1", cleaned).strip()
        cleaned = _CITATION.sub("", cleaned).strip()
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            continue

        sub_sents = [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]
        raw_candidates.extend(sub_sents)

    if not raw_candidates:
        cleaned_fallback = " ".join(_CITATION.sub("", answer).split()).strip()
        return [cleaned_fallback] if cleaned_fallback else []

    if len(raw_candidates) == 1:
        return raw_candidates

    claims: list[str] = []
    for c in raw_candidates:
        words = c.split()
        if len(words) < min_words:
            continue
        if c.endswith(":") and len(words) <= 12:
            continue
        claims.append(c)

    return claims or raw_candidates


_CITATION = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")

# A claim this short is almost always a bare answer ("New Orleans",
# "August 15, 1971") rather than a statement. Eight words is a heuristic, not
# a fit: it covers noun-phrase answers and keeps ordinary sentences untouched.
SHORT_CLAIM_WORDS = 8


def build_hypothesis(claim: str, question: str | None, mode: str) -> str:
    """The text NLI actually tests against each passage.

    ``claim`` (Gate B v1) tests the answer sentence alone. That fails on short
    answers: NLI decides whether a premise entails a *statement*, and
    "New Orleans [1]" is not one, so it scores ~0 even when the passage names
    New Orleans as the answer. On the M3 SQuAD run this suppressed 35 correct
    answers out of 36 Gate B rejections on answerable questions.

    ``qa`` prefixes the question to every claim, so the hypothesis reads as
    "Where was the game played? New Orleans" - something a passage can entail.
    It also puts the question's premise into the hypothesis, which lets NLI
    notice an answer that is in the passage but answers a different question.
    ``qa_short`` does that only for short claims and leaves full sentences as
    they were, the least invasive version.

    The v2 modes also drop citation markers: "[1]" is formatting the generator
    was asked for, not something the passage should have to entail. ``claim``
    is left byte-for-byte as v1 so the saved Milestone 2-4 scores reproduce.
    """
    if mode == "claim":
        return claim
    text = " ".join(_CITATION.sub(" ", claim).split())
    if mode == "claim_clean" or not question:
        return text if mode == "claim_clean" else claim
    if mode == "qa_short" and len(text.split()) > SHORT_CLAIM_WORDS:
        return text
    return f"{question.strip()} {text}"


_PASSAGE_SENTENCE = re.compile(r"(?<=[.!?])\s+")
WINDOW_SENTENCES = 3


def premise_windows(contexts: list[str], size: int = WINDOW_SENTENCES) -> list[tuple[str, int]]:
    """Overlapping sentence windows of each passage, tagged with their passage.

    A whole SQuAD paragraph is 100-350 words. nli-deberta-v3-base, trained on
    one- or two-sentence premises, goes neutral on most claims at that length
    even when one sentence states them outright. Windows of a few consecutive
    sentences (stride 1) keep the premise short while leaving room for "The
    park ..." to find "Woodward Park" a sentence earlier - the SummaC
    approach. Passages no longer than one window are used whole.

    Windows alone fail the other way: a claim that summarises a list ("five
    stages: ingestion, chunking, ...") needs the whole passage, and a window
    holding two of the five items reads as a contradiction. Gate B v2
    therefore uses premise mode "both" - every passage whole *and* its
    windows - and keeps the best-supporting premise, so short facts match a
    window and summaries match the passage.
    """
    out: list[tuple[str, int]] = []
    for i, ctx in enumerate(contexts):
        sents = [s for s in _PASSAGE_SENTENCE.split(" ".join(ctx.split())) if s]
        if len(sents) <= size:
            out.append((ctx, i))
            continue
        out.extend((" ".join(sents[j : j + size]), i) for j in range(len(sents) - size + 1))
    return out


def _device() -> str:
    requested = settings.confidence.nli_device
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return requested
        log.warning(
            "CUDA requested (%s) but torch.cuda.is_available() is False; falling back to CPU.",
            requested,
        )
    except ImportError:  # pragma: no cover
        log.warning("torch not importable; falling back to CPU.")
    return "cpu"


@dataclass
class ClaimScore:
    """NLI verdict for one claim against the best-matching retrieved passage."""

    claim: str
    entailment: float
    neutral: float
    contradiction: float
    best_context_index: int
    is_conflict: bool | None = None

    @property
    def is_supported(self) -> bool:
        return self.entailment >= settings.confidence.support_threshold

    @property
    def is_contradicted(self) -> bool:
        if self.is_conflict is not None:
            return self.is_conflict
        return (
            not self.is_supported
            and self.contradiction >= settings.confidence.contradiction_threshold
            and (self.contradiction - self.entailment) >= settings.confidence.contradiction_margin
        )

    @property
    def taxonomy_label(self) -> str:
        """RAGTruth-style label for this claim.

        Contradiction is checked first because a claim that conflicts with the
        source is a strictly worse failure than one the source is silent about.
        """
        if self.is_contradicted:
            return "evident_conflict"
        if not self.is_supported:
            return "baseless_information"
        return "supported"


@functools.lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    dev = _device()
    log.info("Loading cross-encoder %s on %s", model_name, dev)
    return CrossEncoder(model_name, device=dev, max_length=512)


class NLIScorer:
    """Gate B: claim-level groundedness via natural language inference."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.confidence.nli_model
        self._model = None
        self._label_order: list[str] | None = None
        self._pair_cache: dict[tuple[str, str], list[float]] = {}

    def _ensure(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
            self._label_order = self._resolve_label_order(self._model)
        return self._model

    @staticmethod
    def _resolve_label_order(model) -> list[str]:
        """Read label order from the checkpoint instead of assuming it.

        NLI checkpoints disagree on ordering - some are
        (contradiction, entailment, neutral), others
        (entailment, neutral, contradiction). Hardcoding one silently inverts
        every score on the other, so the mapping is read from the config and
        only falls back to a documented default if that is unavailable.
        """
        id2label = getattr(getattr(model, "config", None), "id2label", None)
        if isinstance(id2label, dict) and id2label:
            try:
                return [str(id2label[i]).lower() for i in sorted(id2label)]
            except Exception:  # noqa: BLE001
                pass
        log.warning("Could not read id2label; assuming DeBERTa NLI default order.")
        return ["contradiction", "entailment", "neutral"]

    def _softmax_rows(self, logits):
        import numpy as np

        arr = np.asarray(logits, dtype=float)
        arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
        arr = arr - arr.max(axis=1, keepdims=True)
        exp = np.exp(arr)
        return exp / exp.sum(axis=1, keepdims=True)


    def score_claims(
        self,
        claims: list[str],
        contexts: list[str],
        *,
        question: str | None = None,
        mode: str | None = None,
        premise: str | None = None,
    ) -> list[ClaimScore]:
        """Score every claim against every context, keeping the best support."""
        if not claims or not contexts:
            return []

        import os
        import time

        t_start = time.perf_counter()
        t_load_0 = time.perf_counter()
        model = self._ensure()
        model_load_ms = (time.perf_counter() - t_load_0) * 1000

        mode = mode or settings.confidence.nli_hypothesis
        premise = premise or settings.confidence.nli_premise
        # The UI shows the original claim; only the NLI input is rewritten.
        hyps = [build_hypothesis(c, question, mode) for c in claims]

        premises: list[str] = []
        origin: list[int] = []
        premise_types: list[str] = []
        seen_premises: set[str] = set()

        def _add_premise(text: str, orig: int, ptype: str):
            norm = " ".join(text.split())
            if norm and norm not in seen_premises:
                seen_premises.add(norm)
                premises.append(text)
                origin.append(orig)
                premise_types.append(ptype)

        for idx, ctx in enumerate(contexts):
            _add_premise(ctx, idx, "passage")

        if premise in ("window", "both", "multi_hop"):
            windows = premise_windows(contexts)
            if premise == "window":
                premises, origin, premise_types = [], [], []
                seen_premises.clear()
            for w, o in windows:
                _add_premise(w, o, "window")

            if premise == "multi_hop" and len(contexts) >= 2:
                k = min(len(contexts), settings.confidence.multi_hop_top_k)
                for i in range(k):
                    for j in range(i + 1, k):
                        _add_premise(f"{contexts[i].strip()} {contexts[j].strip()}", i, "pair")

        t_pair_0 = time.perf_counter()
        pairs = [(p, hyp) for hyp in hyps for p in premises]
        pair_construction_ms = (time.perf_counter() - t_pair_0) * 1000

        t_fwd_0 = time.perf_counter()
        unseen_pairs = [pair for pair in pairs if pair not in self._pair_cache]
        if unseen_pairs:
            new_probs = self._softmax_rows(
                model.predict(
                    unseen_pairs,
                    show_progress_bar=False,
                    batch_size=settings.confidence.nli_batch_size,
                )
            )
            if len(self._pair_cache) > 4096:
                self._pair_cache.clear()
            for pair, prob in zip(unseen_pairs, new_probs, strict=True):
                self._pair_cache[pair] = prob

        probs = [self._pair_cache[pair] for pair in pairs]
        forward_pass_ms = (time.perf_counter() - t_fwd_0) * 1000

        order = self._label_order or ["contradiction", "entailment", "neutral"]
        idx = {name: i for i, name in enumerate(order)}
        e_i, n_i, c_i = idx.get("entailment", 1), idx.get("neutral", 2), idx.get("contradiction", 0)

        n_ctx = len(premises)
        out: list[ClaimScore] = []
        claims_debug_info = []

        for ci, (claim, hyp) in enumerate(zip(claims, hyps, strict=True)):
            rows = probs[ci * n_ctx : (ci + 1) * n_ctx]
            best = int(max(range(n_ctx), key=lambda j: (rows[j][e_i], rows[j][c_i])))
            best_entail = float(rows[best][e_i])
            best_contra = float(rows[best][c_i])
            best_neutral = float(rows[best][n_i])

            has_entailment = any(r[e_i] >= settings.confidence.support_threshold for r in rows)
            margin = settings.confidence.contradiction_margin

            # A claim is "Evident Conflict" only if:
            # 1. No premise entails it (not has_entailment)
            # 2. On its best-entailing premise, p_contra >= contradiction_threshold (0.50)
            # 3. AND (p_contra - p_entail) >= contradiction_margin (0.30)
            is_conflict = (
                not has_entailment
                and best_contra >= settings.confidence.contradiction_threshold
                and (best_contra - best_entail) >= margin
            )

            out.append(
                ClaimScore(
                    claim=claim,
                    entailment=best_entail,
                    neutral=best_neutral,
                    contradiction=best_contra,
                    best_context_index=origin[best],
                    is_conflict=is_conflict,
                )
            )

            if os.getenv("RAG_DEBUG_GATE_B") == "1":
                all_pairs_info = [
                    {
                        "premise": premises[j],
                        "premise_type": premise_types[j],
                        "origin_index": origin[j],
                        "p_entail": float(rows[j][e_i]),
                        "p_neutral": float(rows[j][n_i]),
                        "p_contra": float(rows[j][c_i]),
                    }
                    for j in range(n_ctx)
                ]
                top_3 = sorted(all_pairs_info, key=lambda x: x["p_entail"], reverse=True)[:3]
                veto_trigger = all_pairs_info[best] if is_conflict else None
                claims_debug_info.append(
                    {
                        "claim": hyp,
                        "raw_sentence": claim,
                        "is_supported": best_entail >= settings.confidence.support_threshold,
                        "is_contradicted": is_conflict,
                        "top_3_premises": top_3,
                        "all_pairs": all_pairs_info,
                        "veto_trigger": veto_trigger,
                    }
                )

        if os.getenv("RAG_DEBUG_GATE_B") == "1":
            from src.config import REPORTS_DIR

            debug_dump = {
                "query": question,
                "device": _device(),
                "model_load_time_ms": model_load_ms,
                "n_pairs_scored": len(pairs),
                "timings_ms": {
                    "pair_construction_ms": pair_construction_ms,
                    "forward_pass_ms": forward_pass_ms,
                    "total_score_claims_ms": (time.perf_counter() - t_start) * 1000,
                },
                "claims": claims_debug_info,
            }
            try:
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                (REPORTS_DIR / "debug_gateb_stack.json").write_text(
                    json.dumps(debug_dump, indent=2), encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to write debug_gateb_stack.json: %s", exc)

        return out

    def faithfulness(
        self,
        answer: str,
        contexts: list[str],
        *,
        question: str | None = None,
        mode: str | None = None,
        premise: str | None = None,
    ) -> tuple[float, list[ClaimScore]]:
        """Mean entailment probability over the answer's claims.

        Not the *fraction* of claims that clear ``support_threshold``: a mean of
        probabilities degrades smoothly, so an answer that is 0.49-entailed
        throughout scores near 0.49 instead of collapsing to 0.0 the way a
        hard-counted fraction would. The per-claim labels in ``ClaimScore``
        still use the threshold; only this aggregate ignores it.
        """
        claims = split_claims(answer)
        scores = self.score_claims(claims, contexts, question=question, mode=mode, premise=premise)
        if not scores:
            return 0.0, []
        return sum(s.entailment for s in scores) / len(scores), scores


class RerankScorer:
    """Gate A: query-passage relevance, computed before generation."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.confidence.rerank_model
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    def relevance(self, query: str, contexts: list[str]) -> list[float]:
        """Per-passage relevance in [0, 1].

        ms-marco cross-encoders emit an unbounded logit, so a sigmoid maps it
        into a probability-like range. The transform is monotonic, so it changes
        the calibration of the threshold but never the ranking.
        """
        if not contexts:
            return []
        import numpy as np

        raw = np.asarray(
            self._ensure().predict([(query, c) for c in contexts], show_progress_bar=False),
            dtype=float,
        ).ravel()
        return (1.0 / (1.0 + np.exp(-raw))).tolist()

    def contextual_relevancy(self, query: str, contexts: list[str]) -> tuple[float, list[float]]:
        """Aggregate Gate A signal.

        Uses the mean of the top-2 passages rather than the single best. One
        lucky high-scoring chunk is a weak basis for answering a question, and
        the max alone proved easy to saturate; the top-2 mean is the cheapest
        aggregate that resists both problems.
        """
        scores = self.relevance(query, contexts)
        if not scores:
            return 0.0, []
        ranked = sorted(scores, reverse=True)
        return float(sum(ranked[:2]) / min(2, len(ranked))), scores


def _repair_judge_json(t: str) -> str:
    """Fix the two malformations a small local judge produces inside JSON.

    qwen2.5:7b reliably omits the comma between consecutive string list items
    and occasionally drops one between object members. Rather than failing the
    metric call, insert commas where a quote-close is directly followed by a
    quote-open (or a closing brace) with nothing between.
    """
    return re.sub(r'(["\]}])\s*(?=["{])', r"\1, ", t)


def _parse_judge_json(text: str, schema):
    """Parse a local judge's reply into a DeepEval schema object.

    Small local models wrap JSON in code fences, prepend prose, or emit
    slightly malformed JSON. DeepEval's own fallback assumes a clean response
    and crashes the metric, so the recovery ladder lives here: strip fences,
    parse, repair, salvage the outermost object. One retry with an explicit
    "output only JSON" reminder handles the rest.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.removeprefix("```json").removeprefix("```").strip()
        if t.endswith("```"):
            t = t[: t.rfind("```")].strip()
    for candidate in (t, _repair_judge_json(t)):
        try:
            return schema(**json.loads(candidate))
        except ValueError:
            continue
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        body = t[start : end + 1]
        for candidate in (body, _repair_judge_json(body)):
            try:
                return schema(**json.loads(candidate))
            except ValueError:
                continue
    raise ValueError(f"unparseable judge reply: {text[:120]!r}")


def _build_deepeval_judge():
    """Create the DeepEval judge, subclassing its base class at runtime.

    Built dynamically because DeepEval's base class must be imported before it
    can be subclassed, and importing DeepEval at module scope would make the
    local NLI path depend on a package it never uses.

    ``generate`` honours DeepEval's ``schema=`` protocol: the metrics pass a
    pydantic model and expect an instance back. Without it the caller falls
    back to JSON-parsing the raw text, which a small local judge model fails
    at just often enough to crash the metric with "'str' object has no
    attribute 'verdicts'".
    """
    from deepeval.models.base_model import DeepEvalBaseLLM

    from src.llm import get_judge

    class Judge(DeepEvalBaseLLM):
        def __init__(self):
            self._chat = get_judge()

        def load_model(self):
            return self._chat

        def _invoke(self, prompt: str) -> str:
            return str(self._chat.invoke(prompt).content)

        def generate(self, prompt: str, *args, schema=None, **kwargs):
            text = self._invoke(prompt)
            if schema is None:
                return text
            try:
                return _parse_judge_json(text, schema)
            except ValueError:
                # One retry with an explicit format reminder; a second failure
                # propagates and the caller logs a judge error, as before.
                reminder = prompt + "\n\nRespond with ONLY the JSON object, no prose, no code fences."
                return _parse_judge_json(self._invoke(reminder), schema)

        async def a_generate(self, prompt: str, *args, schema=None, **kwargs):
            text = str((await self._chat.ainvoke(prompt)).content)
            if schema is None:
                return text
            return _parse_judge_json(text, schema)

        def get_model_name(self) -> str:
            return f"{settings.llm.provider}:{settings.llm.judge_model}"

    return Judge()


class LLMJudgeScorer:
    """Reference standard: DeepEval's Faithfulness + Contextual Relevancy.

    Each call costs several LLM round-trips, so this path is used for the
    agreement study and for spot checks in the UI, never for a full sweep over
    thousands of benchmark questions.
    """

    def __init__(self):
        self._judge = None

    def _ensure(self):
        if self._judge is None:
            self._judge = _build_deepeval_judge()
        return self._judge

    def _measure(self, metric_cls, *, query: str, answer: str, contexts: list[str]):
        from deepeval.test_case import LLMTestCase

        metric = metric_cls(model=self._ensure(), include_reason=True, async_mode=False)
        case = LLMTestCase(input=query, actual_output=answer, retrieval_context=contexts)
        metric.measure(case)
        return float(metric.score or 0.0), str(getattr(metric, "reason", "") or "")

    def faithfulness(self, query: str, answer: str, contexts: list[str]) -> tuple[float, str]:
        from deepeval.metrics import FaithfulnessMetric

        return self._measure(FaithfulnessMetric, query=query, answer=answer, contexts=contexts)

    def contextual_relevancy(
        self, query: str, contexts: list[str], answer: str = "n/a"
    ) -> tuple[float, str]:
        from deepeval.metrics import ContextualRelevancyMetric

        return self._measure(
            ContextualRelevancyMetric, query=query, answer=answer, contexts=contexts
        )


@functools.lru_cache(maxsize=1)
def get_nli_scorer() -> NLIScorer:
    return NLIScorer()


@functools.lru_cache(maxsize=1)
def get_rerank_scorer() -> RerankScorer:
    return RerankScorer()


@functools.lru_cache(maxsize=1)
def get_llm_judge() -> LLMJudgeScorer:
    return LLMJudgeScorer()






