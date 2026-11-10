"""Answer generation, constrained to the retrieved context.

The prompt does real work here. A generator told only "answer the question"
will happily fill gaps from parametric memory, which manufactures exactly the
hallucinations the confidence layer then has to catch. Instructing the model to
say when the context is insufficient gives us a second, independent abstention
signal that Gate B can be measured against.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_generator

log = logging.getLogger(__name__)

INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = """You are a careful question-answering assistant.

Rules you must follow:
1. Answer ONLY from the numbered context passages provided. Do not use outside knowledge.
2. If the passages do not contain enough information, reply with exactly: {marker}
3. Do not speculate, do not fill gaps, and do not add plausible-sounding detail.
4. Keep the answer under four sentences.
5. Cite the passage numbers you used in square brackets, e.g. [1] or [2][3].
"""

USER_PROMPT = """Context passages:
{context}

Question: {question}

Answer:"""


@dataclass
class Generation:
    answer: str
    self_reported_insufficient: bool
    latency_ms: float
    prompt_chars: int


def format_context(contexts: list[str]) -> str:
    """Number the passages so the model can cite them and we can parse citations."""
    return "\n\n".join(f"[{i + 1}] {c.strip()}" for i, c in enumerate(contexts))


def _extract_text(reply: object) -> str:
    raw = getattr(reply, "content", reply)
    if isinstance(raw, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
        ).strip()
    return str(raw).strip()


def generate(question: str, contexts: list[str]) -> Generation:
    t0 = time.perf_counter()
    context_block = format_context(contexts)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | get_generator()

    try:
        reply = chain.invoke(
            {
                "marker": INSUFFICIENT_MARKER,
                "context": context_block,
                "question": question,
            }
        )
        answer = _extract_text(reply)
    except Exception as exc:  # noqa: BLE001 - surfaced as a failed generation, not a crash
        log.error("Generation failed: %s", exc)
        answer = f"[generation error: {type(exc).__name__}: {exc}]"

    return Generation(
        answer=answer,
        self_reported_insufficient=INSUFFICIENT_MARKER in answer,
        latency_ms=(time.perf_counter() - t0) * 1000,
        prompt_chars=len(context_block) + len(question) + len(SYSTEM_PROMPT),
    )


def generate_unguarded(question: str, contexts: list[str]) -> Generation:
    """Baseline generation with no abstention instruction.

    This is the control condition for the evaluation chapter: the difference
    between this and ``generate`` isolates how much of the improvement comes
    from prompting alone versus from the confidence gates.
    """
    t0 = time.perf_counter()
    context_block = format_context(contexts)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question using the context passages provided."),
            ("human", USER_PROMPT),
        ]
    )
    try:
        reply = (prompt | get_generator()).invoke(
            {"context": context_block, "question": question}
        )
        answer = _extract_text(reply)
    except Exception as exc:  # noqa: BLE001
        log.error("Baseline generation failed: %s", exc)
        answer = f"[generation error: {type(exc).__name__}: {exc}]"

    return Generation(
        answer=answer,
        self_reported_insufficient=False,
        latency_ms=(time.perf_counter() - t0) * 1000,
        prompt_chars=len(context_block) + len(question),
    )
