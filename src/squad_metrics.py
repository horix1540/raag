"""SQuAD 2.0 answer scoring: the official normalisation, EM and F1.

The metrics are kept in their own model-free module so they can be unit-tested
against the official examples and reused by any evaluation script. The
normalisation follows the reference implementation released with the paper
("Know What You Don't Know", Rajpurkar et al., 2018): lowercase, strip
punctuation and articles, collapse whitespace.

The no-answer convention is what makes SQuAD 2.0 usable for an abstention
study: an abstention is scored as an empty prediction, which earns full credit
on unanswerable questions (where the gold answer is empty) and no credit on
answerable ones. An empty gold answer with a non-empty prediction scores the
raw overlap against that prediction - again matching the reference behaviour.
"""

from __future__ import annotations

import re
import string

# Abstention surfaces as one of these strings rather than an empty prediction.
ABSTAIN_MARKERS = (
    "insufficient_context",  # the generator's own refusal marker
    "not confident enough",  # config.abstain_message
)


# Passage citations the guarded prompt asks for: "[1]", "[2][3]", "[1, 2]".
_CITATION = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


def strip_citations(text: str) -> str:
    """Remove passage citation markers from a prediction before scoring.

    The guarded prompt instructs the model to cite passages, so a correct
    answer arrives as "1894 [1]". Normalisation strips the brackets but keeps
    the digit, which turns it into "1894 1" and fails exact match against
    "1894". Before this was applied every guarded answer scored EM 0
    regardless of content. The marker is formatting the pipeline asked for,
    not part of the answer, so it is removed from predictions only.
    """
    return _CITATION.sub(" ", str(text))


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles/whitespace - the SQuAD reference."""

    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        return t.translate(str.maketrans("", "", string.punctuation))

    text = str(text).lower()
    return white_space_fix(remove_articles(remove_punc(text)))


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def is_abstention(prediction: str) -> bool:
    """True when the pipeline withheld an answer (gate refusal or self-refusal)."""
    if not prediction or not str(prediction).strip():
        return True
    low = str(prediction).lower()
    return any(marker in low for marker in ABSTAIN_MARKERS)


def exact_match_score(prediction: str, gold_answers: list[str]) -> float:
    """1.0 if the prediction normalises to any gold answer, else 0.0.

    An empty gold list (unanswerable) is matched by an empty/abstaining
    prediction, per the SQuAD 2.0 convention.
    """
    pred = normalize_answer(strip_citations(prediction)) if not is_abstention(prediction) else ""
    if not gold_answers:
        return float(pred == "")
    if not pred:
        return 0.0
    return float(any(pred == normalize_answer(g) for g in gold_answers))


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Token-overlap F1 against the best-matching gold answer.

    Unanswerable: an abstaining prediction scores 1.0, anything that produced
    text scores 0.0 (the reference scores the overlap against an empty string,
    which is 0 unless the prediction is also empty - same thing here).
    """
    if not gold_answers:
        return float(is_abstention(prediction))

    pred_toks = _tokens(strip_citations(prediction)) if not is_abstention(prediction) else []
    if not pred_toks:
        return 0.0

    best = 0.0
    for gold in gold_answers:
        gold_toks = _tokens(gold)
        common = set(pred_toks) & set(gold_toks)
        if not common:
            continue
        precision = len(common) / len(pred_toks)
        recall = len(common) / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains_gold(prediction: str, gold_answers: list[str]) -> bool:
    """True when a normalised gold answer appears as a token span in the prediction.

    EM and F1 were designed for extractive span predictors. A generative model
    answers in a sentence ("The war occurred in 1894"), which EM always fails
    and F1 dilutes by the sentence length. Gold containment is the usual
    accuracy for generative QA. It is lenient - a rambling answer that lists
    several candidates can contain the gold - so it is reported next to EM/F1,
    not instead of them. Unanswerable items follow the SQuAD convention:
    correct only when the prediction is an abstention.
    """
    if not gold_answers:
        return is_abstention(prediction)
    if is_abstention(prediction):
        return False
    pred = f" {normalize_answer(strip_citations(prediction))} "
    return any(
        (g := normalize_answer(gold)) and f" {g} " in pred for gold in gold_answers
    )
