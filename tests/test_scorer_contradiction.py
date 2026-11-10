"""Unit tests for the Gate B contradiction-attribution rule.

nli-deberta's contradiction head fires spuriously on premises that have
nothing to do with the claim (measured on the SQuAD harness: an unrelated
etymology paragraph "contradicts" a steam-engine claim at p=0.92). These
tests pin the rule that contradiction is read only from the context the
model actually engages with, using a stub that reproduces exactly those
probability shapes.
"""

from __future__ import annotations

from src.scorers import NLIScorer


class _StubNLI(NLIScorer):
    """NLIScorer with the model replaced by a canned probability table.

    ``rows`` takes one (entailment, neutral, contradiction) triple per context
    in the natural reading order of the tests; ``predict`` re-orders it into
    the checkpoint's (contradiction, entailment, neutral) column order that
    ``score_claims`` maps via ``_label_order``.
    """

    def __init__(self, rows: list[tuple[float, float, float]]):
        super().__init__(model_name="stub")
        self._rows = rows

    def _ensure(self):
        table = [  # reorder (e, n, c) -> checkpoint order (c, e, n)
            [c, e, n] for (e, n, c) in self._rows
        ]

        class _M:  # noqa: N806 - keeps the shape of the real predict() call
            def predict(_self, pairs, show_progress_bar=False, batch_size=32):  # noqa: N805
                # score_claims builds pairs as (ctx, claim) per claim per ctx;
                # one claim only here, so return one row per context.
                return [list(r) for r in table]

        return _M()

    @staticmethod
    def _resolve_label_order(model):  # the documented default order
        return ["contradiction", "entailment", "neutral"]

    def _softmax_rows(self, logits):
        return [list(r) for r in logits]


def test_unrelated_premise_noise_cannot_fail_an_entailed_claim():
    """The Warsaw-vs-Savery shape: gold context entails, a distractor's
    contradiction head fires on a claim it never engages."""
    rows = [
        (0.90, 0.10, 0.00),  # gold context: clearly entails
        (0.00, 0.99, 0.01),
        (0.00, 0.07, 0.93),  # unrelated distractor, spurious contradiction
    ]
    s = _StubNLI(rows).score_claims(["Thomas Savery developed the first steam device."], ["c0", "c1", "c2"])
    assert s[0].entailment > 0.5
    assert s[0].contradiction < 0.5, "unrelated-premise noise must not fail an entailed claim"
    assert s[0].taxonomy_label == "supported"


def test_engaged_contradiction_still_fails_the_claim():
    """The smoke-test shape: a real conflict must survive the rule."""
    rows = [
        (0.00, 0.00, 0.99),  # the context the claim engages, contradicting it
        (0.00, 0.99, 0.01),
    ]
    s = _StubNLI(rows).score_claims(["A RAG pipeline has eleven stages."], ["c0", "c1"])
    assert s[0].contradiction > 0.5
    assert s[0].taxonomy_label == "evident_conflict"


def test_conflict_wins_when_nothing_supports_the_claim():
    """No context entails, one contradicts: the strongest contradiction over
    all passages decides, exactly as before the fix."""
    rows = [
        (0.00, 0.05, 0.35),
        (0.05, 0.05, 0.90),  # engaged contradiction, no support anywhere
        (0.00, 0.99, 0.01),
    ]
    s = _StubNLI(rows).score_claims(["The treaty was signed in 1900."], ["c0", "c1", "c2"])
    assert s[0].contradiction > 0.5
    assert s[0].taxonomy_label == "evident_conflict"


def test_weak_entailment_still_owns_the_decision():
    """A claim supported at only 0.60 by one context keeps a spurious 0.90
    contradiction from an unrelated premise at bay - the false-positive shape
    measured 14 times on the SQuAD harness."""
    rows = [
        (0.60, 0.05, 0.35),  # weak but real entailment
        (0.00, 0.07, 0.90),  # unrelated distractor noise
    ]
    s = _StubNLI(rows).score_claims(["The treaty was signed in 1900."], ["c0", "c1"])
    assert s[0].entailment == 0.60
    assert s[0].contradiction == 0.35, "the entailing context owns the contradiction call"
    assert s[0].taxonomy_label == "supported"


def test_entailing_context_outranks_a_contradicting_distractor():
    """The Savery case in its hardest form: a distractor's spurious
    contradiction sums slightly higher than the gold context's entailment.
    Entailment wins because a supporting source grounds the claim."""
    rows = [
        (0.90, 0.10, 0.00),  # gold context entails
        (0.00, 0.07, 0.93),  # unrelated distractor, spurious contradiction
    ]
    s = _StubNLI(rows).score_claims(["Thomas Savery developed the first steam device."], ["c0", "c1"])
    assert s[0].taxonomy_label == "supported"
    assert s[0].contradiction < 0.5


def test_neutral_everywhere_stays_baseless_not_conflicted():
    rows = [
        (0.00, 0.99, 0.01),
        (0.00, 0.99, 0.01),
        (0.00, 0.98, 0.02),  # noise floor on an unrelated premise
    ]
    s = _StubNLI(rows).score_claims(["New Orleans [1]"], ["c0", "c1", "c2"])
    assert s[0].taxonomy_label == "baseless_information"
    assert s[0].contradiction < 0.5
