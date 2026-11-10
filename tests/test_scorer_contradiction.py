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


def test_split_claims_numbered_list_segmentation():
    """Requirement 1: list/numbered answers are segmented into full-sentence claims,
    scaffolding is dropped, and citations/markdown markers are stripped."""
    from src.scorers import split_claims

    answer = (
        "The five stages of a standard RAG pipeline are:\n"
        "1. Ingestion — raw documents are loaded from sources [1].\n"
        "2. Chunking — documents are divided into passages [1], [2].\n"
        "3. Embedding — chunks are transformed into vector representations [2].\n"
        "4. Retrieval — the top matching chunks are fetched [3].\n"
        "5. Generation — an LLM generates the final answer [3].\n"
        "Summary."  # 1-word fragment, should be dropped (< 4 words)
    )
    claims = split_claims(answer)
    assert len(claims) == 5
    assert not any("The five stages" in c for c in claims)
    assert not any("[1]" in c or "[2]" in c or "[3]" in c for c in claims)
    assert not any(c.startswith("1.") or c.startswith("2.") for c in claims)
    assert not any(len(c.split()) < 4 for c in claims)
    assert claims[0].startswith("Ingestion")
    assert claims[4].startswith("Generation")


def test_contradiction_margin_boundary_conditions():
    """Requirement 4: margin rule at its boundary values (default margin = 0.30)."""
    # Boundary 1: contra=0.55, entail=0.26 -> diff=0.29 < 0.30 margin -> NOT evident conflict
    rows_sub_margin = [(0.26, 0.19, 0.55)]
    s1 = _StubNLI(rows_sub_margin).score_claims(["A claim with weak contrast."], ["c0"])
    assert not s1[0].is_contradicted
    assert s1[0].taxonomy_label == "baseless_information"

    # Boundary 2: contra=0.55, entail=0.25 -> diff=0.30 >= 0.30 margin -> IS evident conflict
    rows_at_margin = [(0.25, 0.20, 0.55)]
    s2 = _StubNLI(rows_at_margin).score_claims(["A claim at exact margin."], ["c0"])
    assert s2[0].is_contradicted
    assert s2[0].taxonomy_label == "evident_conflict"

    # Boundary 3: contra=0.49, entail=0.01 -> contra < 0.50 threshold -> NOT evident conflict
    rows_sub_thresh = [(0.01, 0.50, 0.49)]
    s3 = _StubNLI(rows_sub_thresh).score_claims(["A claim just below threshold."], ["c0"])
    assert not s3[0].is_contradicted
    assert s3[0].taxonomy_label == "baseless_information"

    # Boundary 4: entail=0.50 -> entailed claim is never contradicted even if distractor contra=0.99
    rows_entailed = [
        (0.50, 0.40, 0.10),  # Context 0 entails
        (0.00, 0.01, 0.99),  # Context 1 distractor spurious contra
    ]
    s4 = _StubNLI(rows_entailed).score_claims(["An entailed claim."], ["c0", "c1"])
    assert not s4[0].is_contradicted
    assert s4[0].is_supported
    assert s4[0].taxonomy_label == "supported"


def test_faithfulness_veto_vs_weak_flag(monkeypatch):
    """Gate B whole-answer cap of 0.15 is triggered only when >= 1 claim is a true conflict."""
    from src.confidence import ConfidenceEvaluator

    # Scenario A: Claim has weak flag without margin -> treated as unsupported (no veto cap)
    # 2 claims: claim 1 entailed (1.0), claim 2 weak contra (diff 0.25 < 0.30 margin)
    class _MultiStubNLI(NLIScorer):
        def __init__(self, table):
            super().__init__(model_name="stub")
            self._table = table

        def _ensure(self):
            class _M:
                def predict(_self, pairs, **_):
                    # return pre-configured rows for pairs
                    return [[c, e, n] for (e, n, c) in self._table]

            return _M()

        @staticmethod
        def _resolve_label_order(_):
            return ["contradiction", "entailment", "neutral"]

        def _softmax_rows(self, logits):
            return [list(r) for r in logits]

    # Claim 1: entail=0.90; Claim 2: entail=0.30, contra=0.55 (diff 0.25 < 0.30)
    scorer_no_veto = _MultiStubNLI([(0.90, 0.05, 0.05), (0.30, 0.15, 0.55)])
    monkeypatch.setattr("src.confidence.get_nli_scorer", lambda: scorer_no_veto)
    evaluator = ConfidenceEvaluator()
    res_no_veto = evaluator.gate_b(
        "query",
        "Claim one is supported. Claim two is weakly contested.",
        ["c0"],
    )
    # Mean of 0.90 and 0.30 = 0.60; should NOT be capped at 0.15
    assert not any(c.is_contradicted for c in res_no_veto.claims)
    assert res_no_veto.score == 0.60
    assert res_no_veto.passed

    # Scenario B: Claim has true conflict -> capped at 0.15
    # Claim 1: entail=0.90; Claim 2: entail=0.00, contra=0.95
    scorer_veto = _MultiStubNLI([(0.90, 0.05, 0.05), (0.00, 0.05, 0.95)])
    monkeypatch.setattr("src.confidence.get_nli_scorer", lambda: scorer_veto)
    res_veto = evaluator.gate_b(
        "query",
        "Claim one is supported. Claim two is a direct conflict.",
        ["c0"],
    )
    assert any(c.is_contradicted for c in res_veto.claims)
    assert res_veto.score == 0.15
    assert not res_veto.passed
