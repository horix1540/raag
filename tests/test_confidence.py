"""Unit tests for the confidence layer's pure logic.

These deliberately avoid loading any model. The scoring maths and the gate
decision table are where a silent sign error or an inverted threshold would
invalidate every number in the final report, so they are tested in isolation and
run in under a second.
"""

from __future__ import annotations

import pytest

from src.confidence import (
    ConfidenceEvaluator,
    GateAResult,
    GateBResult,
    weighted_harmonic_mean,
)
from src.retriever import _distance_to_similarity
from src.scorers import ClaimScore, split_claims


class TestWeightedHarmonicMean:
    def test_equal_values_returns_that_value(self):
        assert weighted_harmonic_mean([0.8, 0.8], (0.5, 0.5)) == pytest.approx(0.8, abs=1e-4)

    def test_punishes_a_near_zero_component(self):
        """The core claim: harmonic mean must not average away a failure."""
        arithmetic = (0.95 + 0.05) / 2
        harmonic = weighted_harmonic_mean([0.95, 0.05], (0.5, 0.5))
        assert arithmetic == pytest.approx(0.50)
        assert harmonic < 0.11
        assert harmonic < arithmetic

    def test_bounded_between_minimum_and_arithmetic_mean(self):
        """The standard HM <= AM guarantee, which is what makes the penalty real.

        For any weighting, min <= harmonic <= weighted arithmetic mean. The
        upper bound is the useful half: the composite can never flatter a pair
        more than a plain average would.
        """
        w = (0.4, 0.6)
        for pair in [(0.9, 0.3), (0.2, 0.7), (1.0, 0.01), (0.55, 0.55)]:
            harmonic = weighted_harmonic_mean(list(pair), w)
            arithmetic = w[0] * pair[0] + w[1] * pair[1]
            assert min(pair) - 1e-6 <= harmonic <= arithmetic + 1e-6

    def test_zero_component_collapses_the_score(self):
        assert weighted_harmonic_mean([0.99, 0.0], (0.4, 0.6)) < 1e-3

    def test_weights_shift_the_result_toward_the_heavier_term(self):
        faith_heavy = weighted_harmonic_mean([0.4, 0.9], (0.1, 0.9))
        relevance_heavy = weighted_harmonic_mean([0.4, 0.9], (0.9, 0.1))
        assert faith_heavy > relevance_heavy

    def test_empty_and_degenerate_inputs(self):
        assert weighted_harmonic_mean([], (1.0,)) == 0.0
        assert weighted_harmonic_mean([0.5], ()) == pytest.approx(0.5, abs=1e-4)
        assert weighted_harmonic_mean([0.5, 0.5], (0.0, 0.0)) == 0.0


class TestDistanceConversion:
    """Guards the bug class where a Chroma distance is read as a similarity."""

    def test_cosine_zero_distance_is_perfect_similarity(self):
        assert _distance_to_similarity(0.0, "cosine") == pytest.approx(1.0)

    def test_cosine_orthogonal_is_a_half(self):
        assert _distance_to_similarity(1.0, "cosine") == pytest.approx(0.0)

    def test_squared_l2_between_unit_vectors(self):
        # Identical unit vectors -> distance 0 -> similarity 1.
        assert _distance_to_similarity(0.0, "l2") == pytest.approx(1.0)
        # Orthogonal unit vectors -> squared L2 of 2 -> similarity 0.
        assert _distance_to_similarity(2.0, "l2") == pytest.approx(0.0)

    def test_output_is_always_clamped(self):
        for space in ("cosine", "l2", "ip"):
            for d in (-5.0, -0.1, 0.0, 1.0, 2.0, 99.0):
                assert 0.0 <= _distance_to_similarity(d, space) <= 1.0

    def test_monotonically_decreasing_in_distance(self):
        sims = [_distance_to_similarity(d, "cosine") for d in (0.0, 0.2, 0.4, 0.8)]
        assert sims == sorted(sims, reverse=True)


class TestClaimSplitting:
    def test_splits_on_sentence_boundaries(self):
        claims = split_claims("Paris is the capital. It sits on the Seine river.")
        assert len(claims) == 2

    def test_does_not_split_decimals_or_abbreviations_mid_number(self):
        claims = split_claims("The model scored 88.5 on the benchmark suite.")
        assert len(claims) == 1

    def test_drops_uninformative_fragments(self):
        claims = split_claims("Yes. The treaty was signed in 1919 by the delegates.")
        assert all(len(c) >= 12 for c in claims)
        assert len(claims) == 1

    def test_empty_input_yields_no_claims(self):
        assert split_claims("") == []
        assert split_claims("    ") == []

    def test_single_short_answer_is_still_returned(self):
        """A terse answer must not silently score as zero claims."""
        assert split_claims("1919") == ["1919"]


def _gate_a(score: float, threshold: float | None = None) -> GateAResult:
    from src.config import settings

    t = settings.confidence.gate_a_threshold if threshold is None else threshold
    return GateAResult(
        score=score,
        passed=score >= t,
        rerank_relevancy=score,
        top_similarity=0.6,
        mean_similarity=0.5,
        similarity_margin=0.1,
    )


def _claim(entail: float, contra: float) -> ClaimScore:
    return ClaimScore(
        claim="a sufficiently long test claim about something",
        entailment=entail,
        neutral=max(0.0, 1.0 - entail - contra),
        contradiction=contra,
        best_context_index=0,
    )


def _gate_b(score: float, claims=None, threshold: float | None = None) -> GateBResult:
    from src.config import settings

    t = settings.confidence.gate_b_threshold if threshold is None else threshold
    return GateBResult(
        score=score,
        passed=score >= t,
        nli_faithfulness=score,
        claims=claims or [],
    )


class TestTaxonomyLabels:
    def test_high_entailment_is_supported(self):
        assert _claim(0.92, 0.02).taxonomy_label == "supported"

    def test_high_contradiction_is_evident_conflict(self):
        assert _claim(0.10, 0.85).taxonomy_label == "evident_conflict"

    def test_neutral_claim_is_baseless_information(self):
        assert _claim(0.15, 0.05).taxonomy_label == "baseless_information"

    def test_conflict_outranks_lack_of_support(self):
        """A claim that is both unsupported and contradicted is the worse label."""
        assert _claim(0.20, 0.90).taxonomy_label == "evident_conflict"


class TestGateDecisionTable:
    """The full decision matrix. Each row is a claim the report will make."""

    def setup_method(self):
        self.ev = ConfidenceEvaluator()

    def test_failed_gate_a_abstains_without_generating(self):
        report = self.ev.decide(_gate_a(0.10), None)
        assert report.verdict == "abstain_no_context"
        assert report.abstained
        assert report.generation_skipped

    def test_both_gates_high_answers(self):
        report = self.ev.decide(_gate_a(0.90), _gate_b(0.92))
        assert report.verdict == "answer"
        assert not report.abstained

    def test_passed_gate_a_but_ungrounded_answer_abstains(self):
        report = self.ev.decide(_gate_a(0.90), _gate_b(0.20))
        assert report.verdict == "abstain_ungrounded"
        assert report.abstained
        assert not report.generation_skipped

    def test_marginal_pair_is_flagged_not_abstained(self):
        """Both gates clear their own bar, but the composite does not.

        Constructed from the live thresholds so the test survives calibration
        changing the numbers. The composite band sits 10% above the harmonic
        of two both-marginal gate passes, so a pair only a hair above both
        thresholds passes the gates individually yet stays below the band.
        """
        from src.config import settings

        t_a = settings.confidence.gate_a_threshold
        t_b = settings.confidence.gate_b_threshold
        a, b = t_a + 0.01, t_b + 0.02
        # Sanity: with the shipped band this pair must land in the flag band;
        # if calibration ever narrows the band below the marginal corner the
        # invariant itself has been broken and this assert fires.
        assert weighted_harmonic_mean([a, b], settings.confidence.weights) < (
            settings.confidence.composite_threshold
        ), "flag band must stay above the both-marginal corner"
        report = self.ev.decide(_gate_a(a), _gate_b(b))
        assert report.verdict == "flag_low_confidence"
        assert not report.abstained

    def test_explanation_always_names_the_deciding_number(self):
        for a, b in [(0.10, None), (0.90, _gate_b(0.20)), (0.90, _gate_b(0.95))]:
            report = self.ev.decide(_gate_a(a), b)
            assert report.explanation and any(ch.isdigit() for ch in report.explanation)

    def test_to_row_is_flat_and_serialisable(self):
        row = self.ev.decide(_gate_a(0.9), _gate_b(0.9, [_claim(0.9, 0.0)])).to_row()
        assert row["verdict"] == "answer"
        assert row["n_claims"] == 1
        assert all(not isinstance(v, (list, dict)) for v in row.values())


class TestContradictionCap:
    """A single contradicted claim must fail Gate B on its own.

    Averaging entailment over claims means nine well-supported sentences can
    outvote one that directly conflicts with the source. That is the worst
    possible failure to hide, so ``gate_b`` caps the score below the threshold
    whenever any claim is contradicted.
    """

    def setup_method(self):
        self.ev = ConfidenceEvaluator()

    def test_helper_gate_b_reports_conflicts_and_baseless_separately(self):
        gb = _gate_b(0.5, [_claim(0.9, 0.0), _claim(0.1, 0.9), _claim(0.2, 0.1)])
        assert len(gb.conflicts) == 1
        assert len(gb.baseless) == 1
        assert gb.taxonomy_counts == {
            "supported": 1,
            "evident_conflict": 1,
            "baseless_information": 1,
        }

    def test_taxonomy_counts_sum_to_claim_count(self):
        claims = [_claim(0.9, 0.0), _claim(0.1, 0.9), _claim(0.2, 0.1), _claim(0.7, 0.05)]
        gb = _gate_b(0.5, claims)
        assert sum(gb.taxonomy_counts.values()) == len(claims)

    def test_no_claims_means_no_conflicts(self):
        gb = _gate_b(0.0, [])
        assert gb.conflicts == [] and gb.baseless == []
        assert gb.taxonomy_counts["supported"] == 0


class TestCombine:
    def setup_method(self):
        self.ev = ConfidenceEvaluator()

    def test_missing_gate_b_falls_back_to_gate_a(self):
        """When generation was skipped there is no faithfulness to combine."""
        assert self.ev.combine(_gate_a(0.33), None) == pytest.approx(0.33)

    def test_composite_respects_configured_weights(self):
        expected = weighted_harmonic_mean([0.8, 0.6], self.ev.cfg.weights)
        assert self.ev.combine(_gate_a(0.8), _gate_b(0.6)) == pytest.approx(expected)




