"""Unit tests for the selective-prediction statistics used in Milestone 3.

Model-free: risk-coverage maths, AURC, the tie handling, and the judge
agreement statistics. AURC and kappa are what the final report's central
comparison rests on, so their edge cases are pinned here.
"""

from __future__ import annotations

import math

import pytest

from src.evalstats import (
    _pearson,
    _ranks,
    _spearman,
    aurc,
    auroc,
    coverage_at_risk,
    judge_agreement,
    mcnemar_exact,
    risk_at_coverage,
    risk_coverage_curve,
    wilson_interval,
)


def curve_points(pairs):
    """[(score, correct)] -> [(coverage, risk)] for compact assertions."""
    scores = [s for s, _ in pairs]
    correct = [c for _, c in pairs]
    return [(p.coverage, p.risk) for p in risk_coverage_curve(scores, correct)]


class TestRiskCoverageCurve:
    def test_perfect_confidence_reaches_zero_risk_then_fills_in(self):
        pts = curve_points([(0.9, True), (0.8, True), (0.2, False), (0.1, False)])
        # Coverage 0.5 (top two, both correct) has zero risk.
        assert (0.5, 0.0) in pts
        # Full coverage has 50% risk (two wrong of four).
        assert (1.0, 0.5) in pts

    def test_unsorted_input_is_handled(self):
        assert curve_points([(0.1, False), (0.9, True), (0.8, True), (0.2, False)]) == curve_points(
            [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
        )

    def test_ties_form_one_coverage_point(self):
        # Three items at the same score: the curve must jump in one step, so
        # there is no point at coverage 1/3 whose risk depends on sort order.
        pts = curve_points([(0.5, True), (0.5, False), (0.5, True)])
        coverages = [c for c, _ in pts]
        assert 1 / 3 not in coverages
        assert (1.0, 1 / 3) in pts

    def test_empty_input(self):
        assert risk_coverage_curve([], []) == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            risk_coverage_curve([0.5], [True, False])


class TestAURC:
    def test_perfect_system_has_zero_aurc(self):
        assert aurc(risk_coverage_curve([0.9, 0.8, 0.7], [True] * 3)) == pytest.approx(0.0)

    def test_always_wrong_full_coverage_aurc_is_one(self):
        # AURC includes the risk-free zero-coverage origin, so an always-wrong
        # system lands at 1 - 1/(2n), not 1.0; it approaches 1 as n grows.
        n = 3
        assert aurc(risk_coverage_curve([0.9, 0.8, 0.7], [False] * n)) == pytest.approx(1 - 1 / (2 * n))

    def test_selective_beats_confidently_wrong(self):
        # Both are 50% wrong overall, but the selective one ranks its errors
        # last: risk only starts climbing at coverage 0.5 and reaches 0.5 at
        # full coverage, giving AURC = 1/24 + 5/48 = 7/48. The inverted
        # ranking pays that cost immediately and scores far higher.
        selective = aurc(risk_coverage_curve([0.9, 0.8, 0.2, 0.1], [True, True, False, False]))
        wrong_ranked = aurc(risk_coverage_curve([0.9, 0.8, 0.2, 0.1], [False, False, True, True]))
        assert selective == pytest.approx(7 / 48)
        assert wrong_ranked > selective

    def test_too_few_points_is_nan(self):
        assert math.isnan(aurc([1.0]))


class TestOperatingPoints:
    CURVE = risk_coverage_curve(
        [0.9, 0.8, 0.7, 0.2], [True, True, True, False]
    )

    def test_risk_at_coverage(self):
        assert risk_at_coverage(self.CURVE, 0.5) == pytest.approx(0.0)
        assert risk_at_coverage(self.CURVE, 1.0) == pytest.approx(0.25)

    def test_coverage_at_risk_cap(self):
        # Widest coverage with risk <= 0 is 75%.
        assert coverage_at_risk(self.CURVE, 0.0) == pytest.approx(0.75)
        assert coverage_at_risk(self.CURVE, 0.25) == pytest.approx(1.0)


class TestCorrelation:
    def test_pearson_perfect_linear(self):
        assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
        assert _pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)

    def test_pearson_zero_variance_is_nan(self):
        assert math.isnan(_pearson([1, 1, 1], [1, 2, 3]))

    def test_ranks_handle_ties(self):
        assert _ranks([0.9, 0.5, 0.5, 0.1]) == [4.0, 2.5, 2.5, 1.0]

    def test_spearman_monotonic_nonlinear_is_perfect(self):
        assert _spearman([1, 2, 3], [1, 10, 100]) == pytest.approx(1.0)


class TestJudgeAgreement:
    def test_perfect_agreement(self):
        r = judge_agreement([0.9, 0.8, 0.1], [0.95, 0.85, 0.05], 0.5)
        assert r["pearson"] > 0.99
        d = r["decision_agreement"]
        assert d["cohens_kappa"] == 1.0
        assert d["observed_agreement"] == 1.0

    def test_systematic_disagreement_on_decisions(self):
        # NLI passes all; judge fails all. Observed agreement 0 -> kappa 0.
        r = judge_agreement([0.9, 0.9], [0.1, 0.1], 0.5)
        assert r["decision_agreement"]["cohens_kappa"] == 0.0
        assert r["decision_agreement"]["observed_agreement"] == 0.0

    def test_none_scores_are_dropped(self):
        r = judge_agreement([0.9, None, 0.1], [0.9, 0.4, 0.1], 0.5)
        assert r["n"] == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            judge_agreement([0.5], [0.5, 0.6], 0.5)

    def test_kappa_corrects_for_chance(self):
        # Both agree on 9/10, both overwhelmingly say "pass". High raw
        # agreement, but kappa must stay modest because most of it is the
        # base rate both scorers share.
        xs = [True] * 8 + [False, True]   # 9 pass, 1 fail
        ys = [True] * 9 + [False]         # 9 pass, 1 fail
        from src.evalstats import _binary_agreement

        d = _binary_agreement(xs, ys)
        assert d["observed_agreement"] == pytest.approx(0.8)
        assert d["cohens_kappa"] < 0.5


class TestIntervalsAndTests:
    def test_wilson_contains_point_and_stays_in_unit_range(self):
        lo, hi = wilson_interval(3, 75)
        assert 0.0 <= lo < 3 / 75 < hi <= 1.0
        lo, hi = wilson_interval(0, 10)
        assert lo == 0.0 and hi > 0.0

    def test_wilson_reference_value(self):
        # 8/75: reference Wilson interval is about [0.055, 0.197].
        lo, hi = wilson_interval(8, 75)
        assert abs(lo - 0.0551) < 1e-3 and abs(hi - 0.1972) < 1e-3

    def test_mcnemar_symmetric_and_bounded(self):
        assert mcnemar_exact(0, 0) == 1.0
        assert mcnemar_exact(1, 6) == mcnemar_exact(6, 1)
        # 0 vs 6 discordant: 2 * 0.5**6 = 0.03125
        assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-12

    def test_auroc_extremes(self):
        assert auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
        assert auroc([0.1, 0.2, 0.9, 0.8], [True, True, False, False]) == 0.0
        assert auroc([0.5, 0.5], [True, False]) == 0.5
