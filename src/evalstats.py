"""Selective-prediction statistics: risk-coverage, AURC, and judge agreement.

Model-free maths separated from the scripts that use it, for the same reason
the harmonic mean lives in ``confidence.py``: every number that ends up in the
final report should be unit-testable without a GPU or a language model.

Definitions (Geifman & El-Yaniv, 2017, and the SAnthiA/DeepRisk formulation):
  coverage - the fraction of questions the system chooses to answer.
  risk     - the error rate among those answered questions.
  AURC     - area under the risk-coverage curve; lower is better. It averages
             the risk over all coverage levels, so a system that answers
             everything at 30% error is worse than one that answers half at 5%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    coverage: float
    risk: float


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx = _ranks(xs)
    ry = _ranks(ys)
    return _pearson(rx, ry)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _binary_agreement(xs: list[bool], ys: list[bool]) -> dict:
    """Cohen's kappa and raw proportions for two binary labellings."""
    n = len(xs)
    if n == 0:
        return {"n": 0}
    yes_yes = sum(1 for a, b in zip(xs, ys, strict=True) if a and b)
    yes_no = sum(1 for a, b in zip(xs, ys, strict=True) if a and not b)
    no_yes = sum(1 for a, b in zip(xs, ys, strict=True) if not a and b)
    no_no = sum(1 for a, b in zip(xs, ys, strict=True) if not a and not b)
    po = (yes_yes + no_no) / n
    px = (yes_yes + yes_no) / n
    py = (yes_yes + no_yes) / n
    pe = px * py + (1 - px) * (1 - py)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {
        "n": n,
        "observed_agreement": round(po, 4),
        "cohens_kappa": round(kappa, 4) if not math.isnan(kappa) else None,
        "both_pass": yes_yes,
        "only_x": yes_no,
        "only_y": no_yes,
        "both_fail": no_no,
    }


def risk_coverage_curve(scores: list[float], correct: list[bool]) -> list[Point]:
    """The risk-coverage curve for one arm.

    Sorts questions by confidence descending, then walks down the ranking:
    answering the top k is every coverage level, and the risk at each level is
    the fraction of wrong answers among those k. Ties are handled by treating
    all members of a tie block as included together (a tie block must not
    produce a coverage point mid-block, or the curve depends on sort order).
    """
    if len(scores) != len(correct):
        raise ValueError("scores and correct must be the same length")
    if not scores:
        return []
    pairs = sorted(zip(scores, correct, strict=True), key=lambda p: -p[0])

    total = len(pairs)
    n_wrong = 0
    curve = [Point(0.0, 0.0)]
    i = 0
    while i < total:
        # Extend through the whole tie block.
        j = i
        while j + 1 < total and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            n_wrong += 0 if pairs[k][1] else 1
        n_ans = j + 1
        curve.append(Point(n_ans / total, n_wrong / n_ans))
        i = j + 1
    return curve


def aurc(curve: list[Point]) -> float:
    """Area under the risk-coverage curve via the trapezoidal rule.

    Risk at full coverage is included, so always-answering systems get their
    raw error rate as their AURC and selective systems can only do better.
    """
    if len(curve) < 2:
        return float("nan")
    area = 0.0
    for a, b in zip(curve[:-1], curve[1:], strict=True):
        area += (b.coverage - a.coverage) * (a.risk + b.risk) / 2
    return area


def risk_at_coverage(curve: list[Point], coverage: float) -> float:
    """Risk when answering at most ``coverage`` of the questions.

    The highest-coverage point at or below the target; if the target sits above
    every recorded point, the last point is returned.
    """
    if not curve:
        return float("nan")
    best = curve[0]
    for p in curve:
        if p.coverage <= coverage + 1e-9:
            best = p
    return best.risk


def coverage_at_risk(curve: list[Point], risk_cap: float) -> float:
    """The widest coverage whose risk stays at or below the cap.

    This is the operating point a deployment would actually choose: "answer as
    many questions as you can without exceeding X% error".
    """
    if not curve:
        return float("nan")
    best = curve[0]
    for p in curve:
        if p.risk <= risk_cap + 1e-9:
            best = max(best, p, key=lambda q: q.coverage)
    return best.coverage


def rate_matched_threshold(scores: list[float], n_pass: int) -> float:
    """A cut-off on ``scores`` that lets exactly ``n_pass`` items through.

    Used to compare two scorers whose scales do not coincide. Applying one
    scorer's calibrated threshold to the other measures whether their *scales*
    line up, which is not the question being asked: the question is whether they
    rank the same items as the good ones. Fixing the pass count removes the
    scale from the comparison and leaves the ordering, which is what a scorer
    is actually being trusted to get right.

    Returns a threshold just at or below the ``n_pass``-th highest score. With
    ``n_pass`` of 0 the threshold sits above every score; with ``n_pass`` at or
    beyond the sample size it sits at or below the minimum.
    """
    if not scores:
        return float("nan")
    ranked = sorted(scores, reverse=True)
    if n_pass <= 0:
        return ranked[0] + 1.0
    if n_pass >= len(ranked):
        return ranked[-1]
    return ranked[n_pass - 1]


def judge_agreement(
    nli_scores: list[float],
    llm_scores: list[float],
    threshold: float,
    *,
    llm_threshold: float | None = None,
) -> dict:
    """Agreement between two scorers on the same items.

    Two different questions are reported, because they have different answers
    and conflating them is how a scale mismatch gets mistaken for a
    disagreement:

    ``decision_agreement`` applies ``threshold`` to the local scorer and
    ``llm_threshold`` to the judge. When ``llm_threshold`` is not supplied it
    defaults to ``threshold``, i.e. one cut-off for both - which is only
    meaningful when the two scores share a scale. They frequently do not: a
    judge whose relevancy score never exceeds 0.45 will fail every item under a
    0.45 threshold calibrated for a reranker, and the resulting kappa of zero
    says nothing about whether the two agree on which passages are relevant.

    ``decision_agreement_rate_matched`` removes the scale entirely by giving the
    judge whichever cut-off lets through the same *number* of items the local
    scorer passed. Kappa then measures agreement on the ranking, which is the
    claim being tested - that the cheap scorer picks out the same items.

    Spearman's rho is reported alongside for the same reason: it depends only on
    order, so it survives a scale mismatch that destroys Pearson.
    """
    if len(nli_scores) != len(llm_scores):
        raise ValueError("score lists must be the same length")
    # NaN as well as None: reading telemetry through pandas turns SQL NULL into
    # float('nan'), which passes an ``is not None`` test and then silently
    # poisons every sum it touches. Rows where only one scorer ran are simply
    # not comparable, so they are dropped here rather than reported as NaN.
    pairs = [
        (nli, llm)
        for nli, llm in zip(nli_scores, llm_scores, strict=True)
        if nli is not None and llm is not None
        and not math.isnan(float(nli)) and not math.isnan(float(llm))
    ]
    if not pairs:
        return {"n": 0}
    ns = [float(p[0]) for p in pairs]
    ls = [float(p[1]) for p in pairs]

    t_llm = threshold if llm_threshold is None else llm_threshold
    nli_pass = [s >= threshold for s in ns]
    n_pass = sum(nli_pass)
    t_matched = rate_matched_threshold(ls, n_pass)

    return {
        "n": len(pairs),
        "pearson": round(_pearson(ns, ls), 4),
        "spearman": round(_spearman(ns, ls), 4),
        "mean_nli": round(sum(ns) / len(ns), 4),
        "mean_llm": round(sum(ls) / len(ls), 4),
        # How much of the judge's scale is actually in play. A judge emitting
        # only two distinct values is a binary verdict, and correlating a
        # binary series against a continuous one is close to meaningless.
        "llm_range": [round(min(ls), 4), round(max(ls), 4)],
        "llm_distinct_values": len(set(ls)),
        "nli_threshold": round(threshold, 4),
        "llm_threshold": round(t_llm, 4),
        "decision_agreement": _binary_agreement(
            nli_pass, [s >= t_llm for s in ls]
        ),
        "llm_threshold_rate_matched": round(t_matched, 4),
        "decision_agreement_rate_matched": _binary_agreement(
            nli_pass, [s >= t_matched for s in ls]
        ),
    }


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Preferred over the textbook ``p +/- 1.96*sqrt(p(1-p)/n)`` because that
    interval collapses to zero width at p = 0 or 1 and spills outside [0, 1]
    near the edges - exactly where a 3/75 hallucination count sits.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    Two runs over the *same* items are paired, so comparing their rates with an
    unpaired test throws information away. Only the discordant items (the
    outcome changed between runs) carry evidence; under the null each flip is
    equally likely to go either way, which is a Binomial(n, 0.5) test.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def auroc(scores: list[float], positive: list[bool]) -> float:
    """P(score of a random positive > score of a random negative), ties at half.

    The question "does this confidence score separate correct answers from
    wrong ones" independent of any threshold. 0.5 is a coin flip.
    """
    pos = [s for s, y in zip(scores, positive, strict=True) if y]
    neg = [s for s, y in zip(scores, positive, strict=True) if not y]
    if not pos or not neg:
        return float("nan")
    ranks = _ranks(list(scores))
    rank_sum = sum(r for r, y in zip(ranks, positive, strict=True) if y)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
