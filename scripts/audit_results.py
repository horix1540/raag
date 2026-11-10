"""Milestone 5 audit: re-score every saved run and test the reports' claims.

No model is loaded and nothing is regenerated. Every number here comes from
answers and gate scores already stored in ``data/runs.sqlite3`` (SQuAD 2.0)
and ``reports/generation_hotpot_m4_rows.jsonl`` (HotpotQA), so the audit can
only change how those answers are *scored*, never what the pipeline did.

What it checks, and why each matters for the final report:

  1. Citation scoring. The guarded prompt makes the model cite "[1]", which
     the old scorer turned into an extra token, so every guarded answer scored
     EM 0. Re-scored with citations stripped and with gold containment.
  2. Where abstentions come from. Gate A, the generator's own refusal, or
     Gate B - the reports credit the gates; the split shows how much is the
     abstention-aware prompt.
  3. Uncertainty. Wilson intervals on every rate, and an exact McNemar test on
     the M2 -> M3 hallucination change (same items, so the runs are paired).
  4. Held-out calibration. Thresholds were fitted and evaluated on the same
     150 items; repeated 50/50 splits measure how much of that is optimism.
  5. Hallucination detection proper. On the baseline arm (which always
     answers), does each confidence score rank correct answers above wrong
     ones? AUROC against a 0.5 coin flip, AURC against a random ranking.
  6. HotpotQA. The M4 sweep scored suppressed answers, counted every cited
     answer as a hallucination, and labelled any Gate B failure a
     "contradiction". Re-scored, with abstention sources and Gate B's actual
     separation of correct from incorrect multi-hop answers.

    python scripts/audit_results.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BENCH_DIR, REPORTS_DIR, settings  # noqa: E402
from src.evalstats import (  # noqa: E402
    aurc,
    auroc,
    mcnemar_exact,
    risk_coverage_curve,
    wilson_interval,
)
from src.squad_metrics import (  # noqa: E402
    contains_gold,
    exact_match_score,
    f1_score,
    is_abstention,
)
from src.store import connect  # noqa: E402

SQUAD_RUNS = {"m2_uncalibrated": "squad2-dev", "m3_calibrated": "squad2-v2"}
# Fresh guarded-only runs (M5): identical setup and thresholds, differing only
# in Gate B's premise - v1 whole passages, windows only (rejected: it fails
# claims that summarise a list), and v2 passages plus windows.
GATE_B_RUNS = {"m5_gate_b_v1": "squad2-v5-v1", "m5_gate_b_window_only": "squad2-v5",
               "m5_gate_b_v2": "squad2-v6"}
CAP = settings.confidence.contradiction_cap


def rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "ci95": [round(lo, 4), round(hi, 4)]}


def r4(x: float) -> float | None:
    return None if x != x else round(x, 4)  # NaN -> None for JSON


# --- SQuAD 2.0 ------------------------------------------------------------------


def load_squad(tag: str, labels: dict) -> dict[str, list[dict]]:
    with connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM queries WHERE run_tag = ? ORDER BY id", (tag,))]
    arms: dict[str, list[dict]] = {"guarded": [], "baseline": []}
    for r in rows:
        item = labels[r["item_id"]]
        r["answerable"] = bool(item["answerable"])
        r["golds"] = list(item["answers"])
        r["abstained"] = is_abstention(r["answer"]) or r["verdict"].startswith("abstain")
        r["em_old"] = _em_without_citation_fix(r["answer"], r["golds"])
        r["em"] = exact_match_score(r["answer"], r["golds"])
        r["f1"] = f1_score(r["answer"], r["golds"])
        r["contains"] = contains_gold(r["answer"], r["golds"])
        arms[r["arm"]].append(r)
    return arms


def _em_without_citation_fix(pred: str, golds: list[str]) -> float:
    """The scorer as it was when M2-M4 were written, for the before/after table."""
    from src.squad_metrics import normalize_answer

    p = "" if is_abstention(pred) else normalize_answer(pred)
    if not golds:
        return float(p == "")
    return float(bool(p) and any(p == normalize_answer(g) for g in golds))


def abstention_sources(rows: list[dict]) -> dict:
    """Split guarded abstentions by which mechanism produced them."""
    out = {}
    for name, cond in (("answerable", True), ("unanswerable", False)):
        sub = [r for r in rows if r["answerable"] == cond]
        gate_a = sum(r["verdict"] == "abstain_no_context" and not r["self_abstain"] for r in sub)
        self_ref = sum(bool(r["self_abstain"]) for r in sub)
        suppressed = [r for r in sub if r["verdict"] == "abstain_ungrounded"]
        out[name] = {"n": len(sub), "gate_a": gate_a, "generator_self_refusal": self_ref,
                     "gate_b": len(suppressed),
                     "answered": len(sub) - gate_a - self_ref - len(suppressed)}
        if cond:
            # The telemetry stores the abstain message, not the suppressed answer,
            # but Gate B's claims are the answer split into sentences, so joining
            # them recovers what the user would have seen.
            texts = [" ".join(c["claim"] for c in json.loads(r["claims_json"] or "[]")) for r in suppressed]
            out[name]["gate_b_suppressed_but_correct"] = sum(
                contains_gold(t, r["golds"]) for t, r in zip(texts, suppressed, strict=True))
            out[name]["gate_b_suppressed_with_contradiction"] = sum(r["n_conflicts"] > 0 for r in suppressed)
    return out


def squad_arm_summary(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered_ans = [r for r in ans if not r["abstained"]]
    n = len(rows)
    return {
        "n": n,
        "hallucination": rate(sum(not r["abstained"] for r in una), len(una)),
        "over_abstention": rate(sum(r["abstained"] for r in ans), len(ans)),
        "em_all_as_reported": r4(sum(r["em_old"] for r in rows) / n),
        "em_all_corrected": r4(sum(r["em"] for r in rows) / n),
        "f1_all_corrected": r4(sum(r["f1"] for r in rows) / n),
        "em_answerable_as_reported": r4(sum(r["em_old"] for r in ans) / len(ans)),
        "em_answerable_corrected": r4(sum(r["em"] for r in ans) / len(ans)),
        "contains_gold_answerable": rate(sum(r["contains"] for r in ans), len(ans)),
        # Of the answerable questions the arm chose to answer, how many were right?
        "selective_accuracy_answered": rate(
            sum(r["contains"] for r in answered_ans), len(answered_ans)),
    }


def paired_hallucination_change(before: list[dict], after: list[dict]) -> dict:
    b = {r["item_id"]: not r["abstained"] for r in before if not r["answerable"]}
    a = {r["item_id"]: not r["abstained"] for r in after if not r["answerable"]}
    only_before = sum(b[i] and not a[i] for i in b)
    only_after = sum(a[i] and not b[i] for i in b)
    return {"hallucinated_only_m2": only_before, "hallucinated_only_m3": only_after,
            "mcnemar_exact_p": r4(mcnemar_exact(only_before, only_after))}


def paired_change(before: list[dict], after: list[dict]) -> dict:
    """Item-paired answered/abstained flips between two guarded runs, per condition."""
    out = {}
    b = {r["item_id"]: r for r in before}
    for key, cond in (("unanswerable_answered", False), ("answerable_answered", True)):
        ids = [r["item_id"] for r in after if r["answerable"] == cond and r["item_id"] in b]
        a_ans = {r["item_id"]: not r["abstained"] for r in after}
        only_before = sum(not b[i]["abstained"] and not a_ans[i] for i in ids)
        only_after = sum(a_ans[i] and b[i]["abstained"] for i in ids)
        out[key] = {"only_before": only_before, "only_after": only_after,
                    "mcnemar_exact_p": r4(mcnemar_exact(only_before, only_after))}
    return out


def detection_quality(baseline: list[dict]) -> dict:
    """Does each confidence signal rank the baseline's right answers above wrong ones?

    The baseline arm answers every item and records every gate score without
    acting on it, so it is the clean test of the scores as hallucination
    detectors. Correct = gold contained (answerable) - an answer to an
    unanswerable question is always wrong.
    """
    correct = [bool(r["contains"]) for r in baseline]
    base_risk = 1 - sum(correct) / len(correct)
    out = {"n": len(baseline), "accuracy": r4(1 - base_risk),
           "aurc_random_ranking": r4(base_risk)}
    oracle = risk_coverage_curve([1.0 if c else 0.0 for c in correct], correct)
    out["aurc_oracle"] = r4(aurc(oracle))
    for key in ("composite", "gate_a_score", "gate_b_score", "nli_faithfulness"):
        scores = [float(r[key] or 0.0) for r in baseline]
        out[key] = {"auroc": r4(auroc(scores, correct)),
                    "aurc": r4(aurc(risk_coverage_curve(scores, correct)))}
    # Gate B on answerable questions only: groundedness vs correctness, with
    # answerability taken out of the picture.
    ans = [r for r in baseline if r["answerable"]]
    out["gate_b_answerable_only_auroc"] = r4(auroc(
        [float(r["nli_faithfulness"] or 0) for r in ans], [bool(r["contains"]) for r in ans]))
    return out


def heldout_calibration(baseline: list[dict], *, splits: int = 200, seed: int = 20261102) -> dict:
    """Refit the Gate A threshold on half the items, score it on the other half.

    Uses the same grid and the same criterion (Youden's J) as calibrate.py.
    The in-sample J the report quotes is compared against the distribution of
    held-out J values over repeated splits.
    """
    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    scores = [float(r["rerank_relevancy"]) for r in baseline]
    labels = [r["answerable"] for r in baseline]
    idx = list(range(len(baseline)))

    def youden(ix: list[int], t: float) -> float:
        tp = sum(scores[i] >= t and labels[i] for i in ix)
        fp = sum(scores[i] >= t and not labels[i] for i in ix)
        pos = sum(labels[i] for i in ix)
        neg = len(ix) - pos
        return (tp / pos if pos else 0) - (fp / neg if neg else 0)

    rng = random.Random(seed)
    test_j, chosen, test_auc = [], [], []
    for _ in range(splits):
        rng.shuffle(idx)
        train, test = idx[: len(idx) // 2], idx[len(idx) // 2:]
        t = max(grid, key=lambda g: youden(train, g))
        chosen.append(t)
        test_j.append(youden(test, t))
        test_auc.append(auroc([scores[i] for i in test], [labels[i] for i in test]))
    test_j.sort()
    test_auc.sort()
    in_sample_t = max(grid, key=lambda g: youden(list(range(len(baseline))), g))
    return {
        "splits": splits,
        "in_sample_threshold": in_sample_t,
        "in_sample_youden_j": r4(youden(list(range(len(baseline))), in_sample_t)),
        "heldout_youden_j_median": r4(test_j[splits // 2]),
        "heldout_youden_j_p5_p95": [r4(test_j[int(0.05 * splits)]), r4(test_j[int(0.95 * splits)])],
        "heldout_auroc_median": r4(test_auc[splits // 2]),
        "heldout_auroc_p5_p95": [r4(test_auc[int(0.05 * splits)]), r4(test_auc[int(0.95 * splits)])],
        "threshold_chosen_on_train_halves": {str(t): chosen.count(t) for t in sorted(set(chosen))},
    }


# --- HotpotQA -------------------------------------------------------------------


def hotpot_audit(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = {}
    for gran in ("sentence", "chunk"):
        for arm in ("baseline", "guarded"):
            sub = [r for r in rows if r["granularity"] == gran and r["arm"] == arm]
            n = len(sub)
            # What the user was shown: suppressed answers score as abstentions.
            shown = ["" if r["abstained"] else r["answer"] for r in sub]
            correct = [contains_gold(s, r["gold_answers"]) for s, r in zip(shown, sub, strict=True)]
            answered = [not r["abstained"] for r in sub]
            gate_a_fail = sum(r["gate_a_passed"] is False for r in sub)
            self_ref = sum("INSUFFICIENT_CONTEXT" in (r["answer"] or "") for r in sub)
            gate_b_fail = sum(r["gate_b_passed"] is False for r in sub)
            # The rows' "nli_faithfulness" is Gate B's *post-cap* score. A value of
            # exactly the cap means the contradiction veto fired; a value below the
            # cap may or may not have been vetoed, so this is a lower bound.
            veto_lb = sum(r["nli_faithfulness"] is not None and abs(r["nli_faithfulness"] - CAP) < 1e-9
                          for r in sub)
            entry = {
                "n": n,
                "as_reported": {"f1": r4(sum(r["f1"] for r in sub) / n),
                                "hallucination_rate": r4(sum(r["hallucination"] for r in sub) / n),
                                "contradiction_rate": r4(sum(r["contradiction"] for r in sub) / n)},
                "suppressed_answers_scored_in_f1": sum(r["abstained"] and r["f1"] > 0 for r in sub),
                "f1_shown_corrected": r4(sum(f1_score(s, r["gold_answers"])
                                             for s, r in zip(shown, sub, strict=True)) / n),
                "em_shown_corrected": r4(sum(exact_match_score(s, r["gold_answers"])
                                             for s, r in zip(shown, sub, strict=True)) / n),
                "accuracy_contains_gold": rate(sum(correct), n),
                "answered": sum(answered),
                "wrong_among_answered": rate(sum(a and not c for a, c in zip(answered, correct, strict=True)),
                                             sum(answered)),
                "gate_b_failed": rate(gate_b_fail, n),
                "contradiction_veto_lower_bound": rate(veto_lb, n),
            }
            if arm == "guarded":
                entry["abstention_sources"] = {"gate_a": gate_a_fail, "generator_self_refusal": self_ref,
                                               "gate_b": gate_b_fail}
            else:
                # Gate B as a detector on multi-hop answers: does it fail the wrong
                # ones more often than the right ones?
                raw_correct = [contains_gold(r["answer"], r["gold_answers"]) for r in sub]
                good = [r for r, c in zip(sub, raw_correct, strict=True) if c]
                bad = [r for r, c in zip(sub, raw_correct, strict=True) if not c]
                entry["gate_b_fail_on_correct"] = rate(sum(r["gate_b_passed"] is False for r in good), len(good))
                entry["gate_b_fail_on_incorrect"] = rate(sum(r["gate_b_passed"] is False for r in bad), len(bad))
                entry["gate_b_auroc_correct_vs_incorrect"] = r4(auroc(
                    [float(r["nli_faithfulness"] or 0) for r in sub], raw_correct))
                entry["gate_a_pass_rate"] = rate(sum(r["gate_a_passed"] is True for r in sub), n)
            out[f"{gran}_{arm}"] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=BENCH_DIR / "squad2_dev_150.jsonl")
    ap.add_argument("--hotpot-rows", type=Path, default=REPORTS_DIR / "generation_hotpot_m4_rows.jsonl")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "audit_m5.json")
    args = ap.parse_args()

    labels = {json.loads(line)["item_id"]: json.loads(line)
              for line in args.bench.read_text(encoding="utf-8").splitlines() if line.strip()}

    report: dict = {"squad": {}, "hotpot": None}
    runs = {name: load_squad(tag, labels) for name, tag in SQUAD_RUNS.items()}
    for name, arms in runs.items():
        report["squad"][name] = {
            "guarded": squad_arm_summary(arms["guarded"]),
            "baseline": squad_arm_summary(arms["baseline"]),
            "guarded_abstention_sources": abstention_sources(arms["guarded"]),
        }
    report["squad"]["m2_to_m3_hallucination_change"] = paired_hallucination_change(
        runs["m2_uncalibrated"]["guarded"], runs["m3_calibrated"]["guarded"])
    report["squad"]["detection_quality_baseline_arm"] = detection_quality(runs["m3_calibrated"]["baseline"])
    report["squad"]["heldout_gate_a_calibration"] = heldout_calibration(runs["m3_calibrated"]["baseline"])
    fresh = {name: load_squad(tag, labels)["guarded"] for name, tag in GATE_B_RUNS.items()}
    if all(fresh.values()):
        for name, rows in fresh.items():
            report["squad"][name] = {"guarded": squad_arm_summary(rows),
                                     "guarded_abstention_sources": abstention_sources(rows)}
        report["squad"]["m5_gate_b_v1_to_v2"] = paired_change(
            fresh["m5_gate_b_v1"], fresh["m5_gate_b_v2"])
    report["hotpot"] = hotpot_audit(args.hotpot_rows)

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
