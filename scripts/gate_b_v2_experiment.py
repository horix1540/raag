"""Gate B v2: does a question-conditioned NLI hypothesis fix over-abstention?

The Milestone 5 audit found that Gate B v1 rejected 35 of 36 *correct* answers
it suppressed on the M3 SQuAD run, because a bare answer ("New Orleans [1]")
is not a statement NLI can entail. v2 rewrites the hypothesis (see
``scorers.build_hypothesis``). This script measures the effect offline.

No LLM is called. The generated answers are the ones stored by the M3 run, so
the generator's behaviour is held fixed and only Gate B changes. Each answer is
re-scored through the production ``ConfidenceEvaluator.gate_b`` path under
each configuration:

  v1        - bare claim vs whole passage (must reproduce the stored scores)
  qa_short  - question + claim for claims of <= 8 words, whole passage
  qa        - question + every claim, whole passage
  window    - bare claim vs overlapping 3-sentence windows of each passage
  both      - bare claim vs passages and windows together (Gate B v2)
  qa_short+window, qa+window - both changes together

Measured:

  1. Reproduction. v1 re-scored must match the stored Gate B scores, or every
     comparison below is between different setups.
  2. Detection. On the baseline arm (always answers, gates measured only),
     AUROC of the Gate B score for correct vs incorrect answers.
  3. Guarded outcome. The M3 guarded items that reached Gate B are re-decided;
     Gate A and the generator's self-refusals are unchanged. Reported (a) at
     the existing threshold 0.25 with no refit, and (b) with the threshold
     cross-fitted: chosen on one half of the items (Youden J on the baseline
     arm's correctness labels), applied to the other half, and swapped.
  4. HotpotQA. The same detection test on the M4 baseline answers, with the
     passages rebuilt by the same retrieval. Multi-hop is the case v2 is *not*
     expected to fix; it is here to check that it does not get worse.

    python scripts/gate_b_v2_experiment.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence import get_evaluator, weighted_harmonic_mean  # noqa: E402
from src.config import BENCH_DIR, REPORTS_DIR, settings  # noqa: E402
from src.evalstats import auroc, wilson_interval  # noqa: E402
from src.squad_metrics import contains_gold  # noqa: E402
from src.store import connect  # noqa: E402

# name -> (hypothesis mode, premise mode); "v1" is Gate B as shipped in M1-M4.
CONFIGS = {
    "v1": ("claim", "passage"),
    "qa_short": ("qa_short", "passage"),
    "qa": ("qa", "passage"),
    "window": ("claim", "window"),
    "both": ("claim", "both"),
    "qa_short+window": ("qa_short", "window"),
    "qa+window": ("qa", "window"),
}
MODES = tuple(CONFIGS)  # narrowed by --configs
GRANULARITIES = ("sentence", "chunk")  # narrowed by --hotpot-granularity
RUN_TAG = "squad2-v2"
V1_THRESHOLD = 0.25
GRID = [round(0.05 * i, 2) for i in range(4, 20)]  # 0.20 .. 0.95, never below the cap


def r4(x: float) -> float | None:
    return None if x != x else round(x, 4)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": r4(k / n) if n else None, "ci95": [r4(lo), r4(hi)]}


def gate_b_scores(question: str, answer: str, contexts: list[str]) -> dict[str, dict]:
    """Gate B under every hypothesis mode, through the production code path."""
    ev = get_evaluator()
    out = {}
    cfg = settings.confidence
    original = (cfg.nli_hypothesis, cfg.nli_premise)
    try:
        for name in MODES:
            hyp, prem = CONFIGS[name]
            cfg.nli_hypothesis, cfg.nli_premise = hyp, prem
            gb = ev.gate_b(question, answer, contexts)
            out[name] = {"score": gb.score, "contradiction": bool(gb.conflicts)}
    finally:
        cfg.nli_hypothesis, cfg.nli_premise = original
    return out


def youden_threshold(scores: list[float], correct: list[bool]) -> float:
    def j(t: float) -> float:
        pos = sum(correct)
        neg = len(correct) - pos
        tp = sum(s >= t and c for s, c in zip(scores, correct, strict=True))
        fp = sum(s >= t and not c for s, c in zip(scores, correct, strict=True))
        return (tp / pos if pos else 0) - (fp / neg if neg else 0)

    return max(GRID, key=j)


# --- SQuAD 2.0 ------------------------------------------------------------------


def squad(bench: Path) -> dict:
    items = {}
    for line in bench.read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            items[it["item_id"]] = it

    with connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM queries WHERE run_tag = ? ORDER BY id", (RUN_TAG,))]
    base = {r["item_id"]: r for r in rows if r["arm"] == "baseline"}
    guard = {r["item_id"]: r for r in rows if r["arm"] == "guarded"}

    # SQuAD items carry 5 paragraphs and retrieval keeps top_k=5 from a
    # 5-document index, so the retrieved passages are exactly the item's
    # paragraphs; the max over passages is order-independent.
    assert settings.retrieval.top_k >= 5

    scored_base, scored_guard = {}, {}
    for iid, it in items.items():
        ctx = it["contexts"]
        b = base[iid]
        scored_base[iid] = {
            "answerable": it["answerable"],
            "correct": contains_gold(b["answer"], it["answers"]),
            "gate_a": float(b["gate_a_score"]),
            "stored": b["gate_b_score"],
            "modes": gate_b_scores(it["question"], b["answer"], ctx),
        }
        g = guard[iid]
        if g["verdict"] in ("answer", "abstain_ungrounded"):
            # The telemetry stores the abstain message for suppressed answers;
            # the claims are that answer split into sentences, so rejoin them.
            raw = g["answer"] if g["verdict"] == "answer" else " ".join(
                c["claim"] for c in json.loads(g["claims_json"] or "[]"))
            scored_guard[iid] = {
                "answerable": it["answerable"],
                "correct": contains_gold(raw, it["answers"]),
                "stored": g["gate_b_score"],
                "modes": gate_b_scores(it["question"], raw, ctx),
            }
        print(f"  scored {len(scored_base):>3}/150", end="\r", flush=True)
    print()

    report: dict = {"reproduction": _reproduction(scored_base, scored_guard), "detection": {},
                    "guarded_fixed_threshold": {}, "guarded_cross_fitted": {}}

    ids = sorted(items)
    for mode in MODES:
        sb = [scored_base[i] for i in ids]
        scores = [x["modes"][mode]["score"] for x in sb]
        correct = [x["correct"] for x in sb]
        ans = [x for x in sb if x["answerable"]]
        report["detection"][mode] = {
            "gate_b_auroc_all": r4(auroc(scores, correct)),
            "gate_b_auroc_answerable_only": r4(auroc(
                [x["modes"][mode]["score"] for x in ans], [x["correct"] for x in ans])),
            "composite_auroc_all": r4(auroc(
                [weighted_harmonic_mean([x["gate_a"], x["modes"][mode]["score"]],
                                        settings.confidence.weights) for x in sb], correct)),
            "gate_b_fail_rate_on_correct_at_0.25": rate(
                sum(x["modes"][mode]["score"] < V1_THRESHOLD for x in sb if x["correct"]),
                sum(correct)),
            "gate_b_fail_rate_on_incorrect_at_0.25": rate(
                sum(x["modes"][mode]["score"] < V1_THRESHOLD for x in sb if not x["correct"]),
                len(sb) - sum(correct)),
        }
        report["guarded_fixed_threshold"][mode] = _guarded_outcome(
            scored_guard, {i: V1_THRESHOLD for i in ids}, mode)

        # Cross-fit: threshold chosen on one half, applied to the other.
        rng = random.Random(20261102)
        shuffled = list(ids)
        rng.shuffle(shuffled)
        halves = [set(shuffled[: len(ids) // 2]), set(shuffled[len(ids) // 2:])]
        thr: dict[str, float] = {}
        chosen = []
        for train, test in ((halves[0], halves[1]), (halves[1], halves[0])):
            t = youden_threshold([scored_base[i]["modes"][mode]["score"] for i in sorted(train)],
                                 [scored_base[i]["correct"] for i in sorted(train)])
            chosen.append(t)
            thr.update({i: t for i in test})
        out = _guarded_outcome(scored_guard, thr, mode)
        out["thresholds_per_fold"] = chosen
        report["guarded_cross_fitted"][mode] = out
    return report


def _reproduction(base: dict, guard: dict) -> dict:
    pairs = [(x["stored"], x["modes"]["v1"]["score"]) for x in [*base.values(), *guard.values()]
             if x["stored"] is not None]
    diffs = [abs(a - b) for a, b in pairs]
    return {"n": len(pairs), "within_0.01": sum(d < 0.01 for d in diffs),
            "max_abs_diff": r4(max(diffs)) if diffs else None}


def _guarded_outcome(scored_guard: dict, thr: dict, mode: str) -> dict:
    """Re-decide only the items that reached Gate B; every other item stays abstained."""
    out = {}
    for key, cond in (("answerable", True), ("unanswerable", False)):
        sub = [(i, x) for i, x in scored_guard.items() if x["answerable"] == cond]
        passed = [(i, x) for i, x in sub if x["modes"][mode]["score"] >= thr[i]]
        n = 75
        out[key] = {
            "reached_gate_b": len(sub),
            "answered": len(passed),
            "answered_correct": sum(x["correct"] for _, x in passed),
            "suppressed": len(sub) - len(passed),
            "suppressed_but_correct": sum(x["correct"] for i, x in sub
                                         if x["modes"][mode]["score"] < thr[i]),
        }
        if cond:
            out[key]["over_abstention"] = rate(n - len(passed), n)
            out[key]["answered_correct_rate"] = rate(sum(x["correct"] for _, x in passed), n)
        else:
            out[key]["hallucination"] = rate(len(passed), n)
    return out


# --- HotpotQA -------------------------------------------------------------------


def hotpot(bench: Path, rows_path: Path) -> dict:
    from src.ingest import index_texts
    from src.retriever import retrieve

    items = {}
    for line in bench.read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            items[it["item_id"]] = it
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    report = {}
    for gran in GRANULARITIES:
        sub = [r for r in rows if r["granularity"] == gran and r["arm"] == "baseline"]
        scored = []
        for n, r in enumerate(sub, 1):
            it = items[r["item_id"]]
            name = f"gbv2_{gran}_{re.sub(r'[^A-Za-z0-9]', '', r['item_id'])[:40]}"
            store = index_texts(it[f"{gran}_view"]["units"], collection=name, source=f"hotpot:{gran}")
            try:
                ctx = retrieve(it["question"], store=store).contexts
            finally:
                with contextlib.suppress(Exception):
                    store.delete_collection()
            scored.append({"correct": contains_gold(r["answer"], r["gold_answers"]),
                           "stored": r["nli_faithfulness"],
                           "modes": gate_b_scores(it["question"], r["answer"], ctx)})
            print(f"  hotpot {gran} {n:>3}/{len(sub)}", end="\r", flush=True)
        print()
        correct = [x["correct"] for x in scored]
        entry = {"n": len(scored), "reproduction": _reproduction({i: x for i, x in enumerate(scored)}, {})}
        for mode in MODES:
            scores = [x["modes"][mode]["score"] for x in scored]
            entry[mode] = {
                "gate_b_auroc": r4(auroc(scores, correct)),
                "fail_rate_on_correct_at_0.25": rate(
                    sum(s < V1_THRESHOLD for s, c in zip(scores, correct, strict=True) if c), sum(correct)),
                "fail_rate_on_incorrect_at_0.25": rate(
                    sum(s < V1_THRESHOLD for s, c in zip(scores, correct, strict=True) if not c),
                    len(correct) - sum(correct)),
                "contradiction_rate": rate(sum(x["modes"][mode]["contradiction"] for x in scored), len(scored)),
            }
        report[gran] = entry
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=BENCH_DIR / "squad2_dev_150.jsonl")
    ap.add_argument("--hotpot-bench", type=Path, default=BENCH_DIR / "hotpot_dev_150.jsonl")
    ap.add_argument("--hotpot-rows", type=Path, default=REPORTS_DIR / "generation_hotpot_m4_rows.jsonl")
    ap.add_argument("--skip-hotpot", action="store_true")
    ap.add_argument("--skip-squad", action="store_true")
    ap.add_argument("--configs", help="comma-separated subset of configurations, e.g. v1,both")
    ap.add_argument("--hotpot-granularity", choices=("sentence", "chunk"))
    ap.add_argument("--nli-model", help="override the NLI checkpoint, e.g. cross-encoder/nli-deberta-v3-large")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "gate_b_v2.json")
    args = ap.parse_args()

    settings.confidence.scorer = "nli"
    if args.nli_model:
        settings.confidence.nli_model = args.nli_model
    global MODES, GRANULARITIES
    if args.configs:
        MODES = tuple(c.strip() for c in args.configs.split(","))
        unknown = set(MODES) - set(CONFIGS)
        if unknown:
            raise SystemExit(f"unknown configs: {sorted(unknown)}")
    if args.hotpot_granularity:
        GRANULARITIES = (args.hotpot_granularity,)
    report = {} if args.skip_squad else {"squad": squad(args.bench)}
    if not args.skip_hotpot:
        report["hotpot"] = hotpot(args.hotpot_bench, args.hotpot_rows)

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
