"""Full RAG pipeline sweep: retrieval + generation + gates on HotpotQA.

This is the Milestone 4 end-to-end test. The retrieval-only ablation
(``run_retrieval_hotpot.py``) measured how much chunking helps raw recall;
this script answers whether that recall advantage survives Gate A's reranker
and whether the padded text in 800-character chunks triggers Gate B's NLI
contradiction detector more often than clean sentence-level units.

Two arms per granularity:

  baseline - gates disabled, every question gets an answer. The EM/F1 numbers
      are the ceiling: what the generator can produce when handed the
      retrieved context uncensored.
  guarded  - gates active. Abstentions and contradictions are scored as
      failures, so the metrics reflect the full confidence-aware pipeline's
      end-to-end accuracy and conservatism.

Metrics tracked:

  EM / F1           - standard QA metrics on the answer the user was shown
  accuracy_contains_gold - the gold answer appears in the shown answer
  hallucination_rate - answers that passed the gates but are wrong
  over_abstention   - abstentions where retrieval had surfaced a gold unit
  gate_b_failure_rate - generated answers scoring under Gate B's threshold
  gate_b_contradiction_rate - generated answers with a contradicted claim
  answer_rate       - fraction of items that produced an answer (not abstained)

Every item gets a fresh isolated collection (same teardown discipline as the
retrieval script) so sentence units cannot leak into the chunk-granularity
index. The generator is ``llama3.1:8b`` via Ollama, matching the baseline
control arm from Milestone 2 and 3.

    python scripts/run_generation_hotpot.py
    python scripts/run_generation_hotpot.py --limit 10  # smoke run
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BENCH_DIR, REPORTS_DIR, settings  # noqa: E402
from src.ingest import index_texts  # noqa: E402
from src.pipeline import answer_question  # noqa: E402
from src.squad_metrics import contains_gold, exact_match_score, f1_score  # noqa: E402

GRANULARITIES = ("sentence", "chunk")
ARMS = ("baseline", "guarded")


class CollectionLeakError(RuntimeError):
    """Raised when a benchmark collection cannot be torn down or verified.

    See ``run_retrieval_hotpot.py`` for the rationale: a leaked collection
    contaminates the comparison this script exists to make.
    """


def load_items(path: Path) -> list[dict]:
    items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise SystemExit(f"No items found in {path}")
    return items


def collection_name(item_id: str, granularity: str) -> str:
    """Chroma-safe collection name with granularity baked in."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", item_id)[:40]
    return f"hp_{granularity[:4]}_{safe}"


def teardown(store, name: str, *, must_exist: bool) -> None:
    """Delete a collection, loudly."""
    try:
        store.delete_collection()
    except ValueError:
        if must_exist:
            raise CollectionLeakError(
                f"collection {name!r} vanished before teardown; the index may be shared"
            ) from None
    except Exception as exc:  # noqa: BLE001
        raise CollectionLeakError(f"could not tear down collection {name!r}: {exc}") from exc


def build_clean(units: list[str], name: str, source: str):
    """Build a collection and assert it holds exactly ``len(units)`` vectors."""
    store = index_texts(units, collection=name, source=source)
    count = store._collection.count()
    if count != len(units):
        teardown(store, name, must_exist=True)
        raise CollectionLeakError(
            f"collection {name!r} holds {count} vectors, expected {len(units)}: "
            "a previous collection with this name was not torn down"
        )
    return store


def run_one_item(
    item: dict,
    granularity: str,
    arm: str,
    *,
    llm_model: str,
) -> dict:
    """Run the full pipeline on one item under one configuration.

    Returns a scored result dict with the answer, gates' decisions, and metrics.
    """
    view_key = f"{granularity}_view"
    view = item[view_key]
    units = view["units"]
    name = collection_name(item["item_id"], granularity)

    store = build_clean(units, name, f"hotpot:{granularity}")
    try:
        # Run the pipeline with gates on/off
        baseline = arm == "baseline"
        result = answer_question(
            item["question"],
            store=store,
            collection=name,
            baseline=baseline,
            # Tagged so the dashboard can tell sweep rows from app questions;
            # the M4 sweep logged without one.
            run_tag=f"hotpot-{granularity}",
            item_id=item["item_id"],
        )

        # Extract telemetry
        answer = result.answer or ""
        gold_answers = item["answers"]
        abstained = result.confidence.abstained
        gate_a_passed = result.confidence.gate_a.passed if result.confidence.gate_a else None
        gate_b_passed = result.confidence.gate_b.passed if result.confidence.gate_b else None
        gate_b = result.confidence.gate_b
        # Two different events, kept apart: Gate B failing (score under the
        # threshold, usually low entailment) and the contradiction veto firing
        # (a claim the NLI head says the passages refute). Milestone 4 reported
        # the first under the name of the second.
        gate_b_failed = bool(gate_b and gate_b.score < settings.confidence.gate_b_threshold)
        contradiction = bool(gate_b and gate_b.conflicts)

        # Score what the user was shown. A Gate B abstention keeps the raw
        # answer in ``result.answer`` for inspection, and scoring that gave the
        # guarded arm credit for answers it suppressed.
        shown = "" if abstained else answer
        em = exact_match_score(shown, gold_answers)
        f1 = f1_score(shown, gold_answers)
        correct = contains_gold(shown, gold_answers)

        # Hallucination: answered but wrong. Gold containment rather than EM:
        # a sentence-length answer always fails EM, which counted every
        # answered guarded item as a hallucination.
        hallucination = not abstained and not correct

        # Over-abstention: Gate A refused even though retrieval surfaced gold units
        # (for this we need to check if any retrieved unit is gold)
        retrieved_indices = [
            int(chunk.metadata.get("chunk_index", -1))
            for chunk in result.retrieval.chunks
            if "chunk_index" in chunk.metadata
        ]
        gold_units = set(view["gold_units"])
        retrieval_surfaced_gold = bool(set(retrieved_indices) & gold_units)
        over_abstention = abstained and retrieval_surfaced_gold

        return {
            "item_id": item["item_id"],
            "granularity": granularity,
            "arm": arm,
            "question": item["question"],
            "gold_answers": gold_answers,
            "answer": answer,
            "abstained": abstained,
            "gate_a_passed": gate_a_passed,
            "gate_b_passed": gate_b_passed,
            "exact_match": em,
            "f1": round(f1, 4),
            "correct": correct,
            "self_abstained": bool(result.generation and result.generation.self_reported_insufficient),
            "hallucination": hallucination,
            "over_abstention": over_abstention,
            "gate_b_failed": gate_b_failed,
            "contradiction": contradiction,
            "rerank_relevancy": result.confidence.gate_a.score if result.confidence.gate_a else None,
            # Raw mean entailment, and the post-veto score the gate decided on.
            "nli_faithfulness": gate_b.nli_faithfulness if gate_b else None,
            "gate_b_score": gate_b.score if gate_b else None,
            "composite_score": result.confidence.composite,
            "n_retrieved": len(result.retrieval.chunks),
            "n_gold_retrieved": len(set(retrieved_indices) & gold_units),
        }
    finally:
        teardown(store, name, must_exist=True)


def run_sweep(
    items: list[dict],
    granularity: str,
    arm: str,
    *,
    llm_model: str,
) -> list[dict]:
    """Sweep all items under one granularity and arm configuration."""
    rows: list[dict] = []
    t_start = time.perf_counter()

    for n, item in enumerate(items, 1):
        try:
            row = run_one_item(item, granularity, arm, llm_model=llm_model)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            # Log the error but continue - one bad item shouldn't kill the sweep
            print(f"    ERROR on item {item['item_id']}: {exc}", flush=True)
            rows.append(
                {
                    "item_id": item["item_id"],
                    "granularity": granularity,
                    "arm": arm,
                    "error": str(exc),
                }
            )

        if n % 10 == 0 or n == len(items):
            done = [r for r in rows if "exact_match" in r]
            mean_em = statistics.mean(r["exact_match"] for r in done) if done else 0.0
            mean_f1 = statistics.mean(r["f1"] for r in done) if done else 0.0
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{granularity:<8} {arm:<8} {n:>3}/{len(items)}] "
                f"EM {mean_em:.3f}  F1 {mean_f1:.3f}  ({elapsed:.0f}s)",
                flush=True,
            )

    return rows


def aggregate(rows: list[dict]) -> dict:
    """Compute summary statistics for one configuration."""
    valid = [r for r in rows if "exact_match" in r]
    if not valid:
        return {"n": 0}

    answered = [r for r in valid if not r["abstained"]]
    return {
        "n": len(valid),
        "exact_match": round(statistics.mean(r["exact_match"] for r in valid), 4),
        "f1": round(statistics.mean(r["f1"] for r in valid), 4),
        "accuracy_contains_gold": round(statistics.mean(r["correct"] for r in valid), 4),
        "risk_among_answered": round(
            sum(not r["correct"] for r in answered) / len(answered), 4
        ) if answered else None,
        "answer_rate": round(len(answered) / len(valid), 4),
        "hallucination_rate": round(
            sum(r["hallucination"] for r in valid) / len(valid), 4
        ),
        "over_abstention_rate": round(
            sum(r["over_abstention"] for r in valid) / len(valid), 4
        ),
        "gate_b_failure_rate": round(
            sum(r["gate_b_failed"] for r in valid) / len(valid), 4
        ),
        "gate_b_contradiction_rate": round(
            sum(r["contradiction"] for r in valid) / len(valid), 4
        ),
        "self_abstention_rate": round(
            sum(r["self_abstained"] for r in valid) / len(valid), 4
        ),
        "mean_rerank_relevancy": round(
            statistics.mean(
                r["rerank_relevancy"] for r in valid if r.get("rerank_relevancy") is not None
            ),
            4,
        ) if any(r.get("rerank_relevancy") is not None for r in valid) else None,
        "mean_nli_faithfulness": round(
            statistics.mean(
                r["nli_faithfulness"] for r in valid if r.get("nli_faithfulness") is not None
            ),
            4,
        ) if any(r.get("nli_faithfulness") is not None for r in valid) else None,
    }


def print_table(report: dict) -> None:
    """Print the comparative table across granularities and arms."""
    print("\n" + "=" * 90)
    print(f"Generation sweep, n={report['n_items']} HotpotQA items, model={report['llm_model']}")
    print("=" * 90)

    configs = []
    for g in GRANULARITIES:
        for a in ARMS:
            key = f"{g}_{a}"
            if key in report:
                configs.append((g, a, key))

    if not configs:
        print("No results to display.")
        return

    header = f"{'Metric':<34}" + "".join(f"{g[:4]}.{a[:4]:>8}" for g, a, _ in configs)
    print(header)
    print("-" * len(header))

    def row(label: str, key: str, fmt: str = "{:.4f}") -> None:
        cells = ""
        for _, _, config_key in configs:
            val = report[config_key].get(key)
            if val is None:
                cells += f"{'--':>13}"
            elif isinstance(val, (int, float)):
                cells += f"{fmt.format(val):>13}"
            else:
                cells += f"{str(val):>13}"
        print(f"{label:<34}{cells}")

    row("n items", "n", "{:.0f}")
    row("exact_match", "exact_match")
    row("f1", "f1")
    row("answer_rate", "answer_rate")
    row("hallucination_rate", "hallucination_rate")
    row("over_abstention_rate", "over_abstention_rate")
    row("accuracy_contains_gold", "accuracy_contains_gold")
    row("risk_among_answered", "risk_among_answered")
    row("gate_b_failure_rate", "gate_b_failure_rate")
    row("gate_b_contradiction_rate", "gate_b_contradiction_rate")
    row("self_abstention_rate", "self_abstention_rate")
    row("mean_rerank_relevancy", "mean_rerank_relevancy")
    row("mean_nli_faithfulness", "mean_nli_faithfulness")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=BENCH_DIR / "hotpot_dev_150.jsonl")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "generation_hotpot_m4.json")
    ap.add_argument("--rows-out", type=Path, default=REPORTS_DIR / "generation_hotpot_m4_rows.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items")
    ap.add_argument(
        "--llm-model",
        default="llama3.1:8b",
        help="LLM model name (default llama3.1:8b)",
    )
    ap.add_argument(
        "--granularity",
        choices=("both", *GRANULARITIES),
        default="both",
        help="run one view only",
    )
    ap.add_argument(
        "--arm",
        choices=("both", *ARMS),
        default="both",
        help="run one arm only",
    )
    args = ap.parse_args()

    items = load_items(args.bench)
    if args.limit:
        items = items[: args.limit]

    views = list(GRANULARITIES) if args.granularity == "both" else [args.granularity]
    arms = list(ARMS) if args.arm == "both" else [args.arm]

    print(f"Generation sweep: {len(items)} items, model={args.llm_model}")
    print(f"  granularities: {views}")
    print(f"  arms: {arms}")
    print(f"  embeddings: {settings.retrieval.embedding_model}")
    print(f"  gate_a_threshold: {settings.confidence.gate_a_threshold}")
    print(f"  gate_b_threshold: {settings.confidence.gate_b_threshold}\n")

    all_rows: list[dict] = []
    report: dict = {
        "bench": str(args.bench),
        "n_items": len(items),
        "llm_model": args.llm_model,
    }

    for granularity in views:
        for arm in arms:
            t0 = time.perf_counter()
            rows = run_sweep(items, granularity, arm, llm_model=args.llm_model)
            all_rows.extend(rows)
            config_key = f"{granularity}_{arm}"
            report[config_key] = aggregate(rows)
            report[config_key]["sweep_seconds"] = round(time.perf_counter() - t0, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.rows_out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_table(report)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.rows_out} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
