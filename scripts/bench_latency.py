"""Measure warm per-query latency of the confidence layer.

The smoke test's timings include one-off model downloads and are therefore
useless as a latency budget. This script loads every model once, discards a
warm-up pass, and then reports steady-state cost per query for retrieval,
Gate A and Gate B. No LLM is required: Gate B is measured against a fixed
answer drawn from the corpus, so the number isolates scoring cost from
generation cost.

    .venv/Scripts/python.exe scripts/bench_latency.py [-n 10]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence import get_evaluator  # noqa: E402
from src.config import REPORTS_DIR, settings  # noqa: E402
from src.retriever import get_store, retrieve  # noqa: E402
from src.scorers import get_nli_scorer, get_rerank_scorer  # noqa: E402

QUERIES = [
    "What are the five stages of a standard RAG pipeline?",
    "Why does RAG still hallucinate even with retrieval?",
    "What is the difference between baseless information and evident conflict?",
    "How is faithfulness measured in a RAG system?",
    "What does selective prediction mean?",
]

# A claim the sample corpus supports, so Gate B has real work to do without
# waiting on a generator.
ANSWER = (
    "A standard RAG pipeline has five stages: ingestion, chunking, embedding, "
    "retrieval and generation. Faithfulness can only be measured after an answer "
    "exists, because it compares the answer against the retrieved passages."
)


def stats(samples: list[float]) -> dict[str, float]:
    return {
        "n": len(samples),
        "mean_ms": round(statistics.fmean(samples), 1),
        "median_ms": round(statistics.median(samples), 1),
        "min_ms": round(min(samples), 1),
        "max_ms": round(max(samples), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="timed iterations per stage")
    args = ap.parse_args()

    store = get_store()
    if store._collection.count() == 0:
        print("Index is empty. Run: python scripts/build_index.py --reset")
        return 1

    evaluator = get_evaluator()

    print("Warming models (load + first forward pass, not timed)...")
    t0 = time.perf_counter()
    get_rerank_scorer()
    get_nli_scorer()
    warm = retrieve(QUERIES[0], store=store)
    evaluator.gate_a(warm)
    evaluator.gate_b(QUERIES[0], ANSWER, warm.contexts)
    print(f"  warm-up took {time.perf_counter() - t0:.1f}s\n")

    timings: dict[str, list[float]] = {"retrieval": [], "gate_a": [], "gate_b": []}
    for i in range(args.n):
        query = QUERIES[i % len(QUERIES)]

        t = time.perf_counter()
        result = retrieve(query, store=store)
        timings["retrieval"].append((time.perf_counter() - t) * 1000)

        timings["gate_a"].append(evaluator.gate_a(result).latency_ms)
        timings["gate_b"].append(evaluator.gate_b(query, ANSWER, result.contexts).latency_ms)

    report = {
        "embedding_device": settings.retrieval.embedding_device,
        "scorer_device": settings.confidence.nli_device,
        "scorer": settings.confidence.scorer,
        "top_k": settings.retrieval.top_k,
        "stages": {k: stats(v) for k, v in timings.items()},
    }
    report["stages"]["confidence_total"] = stats(
        [a + b for a, b in zip(timings["gate_a"], timings["gate_b"], strict=True)]
    )

    print(f"{'stage':<20}{'median':>10}{'mean':>10}{'min':>10}{'max':>10}")
    for name, s in report["stages"].items():
        print(f"{name:<20}{s['median_ms']:>10.1f}{s['mean_ms']:>10.1f}"
              f"{s['min_ms']:>10.1f}{s['max_ms']:>10.1f}")

    out = REPORTS_DIR / "latency.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
