"""Milestone 1 smoke test: prove every component actually runs.

Runs in stages and reports each independently, because on day one some pieces
exist and others do not. Retrieval and both scorers work with no LLM at all, so
this script demonstrates a working confidence layer before Ollama is installed.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --with-llm     # also exercise generation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PASS, FAIL, SKIP = "[PASS]", "[FAIL]", "[SKIP]"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"  {status} {name}" + (f" - {detail}" if detail else ""), flush=True)


def stage(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def check_environment() -> None:
    stage("Stage 1 - environment")
    record("python", PASS, sys.version.split()[0])
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        record("torch", PASS, f"{torch.__version__} -> {dev} ({name})")
    except Exception as exc:  # noqa: BLE001
        record("torch", FAIL, str(exc)[:80])

    for mod in ("chromadb", "sentence_transformers", "langchain", "deepeval", "streamlit"):
        try:
            m = __import__(mod)
            record(mod, PASS, getattr(m, "__version__", "?"))
        except Exception as exc:  # noqa: BLE001
            record(mod, FAIL, str(exc)[:80])


def check_ingestion():
    stage("Stage 2 - ingestion and indexing")
    from src.ingest import build_index

    t0 = time.perf_counter()
    store = build_index(reset=True)
    n = store._collection.count()
    if n == 0:
        record("build_index", FAIL, "index is empty; is data/raw populated?")
        return None
    record("build_index", PASS, f"{n} chunks in {time.perf_counter() - t0:.1f}s")
    return store


def check_retrieval(store):
    stage("Stage 3 - retrieval")
    from src.retriever import retrieve

    covered = "What are the five stages of a standard RAG pipeline?"
    uncovered = "What is the annual rainfall in Ecuador's cloud forests?"

    hit = retrieve(covered, store=store)
    miss = retrieve(uncovered, store=store)

    if not hit.chunks:
        record("retrieve", FAIL, "no chunks returned")
        return None, None
    record("retrieve(covered)", PASS, f"top_sim={hit.top_similarity:.3f} k={len(hit.chunks)}")
    record("retrieve(uncovered)", PASS, f"top_sim={miss.top_similarity:.3f}")

    # Not an assertion: MiniLM cosine is uncalibrated, which is exactly why
    # Gate A uses a reranker instead. Recorded so the gap is visible.
    record(
        "similarity separates",
        PASS if hit.top_similarity > miss.top_similarity else SKIP,
        f"delta={hit.top_similarity - miss.top_similarity:+.3f}",
    )
    return hit, miss


def check_gate_a(hit, miss):
    stage("Stage 4 - Gate A (retrieval sufficiency, no LLM needed)")
    from src.confidence import get_evaluator

    ev = get_evaluator()
    t0 = time.perf_counter()
    a_hit = ev.gate_a(hit)
    record(
        "gate_a(covered)",
        PASS if a_hit.passed else FAIL,
        f"score={a_hit.score:.3f} passed={a_hit.passed} ({a_hit.latency_ms:.0f}ms)",
    )
    a_miss = ev.gate_a(miss)
    record(
        "gate_a(uncovered)",
        PASS if not a_miss.passed else FAIL,
        f"score={a_miss.score:.3f} passed={a_miss.passed} - should abstain",
    )
    record("reranker warm+2 queries", PASS, f"{time.perf_counter() - t0:.1f}s total")
    return a_hit, a_miss


def check_gate_b(hit):
    stage("Stage 5 - Gate B (groundedness via NLI, no LLM needed)")
    from src.confidence import get_evaluator

    ev = get_evaluator()
    contexts = hit.contexts

    grounded = (
        "A standard RAG pipeline has five stages: ingestion, chunking, embedding, "
        "retrieval and generation."
    )
    fabricated = (
        "A standard RAG pipeline has eleven stages and was invented at Stanford in 2015 "
        "by a team of forty researchers."
    )

    t0 = time.perf_counter()
    b_good = ev.gate_b("What are the stages of a RAG pipeline?", grounded, contexts)
    record(
        "gate_b(grounded)",
        # Must clear the gate, not merely score above zero: a check this weak
        # let a Gate B change that rejected this exact answer (0.004) pass.
        PASS if b_good.passed else FAIL,
        f"score={b_good.score:.3f} passed={b_good.passed} claims={len(b_good.claims)}",
    )

    b_bad = ev.gate_b("What are the stages of a RAG pipeline?", fabricated, contexts)
    record(
        "gate_b(fabricated)",
        PASS if b_bad.score < b_good.score else FAIL,
        f"score={b_bad.score:.3f} passed={b_bad.passed} taxonomy={b_bad.taxonomy_counts}",
    )
    record(
        "separation grounded>fabricated",
        PASS if b_good.score > b_bad.score else FAIL,
        f"{b_good.score:.3f} vs {b_bad.score:.3f}",
    )
    record("NLI model load + 2 evals", PASS, f"{time.perf_counter() - t0:.1f}s total")
    return b_good, b_bad


def check_composite(a_hit, b_good, b_bad):
    stage("Stage 6 - composite scoring and decision")
    from src.confidence import get_evaluator

    ev = get_evaluator()
    good = ev.decide(a_hit, b_good)
    bad = ev.decide(a_hit, b_bad)
    record(
        "decide(grounded)",
        PASS if not good.abstained else FAIL,
        f"{good.verdict} composite={good.composite:.3f}",
    )
    record(
        "decide(fabricated)",
        PASS if bad.abstained else FAIL,
        f"{bad.verdict} composite={bad.composite:.3f}",
    )
    record(
        "fabricated scores lower",
        PASS if bad.composite < good.composite else FAIL,
        f"{bad.composite:.3f} < {good.composite:.3f}",
    )


def check_telemetry():
    stage("Stage 7 - telemetry store")
    from src.store import init_db, load_dataframe, log_query, summary_stats

    try:
        init_db()
        rid = log_query(
            {
                "question": "smoke test probe",
                "answer": "n/a",
                "verdict": "abstain_no_context",
                "composite": 0.11,
                "gate_a_score": 0.11,
                "gate_a_passed": False,
                "sources": ["rag_course_notes.md"],
                "claims_json": [],
                "run_tag": "smoke",
            }
        )
        df = load_dataframe(limit=5)
        stats = summary_stats()
        record("log_query", PASS, f"row id={rid}")
        record("load_dataframe", PASS, f"{len(df)} rows, {len(df.columns)} cols")
        record("summary_stats", PASS, f"total={stats['total']} abstain_rate={stats['abstention_rate']:.2f}")
    except Exception as exc:  # noqa: BLE001
        record("telemetry", FAIL, f"{type(exc).__name__}: {exc}")


def check_llm():
    stage("Stage 8 - LLM provider (optional)")
    from src.llm import health_check

    ok, msg = health_check()
    record("provider reachable", PASS if ok else SKIP, msg[:110])
    return ok


def check_full_pipeline(store):
    stage("Stage 9 - full pipeline end to end (needs an LLM)")
    from src.pipeline import answer_question

    for q, note in [
        ("What are the five stages of a standard RAG pipeline?", "should answer"),
        ("What is the average rainfall in the Amazon basin?", "should abstain"),
    ]:
        try:
            r = answer_question(q, store=store, run_tag="smoke")
            record(
                f"pipeline({note})",
                PASS,
                f"verdict={r.verdict} composite={r.confidence.composite:.3f} "
                f"{r.total_ms:.0f}ms",
            )
            print(f"        Q: {q}")
            print(f"        A: {r.displayed_answer[:150]}")
        except Exception as exc:  # noqa: BLE001
            record(f"pipeline({note})", FAIL, f"{type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true", help="also exercise generation")
    args = ap.parse_args()

    print("Confidence-Aware RAG - Milestone 1 smoke test")
    check_environment()
    store = check_ingestion()
    if store is None:
        return 1

    hit, miss = check_retrieval(store)
    if hit is None:
        return 1

    a_hit, _ = check_gate_a(hit, miss)
    b_good, b_bad = check_gate_b(hit)
    check_composite(a_hit, b_good, b_bad)
    check_telemetry()

    if args.with_llm:
        if check_llm():
            check_full_pipeline(store)
        else:
            record("full pipeline", SKIP, "no reachable LLM provider")

    stage("Summary")
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\n  Failures:")
        for name, s, detail in results:
            if s == FAIL:
                print(f"    - {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())




