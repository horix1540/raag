"""Benchmark harness: run the SQuAD 2.0 sample through both pipeline arms.

For every item in the benchmark file this script builds a throwaway per-item
index (gold paragraph + distractors, exactly as ``build_benchmark.py`` packaged
them), then runs the question through the pipeline twice:

  guarded  - the real system: Gate A can refuse before generation, Gate B
             can suppress the answer afterwards.
  baseline - the control arm: gates are measured but never enforced, which is
             what a conventional RAG pipeline does.

Every run is persisted to telemetry tagged with ``run_tag``, ``arm`` and
``item_id``, so the evaluation scripts can join rows back to the dataset
labels without this script having to compute any metric itself.

The run is resumable: completed ``(item_id, arm)`` pairs under the tag are
skipped on restart, so a crashed or interrupted sweep loses nothing.

    python scripts/run_benchmark.py                       # full file, both arms
    python scripts/run_benchmark.py --limit 5             # smoke run
    python scripts/run_benchmark.py --arm guarded --tag squad2-dev
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BENCH_DIR, settings  # noqa: E402
from src.ingest import index_texts  # noqa: E402
from src.llm import health_check  # noqa: E402
from src.pipeline import answer_question  # noqa: E402
from src.store import connect, init_db  # noqa: E402

ARMS = ("guarded", "baseline")


def load_items(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        raise SystemExit(f"No items found in {path}")
    return items


def completed(tag: str) -> dict[str, set[str]]:
    """(item_id, arm) pairs already persisted under this run tag."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT item_id, arm FROM queries WHERE run_tag = ?", (tag,)
        ).fetchall()
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r["item_id"], set()).add(r["arm"] or "guarded")
    return out


def collection_name(item_id: str) -> str:
    """Chroma collection names are restricted; benchmark ids are hex but stay defensive."""
    return f"bench_{re.sub(r'[^A-Za-z0-9._-]', '-', item_id)[:50]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=BENCH_DIR / "squad2_dev_150.jsonl")
    ap.add_argument("--tag", default="squad2-dev", help="run_tag written to telemetry")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items")
    ap.add_argument(
        "--arm", choices=("both", "guarded", "baseline"), default="both",
        help="which arm(s) to run; both is the default",
    )
    ap.add_argument("--top-k", type=int, default=None, help="override retrieval top_k")
    ap.add_argument(
        "--scorer", choices=("nli", "llm", "both"), default=None,
        help="override RAG_SCORER for this run (e.g. 'both' for the judge-agreement study)",
    )
    args = ap.parse_args()

    if args.scorer:
        settings.confidence.scorer = args.scorer

    ok, msg = health_check()
    if not ok:
        print(f"LLM provider not reachable: {msg}\nStart Ollama and pull the model first.")
        return 1

    items = load_items(args.bench)
    if args.limit:
        items = items[: args.limit]
    arms = list(ARMS) if args.arm == "both" else [args.arm]
    done = completed(args.tag)

    print(f"Benchmark: {len(items)} items | arms={arms} | tag={args.tag!r}")
    print(
        f"provider={settings.llm.provider} model={settings.llm.generator_model} "
        f"scorer={settings.confidence.scorer}"
    )

    n_run, n_skip, n_err = 0, 0, 0
    t_start = time.perf_counter()

    for n, item in enumerate(items, 1):
        iid = item["item_id"]
        label = "answerable" if item["answerable"] else "unanswerable"
        pending = [a for a in arms if a not in done.get(iid, set())]
        if not pending:
            n_skip += 1
            print(f"[{n:>3}/{len(items)}] {label:<12} already done, skipped", flush=True)
            continue

        store = index_texts(
            item["contexts"], collection=collection_name(iid), source=f"squad2:{item['title']}"
        )
        try:
            for arm in pending:
                t0 = time.perf_counter()
                try:
                    r = answer_question(
                        item["question"],
                        store=store,
                        item_id=iid,
                        run_tag=args.tag,
                        baseline=(arm == "baseline"),
                        top_k=args.top_k,
                    )
                    n_run += 1
                    print(
                        f"[{n:>3}/{len(items)}] {label:<12} {arm:<8} "
                        f"verdict={r.verdict:<22} composite={r.confidence.composite:.3f} "
                        f"({time.perf_counter() - t0:.1f}s)",
                        flush=True,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad item must not kill the sweep
                    n_err += 1
                    print(
                        f"[{n:>3}/{len(items)}] {label:<12} {arm:<8} ERROR "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
        finally:
            with contextlib.suppress(Exception):  # cleanup is best effort
                store.delete_collection()

        elapsed = time.perf_counter() - t_start
        remaining = (len(items) - n) * elapsed / max(n - n_skip, 1)
        print(f"        elapsed {elapsed / 60:.1f} min, eta {remaining / 60:.1f} min", flush=True)

    print(
        f"\nDone: {n_run} runs, {n_skip} skipped, {n_err} errors "
        f"in {(time.perf_counter() - t_start) / 60:.1f} min. Tag: {args.tag!r}"
    )
    print("Evaluate with: python scripts/eval_benchmark.py --tag " + args.tag)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    init_db()
    raise SystemExit(main())
