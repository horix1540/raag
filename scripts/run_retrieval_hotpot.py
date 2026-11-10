"""Retrieval-only ablation: sentence granularity vs 800-character chunks.

This script answers one question and deliberately answers it without a language
model: how much does the project's real 800-character chunking cost retrieval,
measured against HotpotQA's human-annotated supporting facts?

Two indexed views of the same 150 items are compared (both written by
``scripts/build_hotpot.py``):

  sentence - one indexed unit per sentence, so a retrieved unit maps 1:1 onto a
      supporting-fact label. Clean ground truth, but not the pipeline the
      project demonstrates.
  chunk    - units of up to 800 characters, the real setting. A unit counts as
      supporting if it *contains* a gold sentence, which is a noisier label:
      the chunk drags in neighbouring sentences that support nothing.

Metrics per item, at k = 5:

  supporting_recall - of the gold units, the fraction retrieval surfaced. This
      is the number that decides whether Gate B could possibly succeed: a claim
      cannot be entailed by a passage that was never retrieved.
  supporting_prec   - of the k retrieved units, the fraction that are gold.
      Chunking should inflate this mechanically, because each chunk covers more
      text, and that inflation is itself the dilution being measured.

Co-location is tracked separately because it is the multi-hop mechanism worth
naming. A two-hop question needs two supporting sentences. Under chunking those
sentences either land in one unit (a single premise entails the whole claim, so
NLI has an easier job than at sentence granularity) or in different units
(neither premise alone entails it, which is exactly Gate A's known weakness).
Reporting one averaged recall number would hide both effects, so recall and
precision are also broken down by whether chunking co-located the gold
sentences.

Collection lifecycle is explicit and fail-loud, which matters more here than it
looks: ``index_texts`` builds an in-memory collection, and rebuilding the same
collection name *appends* rather than replaces (measured: a two-document build
run twice reports four documents). A silently-failed teardown would therefore
leave sentence units inside the chunk-granularity index and quietly corrupt the
comparison this script exists to make.

    python scripts/run_retrieval_hotpot.py
    python scripts/run_retrieval_hotpot.py --limit 10          # smoke run
    python scripts/run_retrieval_hotpot.py --granularity chunk # one view only
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
from src.retriever import retrieve  # noqa: E402

GRANULARITIES = ("sentence", "chunk")


class CollectionLeakError(RuntimeError):
    """Raised when a benchmark collection cannot be torn down or verified.

    Deliberately fatal. The alternative - suppressing the error and carrying on -
    is how sentence-granularity units end up being retrieved during the chunk
    granularity pass, which would not crash anything and would invalidate every
    number the script prints.
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
    """Chroma restricts collection names; benchmark ids are hex but stay defensive.

    The granularity is part of the name so that a leaked collection from one
    view cannot be reused by the other even if teardown were skipped entirely.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", item_id)[:40]
    return f"hp_{granularity[:4]}_{safe}"


def teardown(store, name: str, *, must_exist: bool) -> None:
    """Delete a collection, loudly.

    ``must_exist=False`` is used for the pre-build guard, where a missing
    collection is the normal case and only a *different* failure is worth
    raising on. After a build the collection certainly exists, so any failure
    is real and must stop the sweep.
    """
    try:
        store.delete_collection()
    except ValueError:
        # Chroma raises ValueError when the collection is already gone.
        if must_exist:
            raise CollectionLeakError(
                f"collection {name!r} vanished before teardown; the index may be shared"
            ) from None
    except Exception as exc:  # noqa: BLE001 - re-raised as a fatal, named error
        raise CollectionLeakError(f"could not tear down collection {name!r}: {exc}") from exc


def build_clean(units: list[str], name: str, source: str):
    """Build a collection and assert it holds exactly ``len(units)`` vectors.

    The assertion is the actual leak detector. Rebuilding an existing collection
    name appends, so a count above the expected value means a previous item's
    units - or the other granularity's - are still present.
    """
    store = index_texts(units, collection=name, source=source)
    count = store._collection.count()
    if count != len(units):
        teardown(store, name, must_exist=True)
        raise CollectionLeakError(
            f"collection {name!r} holds {count} vectors, expected {len(units)}: "
            "a previous collection with this name was not torn down"
        )
    return store


def colocation(item: dict) -> dict:
    """Does chunking merge the gold sentences into one unit?

    Computed structurally from the chunk view's sentence membership, before any
    retrieval, so it is a property of the indexing strategy rather than of a
    particular query's luck.
    """
    gold_sentences = set(item["sentence_view"]["gold_units"])
    memberships = item["chunk_view"]["unit_sentence_indices"]

    hosting = [
        idx for idx, sentences in enumerate(memberships)
        if gold_sentences & set(sentences)
    ]
    return {
        "n_gold_sentences": len(gold_sentences),
        "n_gold_chunks": len(hosting),
        # One chunk holding every gold sentence: NLI gets a unified premise.
        "colocated": len(hosting) == 1 and len(gold_sentences) > 1,
        # The multi-hop case chunking cannot help with.
        "split": len(hosting) > 1,
    }


def score_retrieval(hit_units: list[int], gold_units: list[int]) -> dict:
    """Recall and precision of gold units among the retrieved units."""
    gold = set(gold_units)
    hits = set(hit_units)
    found = gold & hits
    return {
        "n_gold": len(gold),
        "n_retrieved": len(hits),
        "n_gold_retrieved": len(found),
        "supporting_recall": round(len(found) / len(gold), 4) if gold else None,
        "supporting_prec": round(len(found) / len(hits), 4) if hits else 0.0,
        # Every gold unit surfaced: the only case where Gate B is not already
        # handicapped by retrieval before it runs.
        "full_recall": bool(gold) and gold <= hits,
    }


def retrieved_unit_indices(result) -> list[int]:
    """Map retrieved passages back to their unit index.

    ``index_texts`` stamps ``chunk_index`` with the position in the list it was
    given, so the mapping is exact and no text matching is needed.
    """
    out = []
    for chunk in result.chunks:
        idx = chunk.metadata.get("chunk_index")
        if idx is not None:
            out.append(int(idx))
    return out


def run_granularity(
    items: list[dict], granularity: str, *, top_k: int
) -> list[dict]:
    """Sweep every item under one indexing strategy."""
    view_key = f"{granularity}_view"
    rows: list[dict] = []
    t_start = time.perf_counter()

    for n, item in enumerate(items, 1):
        iid = item["item_id"]
        view = item[view_key]
        units = view["units"]
        gold = view["gold_units"]
        name = collection_name(iid, granularity)

        store = build_clean(units, name, f"hotpot:{granularity}")
        try:
            t0 = time.perf_counter()
            result = retrieve(item["question"], top_k=top_k, store=store)
            latency_ms = (time.perf_counter() - t0) * 1000

            hit_units = retrieved_unit_indices(result)
            scored = score_retrieval(hit_units, gold)
            co = colocation(item)

            # Did retrieval surface a single unit containing every gold sentence?
            # Distinct from structural co-location: the unified premise only
            # helps if it was actually retrieved.
            gold_sentences = set(item["sentence_view"]["gold_units"])
            memberships = item["chunk_view"]["unit_sentence_indices"]
            unified_retrieved = False
            if granularity == "chunk":
                unified_retrieved = any(
                    gold_sentences <= set(memberships[u])
                    for u in hit_units
                    if u < len(memberships)
                )

            rows.append(
                {
                    "item_id": iid,
                    "granularity": granularity,
                    "hop_type": item.get("hop_type"),
                    "level": item.get("level"),
                    "n_units": len(units),
                    "retrieval_ms": round(latency_ms, 2),
                    "top_similarity": round(result.top_similarity, 4),
                    "mean_similarity": round(result.mean_similarity, 4),
                    "similarity_margin": round(result.similarity_margin, 4),
                    "unified_premise_retrieved": unified_retrieved,
                    **scored,
                    **co,
                }
            )
        finally:
            teardown(store, name, must_exist=True)

        if n % 25 == 0 or n == len(items):
            done = [r for r in rows if r["supporting_recall"] is not None]
            mean_recall = statistics.mean(r["supporting_recall"] for r in done) if done else 0.0
            print(
                f"  [{granularity:<8} {n:>3}/{len(items)}] "
                f"mean recall so far {mean_recall:.3f}  "
                f"({time.perf_counter() - t_start:.0f}s)",
                flush=True,
            )

    return rows


def aggregate(rows: list[dict]) -> dict:
    """Headline metrics for one granularity, plus the co-location breakdown."""
    if not rows:
        return {"n": 0}

    def summarise(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0}
        recalls = [r["supporting_recall"] for r in subset if r["supporting_recall"] is not None]
        precs = [r["supporting_prec"] for r in subset if r["supporting_prec"] is not None]
        return {
            "n": len(subset),
            "supporting_recall": round(statistics.mean(recalls), 4) if recalls else None,
            "supporting_prec": round(statistics.mean(precs), 4) if precs else None,
            "full_recall_rate": round(
                sum(1 for r in subset if r["full_recall"]) / len(subset), 4
            ),
            "median_retrieval_ms": round(
                statistics.median(r["retrieval_ms"] for r in subset), 2
            ),
            "mean_units_indexed": round(statistics.mean(r["n_units"] for r in subset), 1),
        }

    out = summarise(rows)
    out["by_colocation"] = {
        "colocated": summarise([r for r in rows if r["colocated"]]),
        "split": summarise([r for r in rows if r["split"]]),
    }
    out["unified_premise_retrieved_rate"] = round(
        sum(1 for r in rows if r["unified_premise_retrieved"]) / len(rows), 4
    )
    return out


def print_table(report: dict) -> None:
    """The comparative table, which is the point of the script."""
    arms = report["granularities"]
    names = [g for g in GRANULARITIES if g in arms]

    print("\n" + "=" * 78)
    print(f"Retrieval ablation, k={report['top_k']}, n={report['n_items']} HotpotQA items")
    print("=" * 78)

    header = f"{'Metric':<34}" + "".join(f"{g:>14}" for g in names)
    print(header)
    print("-" * len(header))

    def row(label: str, path: list[str], fmt: str = "{:.4f}") -> None:
        cells = ""
        for g in names:
            node = arms[g]
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            cells += f"{(fmt.format(node) if isinstance(node, (int, float)) else '-'):>14}"
        print(f"{label:<34}{cells}")

    row("mean units indexed", ["mean_units_indexed"], "{:.1f}")
    row("supporting_recall", ["supporting_recall"])
    row("supporting_prec", ["supporting_prec"])
    row("full recall rate", ["full_recall_rate"])
    row("median retrieval ms", ["median_retrieval_ms"], "{:.2f}")
    row("unified premise retrieved", ["unified_premise_retrieved_rate"])

    print("\n-- gold sentences CO-LOCATED by chunking " + "-" * 36)
    row("n items", ["by_colocation", "colocated", "n"], "{:.0f}")
    row("supporting_recall", ["by_colocation", "colocated", "supporting_recall"])
    row("supporting_prec", ["by_colocation", "colocated", "supporting_prec"])
    row("full recall rate", ["by_colocation", "colocated", "full_recall_rate"])

    print("\n-- gold sentences SPLIT across units " + "-" * 40)
    row("n items", ["by_colocation", "split", "n"], "{:.0f}")
    row("supporting_recall", ["by_colocation", "split", "supporting_recall"])
    row("supporting_prec", ["by_colocation", "split", "supporting_prec"])
    row("full recall rate", ["by_colocation", "split", "full_recall_rate"])
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=BENCH_DIR / "hotpot_dev_150.jsonl")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "retrieval_ablation_m4.json")
    ap.add_argument("--rows-out", type=Path, default=REPORTS_DIR / "retrieval_ablation_m4_rows.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items")
    ap.add_argument("--top-k", type=int, default=None, help="retrieval k (default from config)")
    ap.add_argument(
        "--granularity",
        choices=("both", *GRANULARITIES),
        default="both",
        help="run one view only, for debugging",
    )
    args = ap.parse_args()

    top_k = args.top_k or settings.retrieval.top_k
    items = load_items(args.bench)
    if args.limit:
        items = items[: args.limit]

    views = list(GRANULARITIES) if args.granularity == "both" else [args.granularity]

    print(f"Retrieval ablation: {len(items)} items, k={top_k}, views={views}")
    print(f"  embeddings: {settings.retrieval.embedding_model} on {settings.retrieval.embedding_device}")
    print("  no language model is used by this script\n")

    all_rows: list[dict] = []
    report: dict = {
        "bench": str(args.bench),
        "n_items": len(items),
        "top_k": top_k,
        "embedding_model": settings.retrieval.embedding_model,
        "granularities": {},
    }

    for granularity in views:
        t0 = time.perf_counter()
        rows = run_granularity(items, granularity, top_k=top_k)
        all_rows.extend(rows)
        report["granularities"][granularity] = aggregate(rows)
        report["granularities"][granularity]["sweep_seconds"] = round(
            time.perf_counter() - t0, 1
        )

    # Deltas only make sense when both views ran.
    if all(g in report["granularities"] for g in GRANULARITIES):
        s = report["granularities"]["sentence"]
        c = report["granularities"]["chunk"]
        report["delta_chunk_minus_sentence"] = {
            "supporting_recall": round(c["supporting_recall"] - s["supporting_recall"], 4),
            "supporting_prec": round(c["supporting_prec"] - s["supporting_prec"], 4),
            "full_recall_rate": round(c["full_recall_rate"] - s["full_recall_rate"], 4),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.rows_out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_table(report)
    if "delta_chunk_minus_sentence" in report:
        d = report["delta_chunk_minus_sentence"]
        print("chunk minus sentence:")
        print(f"  supporting_recall {d['supporting_recall']:+.4f}")
        print(f"  supporting_prec   {d['supporting_prec']:+.4f}")
        print(f"  full recall rate  {d['full_recall_rate']:+.4f}")
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.rows_out} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
