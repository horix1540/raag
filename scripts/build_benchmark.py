"""Sample a labelled SQuAD 2.0 evaluation set for the benchmark harness.

Why SQuAD 2.0 and not SQuAD 1.1: v2 deliberately includes questions that its
own paragraph cannot answer, written by crowdworkers to look answerable. That
is exactly the label this project needs. An unanswerable item is a question
where the correct behaviour is to abstain, so it gives Gate A a ground truth to
be judged against instead of a threshold chosen by eye.

Each item is stored with its gold paragraph plus a few distractor paragraphs
drawn from *different* Wikipedia articles. The distractors matter: with only the
gold paragraph in the index, retrieval has nothing to choose between, so
top_similarity and similarity_margin degenerate and the harness would measure
the reranker in isolation rather than the pipeline. Distractors are taken from
other titles so that none of them can accidentally contain the answer.

    python scripts/build_benchmark.py                     # defaults, 150 items
    python scripts/build_benchmark.py --n-answerable 200 --n-unanswerable 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BENCH_DIR  # noqa: E402

DATASET = "rajpurkar/squad_v2"
SPLIT = "validation"


def load_squad(split: str = SPLIT) -> list[dict]:
    """Download (or reuse the cached) SQuAD 2.0 dev split."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "context": row["context"],
            "question": row["question"],
            # The HF loader has no is_impossible column: an unanswerable
            # question is one whose answers list is empty.
            "answers": list(row["answers"]["text"]),
        }
        for row in ds
    ]


def unique_paragraphs(rows: list[dict]) -> list[tuple[str, str]]:
    """Distinct (title, paragraph) pairs, used as the distractor pool."""
    seen: set[str] = set()
    pool: list[tuple[str, str]] = []
    for r in rows:
        if r["context"] not in seen:
            seen.add(r["context"])
            pool.append((r["title"], r["context"]))
    return pool


def pick_distractors(
    pool: list[tuple[str, str]], exclude_title: str, k: int, rng: random.Random
) -> list[tuple[str, str]]:
    """k paragraphs from titles other than ``exclude_title``.

    Rejection sampling rather than filtering the whole pool per item: the pool
    holds thousands of paragraphs and only a handful share any given title, so
    a retry is far cheaper than rebuilding a filtered list 150 times.
    """
    out: list[tuple[str, str]] = []
    tries = 0
    while len(out) < k and tries < k * 50:
        tries += 1
        title, para = pool[rng.randrange(len(pool))]
        if title == exclude_title or any(para == p for _, p in out):
            continue
        out.append((title, para))
    return out


def build(
    n_answerable: int, n_unanswerable: int, n_distractors: int, seed: int
) -> list[dict]:
    rows = load_squad()
    answerable = [r for r in rows if r["answers"]]
    unanswerable = [r for r in rows if not r["answers"]]
    print(
        f"  dev split: {len(rows)} questions "
        f"({len(answerable)} answerable, {len(unanswerable)} unanswerable)"
    )
    if n_answerable > len(answerable) or n_unanswerable > len(unanswerable):
        raise SystemExit("Requested more items than the split contains.")

    rng = random.Random(seed)
    pool = unique_paragraphs(rows)
    print(f"  distractor pool: {len(pool)} distinct paragraphs")

    chosen = rng.sample(answerable, n_answerable) + rng.sample(unanswerable, n_unanswerable)
    rng.shuffle(chosen)

    items = []
    for r in chosen:
        distractors = pick_distractors(pool, r["title"], n_distractors, rng)
        contexts = [r["context"]] + [p for _, p in distractors]
        order = list(range(len(contexts)))
        rng.shuffle(order)
        items.append(
            {
                "item_id": r["id"],
                "question": r["question"],
                "title": r["title"],
                "answers": r["answers"],
                "answerable": bool(r["answers"]),
                "contexts": [contexts[i] for i in order],
                "gold_index": order.index(0),
                "distractor_titles": [t for t, _ in distractors],
            }
        )
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-answerable", type=int, default=75)
    ap.add_argument("--n-unanswerable", type=int, default=75)
    ap.add_argument("--distractors", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    total = args.n_answerable + args.n_unanswerable
    out = args.out or BENCH_DIR / f"squad2_dev_{total}.jsonl"

    print(f"Building SQuAD 2.0 sample: {total} items, seed {args.seed}")
    items = build(args.n_answerable, args.n_unanswerable, args.distractors, args.seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_ctx = sum(len(i["contexts"]) for i in items) / len(items)
    print(f"  wrote {len(items)} items to {out}")
    print(f"  mean passages per item: {n_ctx:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
