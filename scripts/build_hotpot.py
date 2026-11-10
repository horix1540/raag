"""Build a labelled HotpotQA multi-hop sample for the Milestone 4 ablation.

Why HotpotQA at all: every question requires combining two paragraphs, and the
dataset ships *human-annotated supporting facts* - the specific sentences an
annotator marked as needed to answer. That is real ground truth for
groundedness, not the answerable/unanswerable proxy Gate B was calibrated
against in Milestone 3. It is also the direct test of Gate A's known weakness,
since a reranker scores passages one at a time and a two-hop answer may score
low on each half.

Why this script downloads a parquet file by hand: ``hotpotqa/hotpot_qa`` on the
Hub is a script-based dataset, and ``datasets`` 4.x refuses to execute dataset
scripts at all (``trust_remote_code`` was removed, not merely gated). The
official CMU mirror times out. The Hub's parquet conversion endpoint serves the
same rows over plain HTTP, so that is what is used.

Two views of the same 150 items are written, which is the whole point of the
ablation:

  sentence granularity - one indexed unit per sentence, so a retrieved unit maps
      1:1 onto a supporting-fact label. Clean ground truth, but not the
      pipeline the project actually demonstrates.
  chunk granularity    - units of up to CHUNK_CHARS characters, the project's
      real setting. A chunk counts as supporting if it contains any gold
      sentence, which is a noisier label: the chunk drags in neighbouring
      sentences that support nothing. Measuring that dilution is the experiment.

Chunks are assembled from whole sentences rather than by running the character
splitter, so every chunk's sentence membership is known exactly and the gold
label can be derived rather than guessed. Paragraph boundaries are respected,
matching the recursive splitter's first separator.

    python scripts/build_hotpot.py               # 150 items, seed 20260916
    python scripts/build_hotpot.py --n 50        # smaller smoke sample
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BENCH_DIR  # noqa: E402

PARQUET_URL = (
    "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/"
    "parquet/distractor/validation/0.parquet"
)
CHUNK_CHARS = 800


def load_hotpot(url: str = PARQUET_URL):
    """Fetch the distractor dev split as a DataFrame.

    Read into memory rather than streamed to a temp file: the split is tens of
    megabytes, and a partial file on disk would be worse than a failed request.
    """
    import pandas as pd

    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=300) as fh:  # noqa: S310 - fixed https URL
        raw = fh.read()
    print(f"  {len(raw) / 1e6:.1f} MB downloaded", flush=True)
    return pd.read_parquet(io.BytesIO(raw))


def paragraphs_of(row) -> list[tuple[str, list[str]]]:
    """(title, sentences) per paragraph, tolerating a length mismatch.

    The parquet encodes context as a struct of parallel title/sentences arrays.
    They should be the same length; a strict zip would abort the whole build
    over one malformed row, so the shorter is taken and the row still counts.
    """
    ctx = row["context"]
    titles = [str(t) for t in list(ctx["title"])]
    sentences = [[str(s) for s in list(group)] for group in list(ctx["sentences"])]
    n = min(len(titles), len(sentences))
    return list(zip(titles[:n], sentences[:n], strict=True))


def supporting_of(row) -> list[tuple[str, int]]:
    """(title, sentence_index) pairs the annotators marked as supporting."""
    sf = row["supporting_facts"]
    titles = [str(t) for t in list(sf["title"])]
    sent_ids = [int(i) for i in list(sf["sent_id"])]
    n = min(len(titles), len(sent_ids))
    return list(zip(titles[:n], sent_ids[:n], strict=True))


def flatten(paras: list[tuple[str, list[str]]]) -> tuple[list[str], list[str], dict]:
    """Flatten paragraphs into one sentence list plus a lookup for gold labels.

    Returns the sentences, the source title per sentence, and a
    ``(title, local_index) -> global_index`` map. The map is what the
    supporting-fact pairs are resolved through.
    """
    sentences: list[str] = []
    sources: list[str] = []
    lookup: dict[tuple[str, int], int] = {}
    for title, group in paras:
        for local, sentence in enumerate(group):
            lookup[(title, local)] = len(sentences)
            sentences.append(sentence)
            sources.append(title)
    return sentences, sources, lookup


def resolve_gold(
    facts: list[tuple[str, int]], lookup: dict
) -> tuple[list[int], list[dict]]:
    """Map supporting facts onto global sentence indices, reporting failures.

    HotpotQA's indices occasionally point past the end of a paragraph's
    sentence list, and a title in ``supporting_facts`` is occasionally absent
    from ``context`` entirely. Both are skipped with a reason rather than
    raising, because dropping one fact from an item still leaves the item usable
    and a crash here would waste the whole download.
    """
    resolved: list[int] = []
    unresolved: list[dict] = []
    for title, local in facts:
        key = (title, local)
        if key in lookup:
            resolved.append(lookup[key])
            continue
        present = any(t == title for t, _ in lookup)
        unresolved.append(
            {
                "title": title,
                "sent_id": local,
                "reason": "sentence index out of range" if present else "title not in context",
            }
        )
    return sorted(set(resolved)), unresolved


def chunk_view(
    paras: list[tuple[str, list[str]]], gold: set[int], *, chunk_chars: int = CHUNK_CHARS
) -> tuple[list[str], list[list[int]], list[int]]:
    """Group whole sentences into <= chunk_chars units, per paragraph.

    Returns the chunk texts, the global sentence indices inside each chunk, and
    the indices of chunks holding at least one gold sentence. A chunk never
    spans two paragraphs, which is what the recursive splitter's "\\n\\n" first
    separator does in the real pipeline.
    """
    chunks: list[str] = []
    membership: list[list[int]] = []
    cursor = 0

    for _title, group in paras:
        current: list[str] = []
        current_ids: list[int] = []
        for sentence in group:
            gid = cursor
            cursor += 1
            candidate = len(" ".join([*current, sentence]))
            if current and candidate > chunk_chars:
                chunks.append(" ".join(current))
                membership.append(current_ids)
                current, current_ids = [sentence], [gid]
            else:
                current.append(sentence)
                current_ids.append(gid)
        if current:
            chunks.append(" ".join(current))
            membership.append(current_ids)

    gold_chunks = [i for i, ids in enumerate(membership) if gold.intersection(ids)]
    return chunks, membership, gold_chunks


def build_item(row) -> dict | None:
    """One benchmark item in both granularities, or None if it has no gold left."""
    paras = paragraphs_of(row)
    sentences, sources, lookup = flatten(paras)
    facts = supporting_of(row)
    gold, unresolved = resolve_gold(facts, lookup)
    if not gold:
        return None

    chunks, membership, gold_chunks = chunk_view(paras, set(gold))
    return {
        "item_id": str(row["id"]),
        "question": str(row["question"]),
        "answers": [str(row["answer"])],
        # Every HotpotQA question is answerable from its gold sentences. The
        # field is kept so the existing SQuAD-shaped evaluation code can read
        # these items without a special case, but it is constant by
        # construction: there is no abstention ground truth here, so every
        # abstention on this benchmark is an over-abstention.
        "answerable": True,
        "hop_type": str(row.get("type", "")),
        "level": str(row.get("level", "")),
        "titles": [t for t, _ in paras],
        "supporting_facts": [[t, i] for t, i in facts],
        "n_unresolved_facts": len(unresolved),
        "unresolved_facts": unresolved,
        "sentence_view": {
            "units": sentences,
            "unit_sources": sources,
            "gold_units": gold,
        },
        "chunk_view": {
            "units": chunks,
            "unit_sentence_indices": membership,
            "gold_units": gold_chunks,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or BENCH_DIR / f"hotpot_dev_{args.n}.jsonl"

    df = load_hotpot()
    print(f"  dev split: {len(df)} questions, columns={list(df.columns)}", flush=True)

    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    print(f"Sampling {len(sample)} items with seed {args.seed}", flush=True)

    items: list[dict] = []
    dropped = 0
    total_facts = 0
    total_unresolved = 0
    for _, row in sample.iterrows():
        item = build_item(row)
        if item is None:
            dropped += 1
            continue
        total_facts += len(item["supporting_facts"])
        total_unresolved += item["n_unresolved_facts"]
        items.append(item)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    resolved = total_facts - total_unresolved
    rate = resolved / total_facts if total_facts else 0.0
    mean_sent = sum(len(i["sentence_view"]["units"]) for i in items) / len(items)
    mean_chunk = sum(len(i["chunk_view"]["units"]) for i in items) / len(items)
    mean_gold_s = sum(len(i["sentence_view"]["gold_units"]) for i in items) / len(items)
    mean_gold_c = sum(len(i["chunk_view"]["gold_units"]) for i in items) / len(items)

    print(f"\n  wrote {len(items)} items to {out}")
    print(f"  items dropped (no resolvable gold): {dropped}")
    print(f"  supporting facts resolved: {resolved}/{total_facts} ({rate:.1%})")
    print(f"  mean paragraphs per item:      {sum(len(i['titles']) for i in items) / len(items):.1f}")
    print(f"  mean sentence units per item:  {mean_sent:.1f}  (gold {mean_gold_s:.2f})")
    print(f"  mean chunk units per item:     {mean_chunk:.1f}  (gold {mean_gold_c:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
