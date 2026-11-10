"""Build or rebuild the Chroma index from data/raw/.

    python scripts/build_index.py            # add new documents
    python scripts/build_index.py --reset    # wipe and rebuild from scratch
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR, settings  # noqa: E402
from src.ingest import build_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(RAW_DIR), help="directory of documents to ingest")
    ap.add_argument("--reset", action="store_true", help="delete the collection first")
    ap.add_argument("--collection", default=settings.retrieval.collection_name)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    source = Path(args.dir)
    if not source.exists():
        print(f"error: {source} does not exist", file=sys.stderr)
        return 1

    files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in
             {".pdf", ".txt", ".md", ".docx"}]
    if not files:
        print(f"error: no .pdf/.txt/.md/.docx files found under {source}", file=sys.stderr)
        return 1

    print(f"Ingesting {len(files)} file(s) from {source} into '{args.collection}'")
    for f in files:
        print(f"  - {f.name}")

    store = build_index(source, reset=args.reset, collection=args.collection)
    total = store._collection.count()
    print(f"\nDone. Collection '{args.collection}' holds {total} chunks.")
    print(f"Embedding model: {settings.retrieval.embedding_model}")
    print(f"Chunk size {settings.retrieval.chunk_size}, overlap {settings.retrieval.chunk_overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
