"""Document ingestion: load -> chunk -> embed -> persist to Chroma.

Design note: every chunk carries ``source`` and ``chunk_index`` metadata. The
confidence layer needs to cite which passage supported which claim, and the
dashboard needs to attribute low-confidence answers back to specific documents,
so provenance is recorded at ingest time rather than reconstructed later.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_DIR, RAW_DIR, settings
from src.embeddings import get_embeddings

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


def _load_one(path: Path) -> list[Document]:
    """Read a file into per-page (PDF) or single (text/docx) Documents.

    Uses pypdf and python-docx directly rather than the langchain-community
    loaders, which are being sunset. Page numbers are preserved for PDFs so a
    low-confidence answer can be traced to an exact page during the demo.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs = [
            Document(page_content=text, metadata={"page": i + 1})
            for i, page in enumerate(reader.pages)
            if (text := (page.extract_text() or "").strip())
        ]
    elif suffix == ".docx":
        import docx

        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs if p.text.strip())
        docs = [Document(page_content=text)] if text.strip() else []
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        docs = [Document(page_content=text)] if text.strip() else []
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    for d in docs:
        d.metadata["source"] = path.name
    return docs


def load_documents(directory: Path | str = RAW_DIR) -> list[Document]:
    directory = Path(directory)
    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            try:
                loaded = _load_one(path)
                docs.extend(loaded)
                log.info("Loaded %s (%d page/section objects)", path.name, len(loaded))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop ingest
                log.error("Failed to load %s: %s", path.name, exc)
    return docs


def _split_markdown_sections(text: str) -> list[Document]:
    """Split a markdown document on its headings before chunking.

    A heading is a semantic boundary, and letting the next section's header
    bleed into the previous chunk's tail is not cosmetic: measured on the
    course notes, a stray ``## Unit 3`` header at the end of a Unit 2 chunk
    collapses nli-deberta entailment for claims that chunk fully supports
    from 0.99 to 0.001 - the model reads the mixed-premise passage as being
    about something else. Sections longer than the chunk size still go through
    the recursive splitter afterwards.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    return splitter.split_text(text)


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split on natural boundaries first, falling back to characters.

    The separator order matters: breaking on paragraph then sentence keeps
    chunks semantically whole, which directly improves the NLI entailment
    signal the confidence layer depends on. Mid-sentence chunks produce
    passages that entail nothing and depress Gate A scores for no good reason.
    Markdown documents are first divided on their headings (see
    ``_split_markdown_sections``).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.retrieval.chunk_size,
        chunk_overlap=settings.retrieval.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    pieces: list[Document] = []
    for doc in docs:
        if doc.metadata.get("source", "").lower().endswith(".md"):
            split_pieces = _split_markdown_sections(doc.page_content)
            for p in split_pieces:
                p.metadata = {**doc.metadata, **p.metadata}
            pieces.extend(split_pieces)
        else:
            pieces.append(doc)

    chunks = splitter.split_documents(pieces)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        c.metadata.setdefault("source", "unknown")
    log.info("Produced %d chunks from %d documents", len(chunks), len(docs))
    return chunks


def _chunk_id(chunk: Document) -> str:
    """Stable content-addressed id so re-ingesting a file cannot duplicate it."""
    key = f"{chunk.metadata.get('source')}::{chunk.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def build_index(
    directory: Path | str = RAW_DIR,
    *,
    reset: bool = False,
    collection: str | None = None,
):
    """Ingest a directory into Chroma and return the vector store."""
    from langchain_chroma import Chroma

    name = collection or settings.retrieval.collection_name
    store = Chroma(
        collection_name=name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": settings.retrieval.distance_metric},
    )

    if reset:
        log.warning("Resetting collection %r", name)
        store.delete_collection()
        store = Chroma(
            collection_name=name,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": settings.retrieval.distance_metric},
        )

    docs = load_documents(directory)
    if not docs:
        log.warning("No supported documents found in %s", directory)
        return store

    chunks = chunk_documents(docs)
    store.add_documents(chunks, ids=[_chunk_id(c) for c in chunks])
    log.info("Index %r now holds %d vectors", name, store._collection.count())
    return store


def index_texts(texts: list[str], *, collection: str, source: str = "inline"):
    """Build a throwaway collection from raw strings.

    Used by the benchmark harness: SQuAD/HotpotQA supply their own context
    paragraphs per question, so each item gets an isolated index rather than
    polluting the demo collection.
    """
    from langchain_chroma import Chroma

    docs = [
        Document(page_content=t, metadata={"source": source, "chunk_index": i})
        for i, t in enumerate(texts)
    ]
    return Chroma.from_documents(
        docs,
        embedding=get_embeddings(),
        collection_name=collection,
        collection_metadata={"hnsw:space": settings.retrieval.distance_metric},
    )
