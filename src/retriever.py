"""Retrieval with explicit, interpretable similarity scores.

``Chroma.similarity_search_with_score`` returns a *distance*, and which distance
depends on the collection's ``hnsw:space``. Silently treating that number as a
similarity is a real and common bug, so the conversion is done here, once, and
unit-tested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import CHROMA_DIR, settings
from src.embeddings import get_embeddings

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    similarity: float  # always in [0, 1], higher is better
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]

    @property
    def contexts(self) -> list[str]:
        return [c.text for c in self.chunks]

    @property
    def top_similarity(self) -> float:
        return self.chunks[0].similarity if self.chunks else 0.0

    @property
    def mean_similarity(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(c.similarity for c in self.chunks) / len(self.chunks)

    @property
    def similarity_margin(self) -> float:
        """Gap between the best and second-best hit.

        A wide margin means the index has one clearly correct passage. A narrow
        margin over several mediocre hits is a classic signature of a query the
        corpus does not actually cover, which is precisely the case we want to
        abstain on.
        """
        if len(self.chunks) < 2:
            return 0.0
        return self.chunks[0].similarity - self.chunks[1].similarity


def _distance_to_similarity(distance: float, space: str) -> float:
    """Map a Chroma distance onto [0, 1] similarity.

    With normalised embeddings: cosine distance is ``1 - cos``, so similarity is
    ``1 - distance``. Squared-L2 between unit vectors is ``2 - 2*cos``, giving
    ``1 - distance/2``. Inner product distance is ``1 - ip``.
    """
    if space == "cosine":
        sim = 1.0 - distance
    elif space in {"l2", "euclidean"}:
        sim = 1.0 - distance / 2.0
    elif space == "ip":
        sim = 1.0 - distance
    else:  # pragma: no cover - defensive
        log.warning("Unknown distance space %r; treating distance as 1 - similarity", space)
        sim = 1.0 - distance
    return max(0.0, min(1.0, sim))


def get_store(collection: str | None = None):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection or settings.retrieval.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": settings.retrieval.distance_metric},
    )


def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    store=None,
    collection: str | None = None,
) -> RetrievalResult:
    k = top_k or settings.retrieval.top_k
    store = store or get_store(collection)
    space = settings.retrieval.distance_metric

    try:
        hits = store.similarity_search_with_score(query, k=k)
    except Exception as exc:  # noqa: BLE001 - empty/missing index must not crash the UI
        log.error("Retrieval failed: %s", exc)
        return RetrievalResult(query=query, chunks=[])

    chunks = [
        RetrievedChunk(
            text=doc.page_content,
            source=str(doc.metadata.get("source", "unknown")),
            similarity=_distance_to_similarity(float(score), space),
            metadata=dict(doc.metadata),
        )
        for doc, score in hits
    ]
    chunks.sort(key=lambda c: c.similarity, reverse=True)
    return RetrievalResult(query=query, chunks=chunks)
