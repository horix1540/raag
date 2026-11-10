"""Embedding model loader with automatic GPU-to-CPU fallback.

Kept separate from ingest/retrieve because both sides must use the *identical*
embedding model. Loading it in two places with two different configs is the
classic way to get a silently broken vector store, so there is exactly one
entry point.
"""

from __future__ import annotations

import functools
import logging

from src.config import settings

log = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    """Honour the requested device if torch can actually use it."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if requested.startswith("cuda") and torch.cuda.is_available():
            return requested
        log.info("CUDA requested but unavailable; falling back to CPU.")
    except ImportError:  # pragma: no cover
        log.warning("torch not importable; falling back to CPU.")
    return "cpu"


@functools.lru_cache(maxsize=2)
def get_embeddings(device: str | None = None):
    """Return a LangChain-compatible embeddings object.

    Normalisation is switched on so that Chroma's cosine space and any manual
    dot products we take in the confidence layer agree with each other.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    cfg = settings.retrieval
    resolved = _resolve_device(device or cfg.embedding_device)
    log.info("Loading embeddings %s on %s", cfg.embedding_model, resolved)
    return HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        model_kwargs={"device": resolved},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def embedding_dimension() -> int:
    """Useful for asserting an existing Chroma collection matches the model."""
    return len(get_embeddings().embed_query("dimension probe"))
