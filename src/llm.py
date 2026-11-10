"""Provider-agnostic LLM factory.

The pipeline never imports a provider SDK directly. It asks for a chat model
here and gets back a LangChain ``BaseChatModel``, so switching from local
Ollama to Groq to Gemini is a one-line change in ``.env`` and requires no code
edits anywhere else. This matters for the evaluation chapter: the same
experiment must be runnable against different backends to show that the
confidence layer is a property of the pipeline, not of one particular model.
"""

from __future__ import annotations

import dataclasses
import functools
import logging

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import LLMConfig, Provider, settings

log = logging.getLogger(__name__)


class ProviderUnavailableError(RuntimeError):
    """Raised when the configured provider cannot be reached or is missing a key."""


def _build_ollama(model: str, temperature: float, cfg: LLMConfig) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=cfg.ollama_base_url,
        num_predict=cfg.max_tokens,
        # Keep the model resident between calls; on an 8 GB card the reload
        # cost otherwise dominates a benchmark sweep.
        keep_alive="30m",
    )


def _build_groq(model: str, temperature: float, cfg: LLMConfig) -> BaseChatModel:
    import os

    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise ProviderUnavailableError(
            "GROQ_API_KEY is not set. Add it to .env or switch RAG_PROVIDER=ollama."
        )
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
    )


def _build_gemini(model: str, temperature: float, cfg: LLMConfig) -> BaseChatModel:
    import os

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.getenv("GOOGLE_API_KEY"):
        raise ProviderUnavailableError(
            "GOOGLE_API_KEY is not set. Add it to .env or switch RAG_PROVIDER=ollama."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max(cfg.max_tokens, 2048),
        thinking_budget=0,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
    )


_BUILDERS = {
    "ollama": _build_ollama,
    "groq": _build_groq,
    "gemini": _build_gemini,
}

# Sensible per-provider defaults, used when the configured model name is
# obviously meant for a different provider (e.g. "llama3.1:8b" on Groq).
_DEFAULTS: dict[Provider, dict[str, str]] = {
    "ollama": {"generator": "llama3.1:8b", "judge": "gemma2:latest"},
    "groq": {"generator": "llama-3.3-70b-versatile", "judge": "llama-3.1-8b-instant"},
    "gemini": {"generator": "gemini-3.5-flash", "judge": "gemini-3.5-flash-lite"},
}


def _resolve_model(name: str, provider: Provider, role: str) -> str:
    """Swap an Ollama-style tag for a cloud model name when the provider changed.

    Ollama tags contain a colon (``llama3.1:8b``); cloud model ids do not. This
    keeps a single .env from breaking the moment you flip RAG_PROVIDER.
    """
    is_ollama_tag = ":" in name
    if provider == "ollama":
        return name if is_ollama_tag else _DEFAULTS["ollama"][role]
    if is_ollama_tag:
        fallback = _DEFAULTS[provider][role]
        log.warning("Model %r looks like an Ollama tag; using %r for %s.", name, fallback, provider)
        return fallback
    return name


@functools.lru_cache(maxsize=8)
def _cached(provider: Provider, model: str, temperature: float, max_tokens: int) -> BaseChatModel:
    cfg = settings.llm
    cfg = dataclasses.replace(cfg, max_tokens=max_tokens)
    try:
        return _BUILDERS[provider](model, temperature, cfg)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProviderUnavailableError(
            f"The LangChain integration for {provider!r} is not installed: {exc}"
        ) from exc


def get_generator() -> BaseChatModel:
    """The model that writes the answer."""
    cfg = settings.llm
    model = _resolve_model(cfg.generator_model, cfg.provider, "generator")
    return _cached(cfg.provider, model, cfg.temperature, cfg.max_tokens)


def get_judge() -> BaseChatModel:
    """The model that scores the answer. Deliberately not the generator."""
    cfg = settings.llm
    model = _resolve_model(cfg.judge_model, cfg.provider, "judge")
    return _cached(cfg.provider, model, cfg.judge_temperature, cfg.judge_max_tokens)


def health_check() -> tuple[bool, str]:
    """Cheap round-trip so the UI can show provider status without crashing."""
    try:
        reply = get_generator().invoke("Reply with the single word: ok")
        raw = getattr(reply, "content", reply)
        if isinstance(raw, list):
            text = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
            ).strip()
        else:
            text = str(raw).strip()
        return True, f"{settings.llm.provider} reachable (replied {text[:40]!r})"
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
        return False, f"{settings.llm.provider} unreachable: {type(exc).__name__}: {exc}"
