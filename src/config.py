"""Central configuration for the confidence-aware RAG pipeline.

Every tunable lives here so that experiments are reproducible from a single
file and so the Streamlit UI has one place to read defaults from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BENCH_DIR = DATA_DIR / "benchmarks"
CHROMA_DIR = DATA_DIR / "chroma"
RUNS_DB = DATA_DIR / "runs.sqlite3"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, BENCH_DIR, CHROMA_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

Provider = Literal["ollama", "groq", "gemini"]


@dataclass
class LLMConfig:
    """Which model answers the question, and which model judges the answer.

    The judge is kept separate from the generator on purpose: a model grading
    its own output is a known source of self-preference bias, and keeping the
    two swappable lets us measure how much that bias costs us.
    """

    provider: Provider = os.getenv("RAG_PROVIDER", "ollama")  # type: ignore[assignment]

    # Generation
    generator_model: str = os.getenv("RAG_GENERATOR_MODEL", "llama3.1:8b")
    temperature: float = 0.0  # deterministic: hallucination results must repeat
    max_tokens: int = 512

    # Judging (LLM-as-judge path)
    # Judge kept separate from the generator (self-preference bias). Note:
    # qwen2.5:7b-instruct was the original choice but answers "no" to every
    # DeepEval faithfulness verdict regardless of the claim (broken polarity,
    # deterministic at temperature 0), which makes its scores meaningless.
    # gemma2 follows the yes/no convention correctly in the same probes.
    judge_model: str = os.getenv("RAG_JUDGE_MODEL", "gemma2:latest")
    judge_temperature: float = 0.0
    # The judge emits JSON verdict lists over up to 5 paragraphs, which can
    # exceed the answer budget and truncate mid-JSON (unparseable), so it gets
    # its own, larger generation cap.
    judge_max_tokens: int = 2048

    # Networking
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    request_timeout: int = 180
    max_retries: int = 3


@dataclass
class RetrievalConfig:
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = os.getenv("RAG_EMBEDDING_DEVICE", "cuda")  # falls back to cpu
    collection_name: str = "cornerstone"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # Chroma returns squared-L2 distance for normalised embeddings; we convert
    # to cosine similarity in retriever.py so all scores are in [0, 1].
    distance_metric: str = "cosine"


@dataclass
class ConfidenceConfig:
    """Thresholds and weights for the two-gate confidence layer.

    The values below are calibrated (Milestone 3) against the labelled SQuAD
    2.0 dev sample: Gate A by Youden's J on answerable/unanswerable, Gate B by
    F1 on the answerable-as-grounded proxy, the composite as the weighted
    harmonic mean of the two chosen operating points. The fitting data and the
    full sweep tables live in reports/calibration.json (run scripts/calibrate.py
    --tag squad2-dev to regenerate).
    """

    # Gate A - retrieval sufficiency, evaluated BEFORE generation.
    # Fitted: AUC 0.652, Youden J 0.24, answerable recall 0.80 at t=0.45.
    gate_a_threshold: float = 0.45
    # Gate B - answer groundedness, evaluated AFTER generation.
    # Fitted: AUC 0.594, F1 0.481 at t=0.25; constrained to stay above the
    # contradiction cap (0.15) so calibration cannot weaken the hard-fail rule.
    gate_b_threshold: float = 0.25
    # Composite score below this is reported as low confidence even if both
    # individual gates squeaked through. Set 10% above the harmonic composite
    # of two both-marginal gate passes: at the corner value itself the band
    # would be empty, because no pair that passed both gates can composite
    # below the corner (the harmonic mean is monotone in both arguments).
    composite_threshold: float = 0.33

    # Weighted harmonic mean weights: (retrieval_relevancy, faithfulness).
    # Faithfulness is weighted higher because a fluent-but-unsupported answer
    # is the failure mode we care most about catching.
    weights: tuple[float, float] = (0.4, 0.6)

    # Scorer backend for Gate A / Gate B.
    #   "nli"  -> local cross-encoders, milliseconds, no API calls
    #   "llm"  -> DeepEval LLM-as-judge, slower but the reference standard
    #   "both" -> run both and log the disagreement (used for the agreement study)
    scorer: Literal["nli", "llm", "both"] = os.getenv("RAG_SCORER", "nli")  # type: ignore[assignment]

    # Gate B backend: natural-language-inference cross-encoder. Premise is a
    # retrieved passage, hypothesis is one claim from the generated answer.
    # RAG_NLI_MODEL may name a local folder, e.g. models/nli-deberta-v3-large.
    nli_model: str = os.getenv("RAG_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
    # What Gate B hands NLI as the hypothesis (see scorers.build_hypothesis):
    #   "claim"    -> the answer sentence alone (Gate B v1, Milestones 1-4)
    #   "qa"       -> question + claim, so a bare answer becomes testable
    #   "qa_short" -> question + claim only for claims of <= 8 words
    # Kept at "claim": question-conditioning helped SQuAD slightly but hurt
    # HotpotQA (reports/gate_b_v2.json), so it does not generalise.
    nli_hypothesis: str = os.getenv("RAG_NLI_HYPOTHESIS", "claim")
    # What NLI uses as the premise:
    #   "passage" -> each retrieved passage whole (Gate B v1, Milestones 1-4)
    #   "window"  -> overlapping 3-sentence windows only
    #   "both"    -> passages and their windows, best support wins (Gate B v2)
    # Windows fix NLI going neutral on long paragraphs; keeping the whole
    # passage too stops a claim that summarises a list from failing because
    # no single window holds the whole list. Evidence: reports/gate_b_v2.json.
    # Set RAG_NLI_PREMISE=passage to reproduce Milestones 1-4.
    nli_premise: str = os.getenv("RAG_NLI_PREMISE", "both")
    # Premise-hypothesis pairs scored per forward pass. Lower it for a large
    # NLI model on an 8 GB GPU (paragraph premises run out of memory at 32).
    nli_batch_size: int = int(os.getenv("RAG_NLI_BATCH_SIZE", "32"))
    # Gate A backend: a relevance reranker, NOT an NLI model. Before generation
    # there is no hypothesis to entail, so the question is "does this passage
    # answer this query", which is a ranking task.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_device: str = os.getenv("RAG_NLI_DEVICE", "cuda")

    # A claim whose contradiction probability exceeds this is logged as
    # RAGTruth "Evident Conflict"; an unsupported non-contradicting claim is
    # logged as "Baseless Information".
    contradiction_threshold: float = 0.50
    # Score a contradicted answer is capped at. Deliberately an absolute value
    # rather than "just under gate_b_threshold": once calibration moves the
    # threshold the old formula would move the penalty with it, and a threshold
    # fitted near zero would turn the cap into a no-op. 0.15 sits below any
    # plausible fitted threshold and is low enough that the harmonic composite
    # collapses, so a contradicted answer can never read as merely borderline.
    contradiction_cap: float = 0.15
    # Below this entailment probability a claim counts as unsupported.
    support_threshold: float = 0.50

    # Margin over entailment required for an un-entailed claim to count as Evident Conflict.
    # A claim is only conflicted if best_contra >= contradiction_threshold AND
    # (best_contra - best_entail) >= contradiction_margin.
    contradiction_margin: float = float(os.getenv("RAG_CONTRADICTION_MARGIN", "0.30"))

    # Minimum word count for a segmented claim. Fragments shorter than this
    # are dropped as non-factual scaffolding.
    min_claim_words: int = int(os.getenv("RAG_MIN_CLAIM_WORDS", "4"))

    # Maximum top passages by Gate A relevance used to generate multi-hop pairs.
    multi_hop_top_k: int = int(os.getenv("RAG_MULTI_HOP_TOP_K", "4"))

    # Abstention message shown to the user when a gate fails.
    abstain_message: str = (
        "I am not confident enough to answer this from the indexed documents. "
        "The retrieved passages do not appear to contain sufficient support."
    )


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)


settings = Settings()
