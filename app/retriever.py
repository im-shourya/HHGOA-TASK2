"""Retrieval as a flat, timed function — the surface the reference harness expects.

The task shipped a benchmark harness that imports this module:

    from app.config import LATENCY_BUDGET_MS
    from app.retriever import search, warmup

    resp = search(query, top_k=5)
    resp.total_ms, resp.embed_ms, resp.search_ms

Everything real lives in `app.retrieval.*` and is driven through
`app.harness.orchestrator.RagPipeline`; this module is a deliberately thin
process-singleton adapter over it, so the shipped harness runs unmodified against
the actual index instead of a parallel toy implementation.

Two things it is careful about:

* **`search()` is the retrieval primitive, not the pipeline.** No guardrails, no
  answer composition, no grounding check. It measures what it claims to measure.
  Use `RagPipeline.answer()` for anything user-facing — refusing to answer is a
  pipeline decision, and this function cannot make it.
* **`search_ms` is the whole hybrid stage**, not just the dense ANN lookup: dense
  cosine + BM25+ + reciprocal-rank fusion + MMR diversification. Reporting only
  the vector-store call would flatter the number by hiding two thirds of the work.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.index_store import RagIndex

logger = logging.getLogger(__name__)

WARMUP_PROBES = (
    "average cost of a replacement windshield",
    "how fast does an eagle travel",
    "what is a corporation",
)


@dataclass(slots=True)
class Hit:
    """One retrieved context, flattened for reporting."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    strategy: str
    dense_rank: int | None = None
    sparse_rank: int | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience for REPL/CLI use
        return f"[{self.score:.3f}] {self.text[:110]}"


@dataclass(slots=True)
class SearchResponse:
    """Result of one retrieval, with the timings split the way the budget is."""

    query: str
    hits: list[Hit]
    embed_ms: float
    search_ms: float
    total_ms: float
    confidence: float
    dense_top: float
    sparse_top: float
    coverage: float
    candidates: int
    backend: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def within_budget(self) -> bool:
        from app.config import LATENCY_BUDGET_MS

        return self.total_ms <= LATENCY_BUDGET_MS

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "doc_id": h.doc_id,
                    "score": h.score,
                    "strategy": h.strategy,
                    "text": h.text,
                }
                for h in self.hits
            ],
            "embed_ms": self.embed_ms,
            "search_ms": self.search_ms,
            "total_ms": self.total_ms,
            "confidence": self.confidence,
            "backend": self.backend,
        }


class Retriever:
    """Index + embedder + hybrid search, loaded once and reused."""

    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or get_settings()
        self.index = RagIndex.load(self.settings.index_dir)
        self.embedder = embedder or get_embedder(self.settings.embedding_model)
        self.hybrid = HybridRetriever(
            self.index,
            self.embedder,
            dense_top_k=self.settings.dense_top_k,
            sparse_top_k=self.settings.sparse_top_k,
            fusion_top_k=self.settings.fusion_top_k,
            context_top_k=self.settings.context_top_k,
            rrf_k=self.settings.rrf_k,
            mmr_lambda=self.settings.mmr_lambda,
        )

    @property
    def backend(self) -> str:
        store = self.index.vector_store
        return store.backend if store else "unset"

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        started = time.perf_counter()
        vector: np.ndarray = self.embedder.encode([query])[0]
        embedded = time.perf_counter()
        result = self.hybrid.retrieve(query, vector, top_k=top_k)
        finished = time.perf_counter()

        embed_ms = (embedded - started) * 1000
        search_ms = (finished - embedded) * 1000
        return SearchResponse(
            query=query,
            hits=[
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=str(chunk.metadata.get("doc_id", "")),
                    text=chunk.text,
                    score=chunk.score,
                    strategy=str(chunk.metadata.get("strategy", "")),
                    dense_rank=chunk.dense_rank,
                    sparse_rank=chunk.sparse_rank,
                )
                for chunk in result.chunks
            ],
            embed_ms=round(embed_ms, 4),
            search_ms=round(search_ms, 4),
            total_ms=round(embed_ms + search_ms, 4),
            confidence=result.confidence,
            dense_top=result.dense_top,
            sparse_top=result.sparse_top,
            coverage=result.coverage,
            candidates=result.candidates,
            backend=self.backend,
            detail=dict(result.detail),
        )

    def warmup(self, rounds: int = 3) -> float:
        """Pay every first-call cost before anything is timed.

        A cold process answers its first query in ~240 ms and every one after it in
        single-digit milliseconds: BLAS selects its kernels, numpy allocates scratch
        buffers, the tokeniser fills its caches. Timing without this reports a number
        the system does not have — in either direction.
        """
        started = time.perf_counter()
        for i in range(max(rounds, 1)):
            self.search(WARMUP_PROBES[i % len(WARMUP_PROBES)], top_k=self.settings.context_top_k)
        return round((time.perf_counter() - started) * 1000, 2)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide singleton. Raises FileNotFoundError if the index is not built."""
    retriever = get_retriever_uncached()
    logger.info(
        "retriever ready: %d chunks / %d passages, backend=%s, model=%s",
        retriever.index.size,
        retriever.index.n_passages,
        retriever.backend,
        retriever.embedder.name,
    )
    return retriever


def get_retriever_uncached() -> Retriever:
    return Retriever()


def search(query: str, top_k: int = 5) -> SearchResponse:
    """Retrieve `top_k` contexts for `query`. Loads the index on first call."""
    return get_retriever().search(query, top_k=top_k)


def warmup(rounds: int = 3) -> float:
    """Load the index and touch every hot path. Returns milliseconds spent."""
    return get_retriever().warmup(rounds=rounds)


def stats() -> dict[str, Any]:
    retriever = get_retriever()
    return {
        "chunks": retriever.index.size,
        "passages": retriever.index.n_passages,
        "embedding_model": retriever.embedder.name,
        "embedding_dim": retriever.embedder.dim,
        "vector_backend": retriever.backend,
        "bm25_vocab": retriever.index.bm25.vocab_size,
    }
