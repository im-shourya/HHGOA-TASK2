"""Vector store backends.

`flat` is exact brute-force cosine search on a contiguous float32 matrix. At this
corpus size that is a single BLAS `sgemv` — around 1 ms for tens of thousands of
chunks — and it is *exact*, so no recall is traded away for speed.

`hnsw` (optional, via hnswlib) kicks in automatically past `hnsw_threshold`
chunks, where brute force would start eating the latency budget. Same interface,
so the rest of the pipeline never learns which one it is talking to.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    backend: str
    size: int
    dim: int

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...


class FlatVectorStore:
    """Exact cosine similarity. Vectors must already be L2-normalised."""

    backend = "flat"

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.size, self.dim = (self.vectors.shape if self.vectors.ndim == 2 else (0, 0))
        self._scores = np.zeros(self.size, dtype=np.float32)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.size:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        np.dot(self.vectors, query, out=self._scores)
        top_k = min(top_k, self.size)
        candidates = np.argpartition(-self._scores, top_k - 1)[:top_k]
        candidates = candidates[np.argsort(-self._scores[candidates], kind="stable")]
        return candidates, self._scores[candidates].copy()


class HnswVectorStore:
    """Approximate HNSW graph search for corpora too large to scan exactly."""

    backend = "hnsw"

    def __init__(self, vectors: np.ndarray, ef_search: int = 96, m: int = 24) -> None:
        import hnswlib  # optional dependency

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.size, self.dim = vectors.shape
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._index.init_index(max_elements=self.size, ef_construction=200, M=m)
        self._index.add_items(vectors, np.arange(self.size))
        self._index.set_ef(ef_search)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.size:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        labels, distances = self._index.knn_query(query, k=min(top_k, self.size))
        return labels[0].astype(np.int64), (1.0 - distances[0]).astype(np.float32)


def build_vector_store(
    vectors: np.ndarray, backend: str = "auto", hnsw_threshold: int = 120_000
) -> VectorStore:
    """Pick a backend. `auto` = exact until the corpus outgrows the budget."""
    wants_hnsw = backend == "hnsw" or (
        backend == "auto" and len(vectors) >= hnsw_threshold
    )
    if wants_hnsw:
        try:
            store = HnswVectorStore(vectors)
            logger.info("vector store: hnsw (%d vectors)", store.size)
            return store
        except ImportError:
            logger.warning("hnswlib not installed; falling back to exact flat search")
    store = FlatVectorStore(vectors)
    logger.info("vector store: flat/exact (%d vectors)", store.size)
    return store


def save_vectors(directory: Path, vectors: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", np.asarray(vectors, dtype=np.float32))
    (directory / "embeddings_meta.json").write_text(
        json.dumps({"count": int(len(vectors)), "dim": int(vectors.shape[1])}),
        encoding="utf-8",
    )


def load_vectors(directory: Path) -> np.ndarray:
    return np.load(directory / "embeddings.npy", mmap_mode=None)
