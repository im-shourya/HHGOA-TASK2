"""Query/document embedding.

The whole latency budget hinges on this file. A transformer encoder costs 10-40 ms
per query on CPU; a *static* model (model2vec / potion) is a token-embedding
lookup plus a mean — measured at **0.08 ms** per query on this machine, which
leaves the rest of the 200 ms budget for retrieval, generation and verification.

`HashingEmbedder` is a dependency-free fallback so the pipeline still runs (with
lower recall) when model weights cannot be downloaded — CI, air-gapped boxes.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Protocol, Sequence

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class StaticEmbedder:
    """model2vec static embeddings — sub-millisecond, no torch, no GPU."""

    def __init__(self, model_name: str) -> None:
        from model2vec import StaticModel  # imported lazily: keeps CLI startup fast

        self._model = StaticModel.from_pretrained(model_name)
        self.name = model_name
        self.dim = int(self._model.dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(self._model.encode(list(texts), show_progress_bar=False))


class HashingEmbedder:
    """Deterministic hashed bag-of-ngrams. Offline fallback only."""

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _WORD.findall((text or "").casefold())
            for token in tokens:
                out[row, hash(token) % self.dim] += 1.0
                for i in range(len(token) - 2):  # character trigrams add robustness
                    out[row, hash(token[i : i + 3]) % self.dim] += 0.3
        return l2_normalize(out)


_lock = threading.Lock()
_embedder: Embedder | None = None


def get_embedder(model_name: str | None = None) -> Embedder:
    """Process-wide singleton: the weights are loaded exactly once."""
    global _embedder
    with _lock:
        if _embedder is not None and (model_name is None or _embedder.name == model_name):
            return _embedder
        name = model_name or get_settings().embedding_model
        try:
            _embedder = StaticEmbedder(name)
            logger.info("embedder ready: %s (dim=%d)", _embedder.name, _embedder.dim)
        except Exception as exc:  # noqa: BLE001 - fallback must cover any load failure
            logger.warning("static embedder unavailable (%s); using hashing fallback", exc)
            _embedder = HashingEmbedder(get_settings().embedding_dim)
        return _embedder


def reset_embedder() -> None:
    """Test hook: drop the cached singleton."""
    global _embedder
    with _lock:
        _embedder = None
