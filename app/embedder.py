"""Adapter: the eval-loop embedder contract, over this project's real embedder.

`rag-local-eval-loop` (TARGET_INTERFACE.md) verifies a target by importing a
module and checking for `embed` / `embed_one` / `get_model`. This project's
embedder is a class with an `encode` method behind a process-wide singleton
(app/retrieval/embedder.py), so this module is a thin renaming shim — not a
second implementation. It loads the same `potion-retrieval-32M` weights the
served pipeline uses, so the suite's Recall/latency numbers describe the real
retrieval path rather than something built only for grading.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from app.retrieval.embedder import get_embedder


def get_model() -> Any:
    """Load the weights once. The suite calls this for its side effect only."""
    return get_embedder()


def embed(texts: Sequence[str]) -> np.ndarray:
    """(n, dim) L2-normalised float32 — the same vectors the index was built on."""
    return get_embedder().encode(list(texts))


def embed_one(text: str) -> np.ndarray:
    """(dim,) for a single string. Must support .reshape(1, -1) and .shape[-1]."""
    return embed([text])[0]
