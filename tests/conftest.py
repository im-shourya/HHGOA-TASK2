"""Shared fixtures.

Unit tests must run without a built index, because the index is derived data and
CI should not need a 5,979-passage corpus to check that plural folding works. The
`rag_index` fixture therefore skips rather than fails when `data/index` is absent,
and everything that can be tested on hand-built inputs is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.chunking.base import Chunk  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.schemas import RetrievedChunk  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def rag_index(settings):
    """The real index, or a skip if it has not been built in this checkout."""
    from app.retrieval.index_store import RagIndex

    if not (settings.index_dir / "chunks.jsonl").exists():
        pytest.skip(f"no index at {settings.index_dir} — run scripts/build_index.py")
    return RagIndex.load(settings.index_dir)


@pytest.fixture(scope="session")
def embedder(settings):
    from app.retrieval.embedder import get_embedder

    return get_embedder(settings.embedding_model)


def make_chunk(
    text: str,
    *,
    chunk_id: str = "c1",
    doc_id: str = "d1",
    parent_id: str = "p1",
    strategy: str = "sentence_window",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        parent_id=parent_id,
        text=text,
        strategy=strategy,
        strategies=[strategy],
        lang="eng_Latn",
        metadata={},
    )


def make_context(
    text: str,
    *,
    chunk_id: str = "c1",
    parent_id: str = "p1",
    score: float = 0.9,
    context_text: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        text=text,
        context_text=context_text or text,
        score=score,
        strategies=["sentence_window"],
        lang="eng_Latn",
        metadata={"doc_id": "d1", "strategy": "sentence_window"},
    )
