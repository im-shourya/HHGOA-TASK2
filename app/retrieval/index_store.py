"""On-disk index artifacts and the in-memory index they load into.

Layout of `data/index/`:

    chunks.jsonl        one reconciled chunk per line (text, parent, strategies)
    embeddings.npy      float32 [n_chunks, dim], L2-normalised, row-aligned to chunks
    bm25.npz            flat postings arrays + doc lengths
    bm25_meta.json      vocabulary and BM25+ hyper-parameters
    manifest.json       provenance: model, dims, corpus fingerprint, chunk stats

Everything is derived data: delete the directory and `scripts/build_index.py`
rebuilds it byte-for-byte from the corpus.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.chunking.base import Chunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_store import (
    VectorStore,
    build_vector_store,
    load_vectors,
    save_vectors,
)
from app.text import lexical_tokens

logger = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass
class RagIndex:
    """Everything retrieval needs, resident in memory."""

    chunks: list[Chunk]
    vectors: np.ndarray
    bm25: BM25Index
    manifest: dict[str, Any] = field(default_factory=dict)
    vector_store: VectorStore | None = None
    _by_id: dict[str, int] = field(default_factory=dict, repr=False)
    _parent_text: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}
        self._parent_text = {
            chunk.chunk_id: chunk.text
            for chunk in self.chunks
            if chunk.strategy == "passage"
        }

    # ------------------------------------------------------------------ access
    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def n_passages(self) -> int:
        return len({chunk.doc_id for chunk in self.chunks})

    def index_of(self, chunk_id: str) -> int | None:
        return self._by_id.get(chunk_id)

    def context_for(self, chunk: Chunk) -> str:
        """Small-to-big expansion: the parent window if we have one, else the chunk."""
        parent = self._parent_text.get(chunk.parent_id)
        if not parent:
            return chunk.text
        return parent if len(parent) >= len(chunk.text) else chunk.text

    def attach_vector_store(self, backend: str = "auto", threshold: int = 120_000) -> None:
        self.vector_store = build_vector_store(
            self.vectors, backend=backend, hnsw_threshold=threshold
        )

    # ---------------------------------------------------------------- persist
    def save(self, directory: Path, extra_manifest: dict[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n")
        save_vectors(directory, self.vectors)
        self.bm25.save(directory)
        manifest = {
            **self.manifest,
            **(extra_manifest or {}),
            "chunks": self.size,
            "passages": self.n_passages,
            "dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
            "bm25_vocab": self.bm25.vocab_size,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (directory / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.manifest = manifest
        logger.info("index saved to %s (%d chunks)", directory, self.size)

    @classmethod
    def load(cls, directory: Path) -> "RagIndex":
        directory = Path(directory)
        chunks_path = directory / CHUNKS_FILE
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"no index at {directory} — run scripts/build_index.py first"
            )
        with chunks_path.open("r", encoding="utf-8") as handle:
            chunks = [Chunk.from_record(json.loads(line)) for line in handle if line.strip()]
        manifest_path = directory / MANIFEST_FILE
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        return cls(
            chunks=chunks,
            vectors=load_vectors(directory),
            bm25=BM25Index.load(directory),
            manifest=manifest,
        )

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        manifest: dict[str, Any] | None = None,
    ) -> "RagIndex":
        bm25 = BM25Index().build([lexical_tokens(chunk.embed_text) for chunk in chunks])
        return cls(
            chunks=list(chunks),
            vectors=np.asarray(vectors, dtype=np.float32),
            bm25=bm25,
            manifest=dict(manifest or {}),
        )
