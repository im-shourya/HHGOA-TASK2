"""Multi-strategy chunking pipeline: run every splitter, then reconcile.

Three passes:

1. **Split** — each strategy sees every document independently.
2. **Reconcile** — near-duplicate chunks (inevitable when five strategies read
   the same passage) are merged, keeping one copy that remembers every strategy
   that produced it, and each child is linked to the parent window that contains
   it (small-to-big retrieval).
3. **Decorate** — a metadata-aware header is prepended to the *embedded* text
   only: the passage's leading clause plus its top IDF keyphrases, so a chunk
   pulled from the middle of a passage still carries document-level context.

Leakage note: headers are derived strictly from the passage itself. The dataset's
gold query is deliberately never used as chunk metadata — doing so would inflate
retrieval scores on the very queries used for evaluation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from app.chunking.base import Chunk, Chunker, Document
from app.chunking.strategies import (
    EncodeFn,
    FixedWindowChunker,
    PassageChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    SentenceWindowChunker,
)
from app.text import content_tokens, jaccard, split_sentences

# Which strategy's wording survives when two chunks are near-identical.
_PRIORITY = {
    "sentence_window": 0,
    "semantic": 1,
    "fixed_window": 2,
    "recursive_char": 3,
    "passage": 4,
}


@dataclass
class ChunkingStats:
    documents: int = 0
    chunks_before_dedup: int = 0
    chunks: int = 0
    merged: int = 0
    per_strategy: dict[str, int] = field(default_factory=dict)
    words_mean: float = 0.0
    words_p50: float = 0.0
    words_p95: float = 0.0
    words_max: int = 0
    parents: int = 0
    multi_strategy_chunks: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks_before_dedup": self.chunks_before_dedup,
            "chunks": self.chunks,
            "merged_duplicates": self.merged,
            "dedup_ratio": round(
                1 - (self.chunks / self.chunks_before_dedup), 4
            ) if self.chunks_before_dedup else 0.0,
            "per_strategy": self.per_strategy,
            "words": {
                "mean": round(self.words_mean, 1),
                "p50": round(self.words_p50, 1),
                "p95": round(self.words_p95, 1),
                "max": self.words_max,
            },
            "parents": self.parents,
            "multi_strategy_chunks": self.multi_strategy_chunks,
        }


def default_chunkers(encode: EncodeFn | None = None) -> list[Chunker]:
    """The shipped strategy set. `encode` enables semantic splitting."""
    chunkers: list[Chunker] = [
        PassageChunker(max_words=220),
        FixedWindowChunker(window=90, overlap=24),
        SentenceWindowChunker(window=3, stride=1),
        RecursiveCharacterChunker(chunk_size=420, overlap=80),
    ]
    if encode is not None:
        chunkers.append(SemanticChunker(encode=encode, percentile=78.0))
    return chunkers


class ChunkingPipeline:
    """Runs every chunker, merges duplicates, links parents, adds headers."""

    def __init__(
        self,
        chunkers: Sequence[Chunker] | None = None,
        encode: EncodeFn | None = None,
        dedup_threshold: float = 0.92,
        header_words: int = 12,
        keyphrases: int = 3,
    ) -> None:
        self.chunkers = list(chunkers or default_chunkers(encode))
        self.dedup_threshold = dedup_threshold
        self.header_words = header_words
        self.keyphrases = keyphrases
        self.stats = ChunkingStats()

    # ------------------------------------------------------------------ public
    def run(self, documents: Iterable[Document]) -> list[Chunk]:
        documents = list(documents)
        self.stats = ChunkingStats(documents=len(documents))

        raw: list[Chunk] = []
        for doc in documents:
            for chunker in self.chunkers:
                raw.extend(chunker.split(doc))
        self.stats.chunks_before_dedup = len(raw)

        chunks = self._reconcile(raw)
        idf = self._corpus_idf(documents)
        by_doc = {doc.doc_id: doc for doc in documents}
        for chunk in chunks:
            chunk.embed_text = self._decorate(chunk, by_doc.get(chunk.doc_id), idf)
        self._finalise_stats(chunks)
        return chunks

    # ----------------------------------------------------------------- private
    def _reconcile(self, raw: list[Chunk]) -> list[Chunk]:
        """Merge near-duplicates and attach parents, document by document.

        Duplicates only ever arise between chunks of the *same* passage, so the
        quadratic comparison is bounded by the handful of chunks per document.
        Passage-level chunks are never dropped: they are the expansion targets
        for small-to-big retrieval, so a child that merely restates its parent is
        folded into it instead of the other way round.
        """
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in raw:
            grouped[chunk.doc_id].append(chunk)

        kept: list[Chunk] = []
        merged = 0
        for group in grouped.values():
            parents = sorted(
                (c for c in group if c.strategy == "passage"), key=lambda c: c.start
            )
            children = sorted(
                (c for c in group if c.strategy != "passage"),
                key=lambda c: (_PRIORITY.get(c.strategy, 9), c.start),
            )
            survivors: list[tuple[Chunk, set[str]]] = [
                (p, set(content_tokens(p.text))) for p in parents
            ]
            keep_children: list[Chunk] = []
            for chunk in children:
                tokens = set(content_tokens(chunk.text))
                match = next(
                    (
                        s
                        for s, s_tokens in survivors
                        if jaccard(tokens, s_tokens) >= self.dedup_threshold
                    ),
                    None,
                )
                if match is not None:
                    if chunk.strategy not in match.strategies:
                        match.strategies.append(chunk.strategy)
                    merged += 1
                    continue
                survivors.append((chunk, tokens))
                keep_children.append(chunk)

            for chunk in (*parents, *keep_children):
                chunk.parent_id = self._parent_for(chunk, parents)
                kept.append(chunk)
        self.stats.merged = merged
        return kept

    @staticmethod
    def _parent_for(chunk: Chunk, parents: list[Chunk]) -> str:
        if chunk.strategy == "passage" or not parents:
            return chunk.chunk_id if chunk.strategy == "passage" else chunk.doc_id
        midpoint = (chunk.start + chunk.end) // 2
        for parent in parents:
            if parent.start <= midpoint <= parent.end:
                return parent.chunk_id
        return parents[0].chunk_id

    @staticmethod
    def _corpus_idf(documents: list[Document]) -> dict[str, float]:
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(content_tokens(doc.text)))
        total = max(len(documents), 1)
        return {term: math.log(total / (1 + count)) for term, count in df.items()}

    def _decorate(
        self, chunk: Chunk, doc: Document | None, idf: dict[str, float]
    ) -> str:
        """Build the text that gets embedded: context header + chunk body."""
        parts: list[str] = []
        if doc is not None:
            sentences = split_sentences(doc.text)
            if sentences:
                lead = " ".join(sentences[0].split()[: self.header_words])
                if lead and not chunk.text.startswith(lead[: min(len(lead), 20)]):
                    parts.append(lead)
            counts = Counter(content_tokens(doc.text))
            ranked = sorted(
                counts.items(),
                key=lambda kv: kv[1] * idf.get(kv[0], 1.0),
                reverse=True,
            )
            phrases = [term for term, _ in ranked[: self.keyphrases]]
            if phrases:
                parts.append(" ".join(phrases))
        header = " | ".join(parts)
        chunk.metadata["context_header"] = header
        return f"{header} :: {chunk.text}" if header else chunk.text

    def _finalise_stats(self, chunks: list[Chunk]) -> None:
        lengths = np.array([len(c.text.split()) for c in chunks] or [0])
        self.stats.chunks = len(chunks)
        self.stats.per_strategy = dict(Counter(c.strategy for c in chunks))
        self.stats.words_mean = float(lengths.mean())
        self.stats.words_p50 = float(np.percentile(lengths, 50))
        self.stats.words_p95 = float(np.percentile(lengths, 95))
        self.stats.words_max = int(lengths.max())
        self.stats.parents = sum(1 for c in chunks if c.strategy == "passage")
        self.stats.multi_strategy_chunks = sum(1 for c in chunks if len(c.strategies) > 1)


