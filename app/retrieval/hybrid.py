"""Hybrid retrieval: dense + BM25, fused with RRF, diversified with MMR.

Why all three layers:

* **Dense** static embeddings catch paraphrase ("cost of" vs "how much does").
* **BM25** catches the rare literal tokens embeddings smooth away (model numbers,
  years, proper nouns) — and it works on Indic script without a multilingual
  encoder, because a match on "बेंगलुरु" is a match.
* **RRF** fuses the two ranked lists without needing their scores to live on the
  same scale, which is the usual failure of naive weighted-sum hybrids.
* **MMR** then trades a little relevance for coverage, so four contexts are four
  *different* pieces of evidence rather than four copies of the best sentence.

RRF ranks well but says nothing about absolute confidence — its top hit scores the
same whether the corpus contained the answer or not. So `confidence` is computed
from the raw dense cosine and BM25 magnitude instead, and that is what the
"should I answer at all?" guardrail reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.chunking.base import Chunk
from app.retrieval.embedder import Embedder
from app.retrieval.index_store import RagIndex
from app.schemas import RetrievedChunk
from app.text import folded_tokens, lexical_tokens


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    confidence: float
    dense_top: float
    sparse_top: float
    margin: float
    candidates: int
    detail: dict[str, float | int]
    coverage: float = 0.0


def _saturate(value: float, half: float = 6.0) -> float:
    """Map an unbounded BM25 score into (0, 1) with `half` as the midpoint."""
    return float(value / (value + half)) if value > 0 else 0.0


class HybridRetriever:
    def __init__(
        self,
        index: RagIndex,
        embedder: Embedder,
        *,
        dense_top_k: int = 40,
        sparse_top_k: int = 40,
        fusion_top_k: int = 12,
        context_top_k: int = 4,
        rrf_k: int = 60,
        mmr_lambda: float = 0.72,
        one_per_parent: bool = True,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fusion_top_k = fusion_top_k
        self.context_top_k = context_top_k
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.one_per_parent = one_per_parent
        if index.vector_store is None:
            index.attach_vector_store()

    # ------------------------------------------------------------------ public
    def retrieve(self, query: str, query_vector: np.ndarray | None = None,
                 top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.context_top_k
        if query_vector is None:
            query_vector = self.embedder.encode([query])[0]

        assert self.index.vector_store is not None
        dense_idx, dense_scores = self.index.vector_store.search(
            query_vector, self.dense_top_k
        )
        sparse_idx, sparse_scores = self.index.bm25.search(
            lexical_tokens(query), self.sparse_top_k
        )

        fused = self._reciprocal_rank_fusion(
            dense_idx, dense_scores, sparse_idx, sparse_scores
        )
        if not fused:
            return RetrievalResult([], 0.0, 0.0, 0.0, 0.0, 0, {"fused": 0})
        ranked = sorted(fused.values(), key=lambda r: -r["rrf"])[: self.fusion_top_k]
        selected = self._mmr(ranked, query_vector, top_k)
        contexts = [self._to_model(r) for r in selected]

        dense_top = float(dense_scores[0]) if dense_scores.size else 0.0
        sparse_top = float(sparse_scores[0]) if sparse_scores.size else 0.0
        second = float(dense_scores[1]) if dense_scores.size > 1 else 0.0
        coverage = self._focus_coverage(query, contexts)
        # Three independent signals, because each fails differently: cosine is
        # fooled by topical-but-irrelevant text, BM25 by common words, and term
        # coverage by paraphrase. Agreement between them is what confidence means.
        confidence = (
            0.45 * max(dense_top, 0.0) + 0.25 * _saturate(sparse_top) + 0.30 * coverage
        )

        return RetrievalResult(
            chunks=contexts,
            confidence=round(confidence, 4),
            dense_top=round(dense_top, 4),
            sparse_top=round(sparse_top, 4),
            margin=round(dense_top - second, 4),
            candidates=len(fused),
            coverage=round(coverage, 4),
            detail={
                "dense_hits": int(dense_idx.size),
                "sparse_hits": int(sparse_idx.size),
                "fused_candidates": len(fused),
                "returned": len(selected),
                "coverage": round(coverage, 4),
            },
        )

    @staticmethod
    def _focus_coverage(query: str, contexts: list[RetrievedChunk]) -> float:
        """Fraction of the query's content terms that appear in the contexts.

        Catches the case cosine similarity cannot: text about the right *topic*
        that never mentions what was actually asked.
        """
        focus = set(folded_tokens(query))
        if not focus:
            return 0.0
        pooled: set[str] = set()
        for context in contexts:
            pooled |= set(folded_tokens(context.context_text or context.text))
        return len(focus & pooled) / len(focus)

    # ----------------------------------------------------------------- private
    def _reciprocal_rank_fusion(
        self,
        dense_idx: np.ndarray,
        dense_scores: np.ndarray,
        sparse_idx: np.ndarray,
        sparse_scores: np.ndarray,
    ) -> dict[int, dict]:
        fused: dict[int, dict] = {}
        for rank, (position, score) in enumerate(zip(dense_idx, dense_scores), start=1):
            fused[int(position)] = {
                "idx": int(position),
                "rrf": 1.0 / (self.rrf_k + rank),
                "dense_score": float(score),
                "dense_rank": rank,
                "sparse_score": None,
                "sparse_rank": None,
            }
        for rank, (position, score) in enumerate(zip(sparse_idx, sparse_scores), start=1):
            entry = fused.setdefault(
                int(position),
                {
                    "idx": int(position),
                    "rrf": 0.0,
                    "dense_score": None,
                    "dense_rank": None,
                    "sparse_score": None,
                    "sparse_rank": None,
                },
            )
            entry["rrf"] += 1.0 / (self.rrf_k + rank)
            entry["sparse_score"] = float(score)
            entry["sparse_rank"] = rank
        # Normalise so a chunk ranked #1 by both retrievers scores exactly 1.0.
        ceiling = 2.0 / (self.rrf_k + 1)
        for entry in fused.values():
            entry["fused"] = min(entry["rrf"] / ceiling, 1.0)
        return fused

    def _mmr(self, ranked: list[dict], query_vector: np.ndarray, top_k: int) -> list[dict]:
        """Maximal Marginal Relevance over the fused candidates."""
        if not ranked:
            return []
        positions = [r["idx"] for r in ranked]
        vectors = self.index.vectors[positions]
        relevance = vectors @ np.asarray(query_vector, dtype=np.float32).reshape(-1)
        chosen: list[int] = []
        used_parents: set[str] = set()
        one_per_parent = self.one_per_parent
        while len(chosen) < min(top_k, len(ranked)):
            best, best_score = -1, -np.inf
            for i in range(len(ranked)):
                if i in chosen:
                    continue
                parent = self.index.chunks[positions[i]].parent_id
                if one_per_parent and parent in used_parents:
                    continue
                penalty = (
                    max(float(vectors[i] @ vectors[j]) for j in chosen) if chosen else 0.0
                )
                score = self.mmr_lambda * float(relevance[i]) - (1 - self.mmr_lambda) * penalty
                if score > best_score:
                    best, best_score = i, score
            if best < 0:  # every remaining candidate shares a parent we already used
                if not one_per_parent:
                    break
                one_per_parent = False  # relax the constraint rather than return short
                continue
            chosen.append(best)
            used_parents.add(self.index.chunks[positions[best]].parent_id)
        return [ranked[i] for i in chosen]

    def _to_model(self, entry: dict) -> RetrievedChunk:
        chunk: Chunk = self.index.chunks[entry["idx"]]
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            parent_id=chunk.parent_id,
            text=chunk.text,
            context_text=self.index.context_for(chunk),
            score=round(float(entry["fused"]), 4),
            dense_score=entry["dense_score"],
            sparse_score=entry["sparse_score"],
            dense_rank=entry["dense_rank"],
            sparse_rank=entry["sparse_rank"],
            strategies=list(chunk.strategies),
            lang=chunk.lang,
            metadata={
                "doc_id": chunk.doc_id,
                "strategy": chunk.strategy,
                **{
                    k: v
                    for k, v in chunk.metadata.items()
                    if k in {"source", "title", "query_type", "url", "is_selected"}
                },
            },
        )

