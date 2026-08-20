"""Retrieval mechanics: BM25+, reciprocal-rank fusion, MMR.

These run on hand-built inputs rather than the real index, so they pin the
*algorithms* — including the RRF property that caused the eagle bug (it fuses
ranks, not scores, so a mediocre-on-both-lists candidate can displace a dense #1
that is missing from the sparse list).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.text import lexical_tokens, tokenize
from tests.conftest import make_chunk


@pytest.fixture
def corpus():
    return [
        "Eagles fly 30 to 55 mph and dive at over 100 mph",
        "Travel must occur between January and May on the Texas Eagle",
        "A corporation is a legal entity separate from its owners",
        "Corn meal and corn flour are not the same thing",
    ]


class TestBm25:
    def test_scores_the_matching_document_highest(self, corpus):
        index = BM25Index().build([lexical_tokens(d) for d in corpus])
        scores = index.scores(lexical_tokens("corporation legal entity"))
        assert int(np.argmax(scores)) == 2

    def test_plural_folding_rescues_a_zero_score(self, corpus):
        """The measured regression: "eagle" against a document saying "Eagles".

        Unfolded, the document that literally answers the query scores exactly
        0.0 — which is what pushed it to sparse rank 871 and out of the fused
        context set entirely.
        """
        unfolded = BM25Index().build([tokenize(d) for d in corpus])
        folded = BM25Index().build([lexical_tokens(d) for d in corpus])
        assert unfolded.scores(tokenize("how fast does an eagle travel"))[0] == 0.0
        assert folded.scores(lexical_tokens("how fast does an eagle travel"))[0] > 0.0

    def test_search_returns_descending_scores(self, corpus):
        index = BM25Index().build([lexical_tokens(d) for d in corpus])
        _, scores = index.search(lexical_tokens("corn flour"), top_k=4)
        assert list(scores) == sorted(scores, reverse=True)

    def test_unknown_terms_score_zero_rather_than_raising(self, corpus):
        index = BM25Index().build([lexical_tokens(d) for d in corpus])
        assert index.scores(lexical_tokens("xyzzy plugh")).sum() == 0.0

    def test_empty_query_is_safe(self, corpus):
        index = BM25Index().build([lexical_tokens(d) for d in corpus])
        assert index.scores([]).sum() == 0.0

    def test_roundtrips_through_disk(self, corpus, tmp_path):
        original = BM25Index().build([lexical_tokens(d) for d in corpus])
        original.save(tmp_path)
        loaded = BM25Index.load(tmp_path)
        query = lexical_tokens("corporation legal entity")
        assert np.allclose(original.scores(query), loaded.scores(query))
        assert loaded.vocab_size == original.vocab_size


class TestReciprocalRankFusion:
    def _fuse(self, dense, sparse):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60
        return retriever._reciprocal_rank_fusion(
            np.array([d[0] for d in dense]), np.array([d[1] for d in dense], dtype=np.float32),
            np.array([s[0] for s in sparse]), np.array([s[1] for s in sparse], dtype=np.float32),
        )

    def test_agreement_between_retrievers_scores_one(self):
        fused = self._fuse([(7, 0.9)], [(7, 12.0)])
        assert fused[7]["fused"] == pytest.approx(1.0)

    def test_a_candidate_on_both_lists_beats_a_better_one_on_only_a_single_list(self):
        """The eagle bug, as an algorithmic property rather than an anecdote.

        Document 1 is dense #0 but absent from the sparse list; document 2 is
        merely 3rd and 2nd. RRF still prefers document 2 — which is why the
        missing BM25 score, not the fusion, was the thing to fix.
        """
        fused = self._fuse(
            [(1, 0.95), (5, 0.80), (2, 0.70)],
            [(9, 20.0), (2, 15.0), (8, 9.0)],
        )
        assert fused[2]["fused"] > fused[1]["fused"]

    def test_records_both_ranks_for_explainability(self):
        fused = self._fuse([(3, 0.5)], [(3, 4.0)])
        assert fused[3]["dense_rank"] == 1 and fused[3]["sparse_rank"] == 1
        assert fused[3]["dense_score"] is not None
        assert fused[3]["sparse_score"] is not None

    def test_dense_only_candidate_keeps_a_null_sparse_score(self):
        fused = self._fuse([(4, 0.6)], [(9, 3.0)])
        assert fused[4]["sparse_score"] is None
        assert fused[4]["sparse_rank"] is None


class TestMmrOnePerParent:
    """MMR must return four *different* pieces of evidence, not four copies."""

    def _retriever(self, parents, vectors):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.mmr_lambda = 0.72
        retriever.one_per_parent = True

        class _Index:
            pass

        index = _Index()
        index.chunks = [
            make_chunk(f"chunk {i}", chunk_id=f"c{i}", parent_id=p)
            for i, p in enumerate(parents)
        ]
        index.vectors = np.asarray(vectors, dtype=np.float32)
        retriever.index = index
        return retriever

    def test_picks_one_chunk_per_parent(self):
        vectors = np.eye(4, dtype=np.float32)
        retriever = self._retriever(["p1", "p1", "p2", "p3"], vectors)
        ranked = [{"idx": i, "rrf": 1.0 - 0.1 * i} for i in range(4)]
        chosen = retriever._mmr(ranked, np.array([1, 0, 0, 0], dtype=np.float32), 3)
        parents = {retriever.index.chunks[c["idx"]].parent_id for c in chosen}
        assert len(parents) == len(chosen)

    def test_relaxes_the_constraint_rather_than_returning_short(self):
        """Four contexts were asked for and only two parents exist."""
        vectors = np.eye(4, dtype=np.float32)
        retriever = self._retriever(["p1", "p1", "p2", "p2"], vectors)
        ranked = [{"idx": i, "rrf": 1.0 - 0.1 * i} for i in range(4)]
        chosen = retriever._mmr(ranked, np.array([1, 0, 0, 0], dtype=np.float32), 4)
        assert len(chosen) == 4

    def test_empty_candidates(self):
        retriever = self._retriever([], np.zeros((0, 4), dtype=np.float32))
        assert retriever._mmr([], np.array([1, 0, 0, 0], dtype=np.float32), 4) == []
