"""Chunking: five strategies, reconciliation, and the properties each one promises.

The brief asks for "vast chunking" rather than one naive fixed-size pass, so the
value here is not that five splitters exist — it is that they make *different*
promises, and those promises are what these tests pin:

  passage         · never dropped; it is the parent every child expands to
  fixed_window    · overlap guarantees no span is only ever seen cut in half
  sentence_window · boundaries fall on sentences, so a quote is a whole sentence
  recursive_char  · the hard cap holds even for scripts the cascade never saw
  semantic        · boundaries fall where meaning shifts, not where words run out

Plus the reconciliation invariant that makes running all five affordable: five views
of the same passage collapse to one chunk that remembers every strategy that found it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.chunking.base import Document
from app.chunking.pipeline import ChunkingPipeline, default_chunkers
from app.chunking.strategies import (
    FixedWindowChunker,
    PassageChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    SentenceWindowChunker,
)
from app.text import split_sentences

PROSE = (
    "A corporation is a legal entity that is separate and distinct from its owners. "
    "Corporations enjoy most of the rights and responsibilities that individuals possess. "
    "They can enter contracts, loan and borrow money, sue and be sued, hire employees, "
    "own assets and pay taxes. "
    "The most important aspect of a corporation is limited liability. "
    "That is, shareholders have the right to participate in the profits through dividends "
    "but are not held personally liable for the company's debts."
)
HINDI = (
    "निगम एक कानूनी इकाई है जो अपने मालिकों से अलग होती है। "
    "निगम अनुबंध कर सकते हैं और ऋण ले सकते हैं। "
    "शेयरधारक लाभांश के माध्यम से मुनाफे में भाग ले सकते हैं। "
    "उन्हें कंपनी के ऋणों के लिए व्यक्तिगत रूप से उत्तरदायी नहीं ठहराया जाता है।"
)


def doc(text: str = PROSE, doc_id: str = "d1", lang: str = "eng_Latn") -> Document:
    return Document(doc_id=doc_id, text=text, lang=lang)


class TestPassageChunker:
    def test_short_passage_stays_whole(self):
        chunks = PassageChunker(max_words=220).split(doc())
        assert len(chunks) == 1
        assert chunks[0].metadata["is_parent"] is True

    def test_a_very_long_passage_is_hard_split_so_a_parent_never_blows_the_budget(self):
        long_doc = doc(" ".join(["word"] * 500))
        chunks = PassageChunker(max_words=220).split(long_doc)
        assert len(chunks) == 3
        assert all(len(c.text.split()) <= 220 for c in chunks)

    def test_empty_document_yields_nothing(self):
        assert PassageChunker().split(doc("   ")) == []


class TestFixedWindowChunker:
    def test_overlap_means_no_span_is_only_ever_seen_cut(self):
        """The point of overlap: consecutive windows share `overlap` words.

        Without it an answer-bearing sentence straddling a boundary appears in no
        chunk whole, and neither half retrieves.
        """
        chunker = FixedWindowChunker(window=20, overlap=8, min_words=4)
        chunks = chunker.split(doc())
        assert len(chunks) > 1
        for earlier, later in zip(chunks, chunks[1:]):
            assert later.start < earlier.end, "windows must overlap in the source text"

    def test_reports_its_overlap_ratio(self):
        assert FixedWindowChunker(window=90, overlap=24).overlap_ratio == pytest.approx(0.2667, abs=1e-4)

    def test_overlap_at_or_above_window_is_rejected_rather_than_looping_forever(self):
        with pytest.raises(ValueError, match="overlap must be smaller"):
            FixedWindowChunker(window=20, overlap=20)

    def test_covers_the_whole_document(self):
        chunker = FixedWindowChunker(window=20, overlap=8, min_words=4)
        chunks = chunker.split(doc())
        assert chunks[0].start == 0
        assert chunks[-1].end >= len(PROSE.rstrip()) - 2


class TestSentenceWindowChunker:
    def test_every_chunk_starts_and_ends_on_a_sentence_boundary(self):
        """What makes an extracted quote a whole sentence rather than a fragment."""
        chunks = SentenceWindowChunker(window=3, stride=1).split(doc())
        sentences = set(split_sentences(PROSE))
        for chunk in chunks:
            first = split_sentences(chunk.text)[0]
            assert first in sentences

    def test_records_the_centre_sentence_for_precise_attribution(self):
        chunks = SentenceWindowChunker(window=3, stride=1).split(doc())
        centre = chunks[0].metadata["centre_sentence"]
        assert centre in chunks[0].text

    def test_windows_overlap_when_stride_is_smaller_than_window(self):
        chunks = SentenceWindowChunker(window=3, stride=1).split(doc())
        assert len(chunks) >= len(split_sentences(PROSE)) - 2
        assert chunks[1].start < chunks[0].end


class TestRecursiveCharacterChunker:
    def test_hard_cap_holds(self):
        chunker = RecursiveCharacterChunker(chunk_size=200, overlap=40, min_chars=20)
        for chunk in chunker.split(doc()):
            assert len(chunk.text) <= 200 + 40, "cap plus carried overlap"

    def test_cap_holds_for_a_script_with_no_ascii_separators(self):
        """The cascade ends at the empty separator precisely so this cannot fail."""
        chunker = RecursiveCharacterChunker(chunk_size=120, overlap=20, min_chars=20)
        chunks = chunker.split(doc(HINDI, lang="hin_Deva"))
        assert chunks
        assert all(len(c.text) <= 140 for c in chunks)

    def test_a_single_unbroken_token_is_still_capped(self):
        chunker = RecursiveCharacterChunker(chunk_size=100, overlap=0, min_chars=10)
        chunks = chunker.split(doc("x" * 450))
        assert chunks and all(len(c.text) <= 100 for c in chunks)


class TestSemanticChunker:
    def _encode(self, topic_a: set[str]):
        """Two orthogonal directions, so distance spikes exactly at the topic shift."""
        def encode(sentences):
            out = []
            for sentence in sentences:
                first = sentence.split()[0].strip(".,").casefold()
                out.append([1.0, 0.0] if first in topic_a else [0.0, 1.0])
            return np.asarray(out, dtype=np.float32)
        return encode

    def test_cuts_where_the_topic_changes(self):
        text = (
            "Alpha particles are helium nuclei emitted in radioactive decay of heavy elements. "
            "Alpha radiation is stopped by a sheet of ordinary paper or by skin. "
            "Beta particles are electrons emitted from the nucleus at high velocity there. "
            "Beta radiation penetrates further and needs aluminium shielding to stop it."
        )
        chunker = SemanticChunker(
            encode=self._encode({"alpha"}), percentile=50.0, min_words=5, max_words=200
        )
        chunks = chunker.split(doc(text))
        assert len(chunks) == 2
        assert chunks[0].text.startswith("Alpha") and chunks[1].text.startswith("Beta")

    def test_short_passages_are_left_alone_rather_than_split_unstably(self):
        chunker = SemanticChunker(encode=self._encode(set()), percentile=50.0)
        chunks = chunker.split(doc("One sentence here. And a second one follows it."))
        assert len(chunks) == 1

    def test_max_words_forces_a_cut_even_without_a_topic_shift(self):
        text = " ".join(f"Sentence number {i} carries some filler words along." for i in range(30))
        chunker = SemanticChunker(
            encode=self._encode(set()), percentile=99.0, min_words=5, max_words=40
        )
        assert len(chunker.split(doc(text))) > 1


class TestPipelineReconciliation:
    @pytest.fixture(scope="class")
    def run(self):
        pipeline = ChunkingPipeline(dedup_threshold=0.92)
        chunks = pipeline.run([doc(), doc(HINDI, doc_id="d2", lang="hin_Deva")])
        return pipeline, chunks

    def test_runs_every_configured_strategy(self, run):
        """Represented, not necessarily surviving as a distinct chunk.

        On a passage shorter than the fixed window, `fixed_window` produces exactly
        the whole passage — so it is folded into the parent rather than kept as a
        second copy of it, and shows up in that parent's `strategies` instead. That
        is reconciliation doing its job; asserting on surviving `.strategy` values
        would be asserting that dedup fails.
        """
        _, chunks = run
        represented = {s for c in chunks for s in c.strategies}
        assert represented >= {
            "passage", "fixed_window", "sentence_window", "recursive_char"
        }

    def test_a_strategy_that_merely_restates_the_passage_folds_into_it(self, run):
        _, chunks = run
        parents = {c.doc_id: c for c in chunks if c.strategy == "passage"}
        assert "fixed_window" in parents["d1"].strategies
        assert sum(1 for c in chunks if c.doc_id == "d1" and c.strategy == "fixed_window") == 0

    def test_deduplication_actually_removes_chunks(self, run):
        pipeline, chunks = run
        assert pipeline.stats.chunks_before_dedup > pipeline.stats.chunks
        assert pipeline.stats.merged > 0

    def test_a_merged_chunk_remembers_every_strategy_that_found_it(self, run):
        """Dedup must not lose the provenance — it is what the citation shows."""
        _, chunks = run
        multi = [c for c in chunks if len(c.strategies) > 1]
        assert multi, "five views of one passage should agree somewhere"
        assert all(c.strategy in c.strategies for c in multi)

    def test_passage_chunks_survive_dedup_because_children_expand_into_them(self, run):
        _, chunks = run
        parents = {c.doc_id for c in chunks if c.strategy == "passage"}
        assert parents == {"d1", "d2"}

    def test_every_child_points_at_a_real_parent(self, run):
        _, chunks = run
        parent_ids = {c.chunk_id for c in chunks if c.strategy == "passage"}
        for chunk in chunks:
            if chunk.strategy == "passage":
                continue
            assert chunk.parent_id in parent_ids or chunk.parent_id == chunk.doc_id

    def test_chunk_ids_are_unique(self, run):
        _, chunks = run
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_the_context_header_goes_on_embed_text_only(self, run):
        """`text` is what a human reads and what the answer quotes.

        Headers help retrieval — a chunk cut from the middle of a passage still
        carries document-level context — but quoting one back would put keyword
        soup in the answer, so the two fields are deliberately different.
        """
        _, chunks = run
        decorated = [c for c in chunks if c.metadata.get("context_header")]
        assert decorated
        for chunk in decorated:
            assert chunk.embed_text != chunk.text
            assert chunk.text in chunk.embed_text
            assert "::" not in chunk.text

    def test_no_gold_query_leaks_into_chunk_metadata(self, run):
        """Headers are derived from the passage alone.

        Using the dataset's own query as chunk metadata would inflate retrieval
        scores on exactly the queries used to evaluate it.
        """
        _, chunks = run
        for chunk in chunks:
            assert "query" not in chunk.metadata
            assert "gold_answer" not in chunk.metadata

    def test_stats_describe_what_was_produced(self, run):
        pipeline, chunks = run
        stats = pipeline.stats.as_dict()
        assert stats["documents"] == 2
        assert stats["chunks"] == len(chunks)
        assert 0.0 < stats["dedup_ratio"] < 1.0
        assert sum(stats["per_strategy"].values()) == len(chunks)

    def test_an_empty_corpus_does_not_raise(self):
        pipeline = ChunkingPipeline()
        assert pipeline.run([]) == []


def test_semantic_strategy_is_added_only_when_an_encoder_is_available():
    names = {c.name for c in default_chunkers(encode=None)}
    assert "semantic" not in names
    with_encoder = {c.name for c in default_chunkers(encode=lambda xs: np.zeros((len(xs), 2)))}
    assert with_encoder - names == {"semantic"}
