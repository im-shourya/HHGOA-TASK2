"""Extractive composition — the sentence filters and the no-hallucination invariant.

Two corpus pathologies drive the filters here, both found by reading actual
answers rather than by imagining failure modes:

  1. machine-translation loops — "Fast mode. Fast mode. Nonclustered index…"
  2. line-wrap artefacts, where the scrape put a full stop at every line break:
     "An eagle chick will eat as much as it can at a single." / "feeding, storing
     food in its crop."

Both are perfectly *grounded* — every word is verbatim from a retrieved passage —
so the output guard cannot catch them. They have to be filtered at selection.
"""

from __future__ import annotations

import pytest

from app.generation.extractive import ExtractiveComposer
from app.generation.intent import classify
from tests.conftest import make_context

C = ExtractiveComposer


class TestIsDegenerate:
    def test_catches_machine_translation_loop(self):
        assert C._is_degenerate("Fast mode. Fast mode. Fast mode. Fast mode.")

    def test_catches_repeated_phrase(self):
        assert C._is_degenerate("the cost the cost the cost the cost of it")

    @pytest.mark.parametrize(
        "sentence",
        [
            "Eagles fly 30 to 55 mph and dive at over 100 mph.",
            "A corporation is a legal entity separate from its owners.",
            "Redding, California reached a record high of 118 degrees.",
        ],
    )
    def test_passes_healthy_sentences(self, sentence):
        assert not C._is_degenerate(sentence)

    def test_short_sentences_are_exempt(self):
        """Under four content tokens the ratio is noise, not evidence."""
        assert not C._is_degenerate("Que sera sera")


class TestIsFragment:
    def test_catches_lowercase_start(self):
        assert C._is_fragment("feeding, storing food in its crop.")

    def test_catches_dangling_function_word(self):
        assert C._is_fragment("The male parent does most of the hunting during the early weeks of.")

    def test_catches_sentence_whose_continuation_was_orphaned(self):
        """The real bug: a truncated head that looks well-formed on its own.

        Only the lowercase half was being dropped, which left this behind with a
        full stop after "at a single" and put it straight into an answer.
        """
        assert C._is_fragment(
            "An eagle chick will eat as much as it can at a single.",
            "feeding, storing food in its crop.",
        )

    def test_keeps_sentence_followed_by_a_proper_one(self):
        assert not C._is_fragment(
            "Eagles fly 30 to 55 mph.", "They dive at over 100 mph."
        )

    def test_lowercase_brand_names_are_not_fragments(self):
        assert not C._is_fragment("iPhone 7 was released in September 2016.")
        assert not C._is_fragment("eBay charges a final value fee.")

    def test_empty_is_a_fragment(self):
        assert C._is_fragment("")


class TestCandidates:
    def test_filters_both_pathologies_out_of_one_passage(self):
        damaged = (
            "An eagle chick will eat as much as it can at a single. "
            "feeding, storing food in its crop. "
            "The crop, an organ located. "
            "near the base of the bird's neck, will enlarge as it fills. "
            "Fast mode. Fast mode. Fast mode. Fast mode."
        )
        assert C._candidates([make_context(damaged)]) == []

    def test_keeps_the_good_sentence_from_an_undamaged_passage(self):
        body = (
            "Quick Answer. Eagles fly 30 to 55 mph and dive at over 100 mph. "
            "They store food in the crop."
        )
        kept = [c.text for c in C._candidates([make_context(body)])]
        assert any("30 to 55 mph" in text for text in kept)

    def test_a_sentence_before_a_lowercase_continuation_goes_with_it(self):
        """The rule is deliberately blunt, and this is the cost of it.

        Inside real line-wrap damage the preceding sentence genuinely is cut, so
        dropping it is right. When a complete sentence merely happens to sit
        before damaged text it is dropped too — measured at ~0.35% of all
        candidate sentences, accepted because quoting broken text is the more
        visible failure. Pinned as a test so the trade-off is explicit rather
        than a surprise to whoever reads the filter next.
        """
        body = "Eagles fly 30 to 55 mph and dive at over 100 mph. storing food in its crop."
        assert C._candidates([make_context(body)]) == []

    def test_deduplicates_identical_sentences_across_contexts(self):
        sentence = "A corporation is a legal entity separate from its owners."
        contexts = [
            make_context(sentence, chunk_id="c1", parent_id="p1"),
            make_context(sentence, chunk_id="c2", parent_id="p2"),
        ]
        assert len(C._candidates(contexts)) == 1


@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_every_word_of_the_answer_comes_from_a_context(embedder, floor):
    """The invariant that makes hallucination structurally impossible."""
    body = (
        "A corporation is a legal entity that is separate from its owners. "
        "It can own property, incur debt and be sued in its own name."
    )
    contexts = [make_context(body)]
    composer = ExtractiveComposer(embedder, padding_floor=floor)
    draft = composer.compose("what is a corporation", contexts, classify("what is a corporation"))
    assert draft.text
    stripped = draft.text.replace(" [1]", "").replace(" [2]", "")
    for sentence in stripped.split(". "):
        core = sentence.strip().rstrip(".")
        assert core and core in body, f"{core!r} is not verbatim in the context"


def test_answer_cites_every_source_it_quotes(embedder):
    contexts = [
        make_context("Eagles fly 30 to 55 mph.", chunk_id="c1", parent_id="p1"),
        make_context("Eagles can soar for hours on warm air currents.",
                    chunk_id="c2", parent_id="p2"),
    ]
    composer = ExtractiveComposer(embedder)
    draft = composer.compose("how fast does an eagle travel", contexts,
                             classify("how fast does an eagle travel"))
    markers = {c.marker for c in draft.citations}
    assert markers
    for marker in markers:
        assert f"[{marker}]" in draft.text


def test_plural_query_matches_singular_passage_in_scoring(embedder):
    """The coverage fold, at the level that actually chooses sentences.

    Before folding, "eagle" scored 0.00 coverage against "Eagles fly 30 to 55
    mph" and lost to a scrape fragment whose only merit was the singular spelling.
    """
    contexts = [
        make_context("Eagles fly 30 to 55 mph and dive at over 100 mph.",
                     chunk_id="c1", parent_id="p1"),
        make_context("Promotion is good on Texas Eagle and Heartland Flyer only.",
                     chunk_id="c2", parent_id="p2", score=0.88),
    ]
    composer = ExtractiveComposer(embedder)
    draft = composer.compose("how fast does an eagle travel", contexts,
                             classify("how fast does an eagle travel"))
    assert "30 to 55 mph" in draft.text
    assert draft.text.index("30 to 55") < len(draft.text) / 2, "the answer must lead with it"


def test_empty_contexts_produce_no_answer_rather_than_a_guess(embedder):
    draft = ExtractiveComposer(embedder).compose("anything", [], classify("anything"))
    assert draft.text == ""
