"""Tokenisation and English morphology.

The plural fold is the fix for a measured retrieval failure: BM25 scored a passage
literally answering "how fast does an eagle travel" at exactly 0.0 because it says
"Eagles" and the query says "eagle". These tests pin the two properties that make
the fold safe — symmetry, and a conservative keep-list — rather than asserting a
list of pretty singulars.
"""

from __future__ import annotations

import pytest

from app.text import (
    content_tokens,
    fold_plural,
    folded_tokens,
    lexical_tokens,
    split_sentences,
    tokenize,
)


class TestFoldPlural:
    @pytest.mark.parametrize(
        "plural,singular",
        [
            ("eagles", "eagle"),
            ("dogs", "dog"),
            ("cities", "city"),
            ("companies", "company"),
            ("glasses", "glass"),
            ("watches", "watch"),
            ("boxes", "box"),
            ("dishes", "dish"),
            ("areas", "area"),
            ("videos", "video"),
            ("topics", "topic"),
        ],
    )
    def test_folds_plurals(self, plural, singular):
        assert fold_plural(plural) == singular

    @pytest.mark.parametrize(
        "word",
        [
            "class",     # -ss is a real singular
            "virus",     # -us
            "analysis",  # -is
            "gas",       # too short to fold
            "its",       # stopword
            "bus",
        ],
    )
    def test_keeps_real_singulars(self, word):
        assert fold_plural(word) == word

    @pytest.mark.parametrize("token", ["बाज़", "eagle3", "n95", "covid-19"])
    def test_leaves_non_ascii_and_non_alpha_alone(self, token):
        assert fold_plural(token) == token

    def test_is_idempotent(self):
        """Folding twice must equal folding once, or index and query can diverge."""
        for word in ("eagles", "cities", "boxes", "class", "movies", "analysis"):
            assert fold_plural(fold_plural(word)) == fold_plural(word)

    def test_symmetry_is_what_matters_not_correctness(self):
        """An imperfect fold is safe as long as both sides get the same one.

        "movies" -> "movy" is not a word, but the query and the postings are
        folded through this same function, so they still meet.
        """
        assert fold_plural("movies") == fold_plural("movies")
        assert lexical_tokens("movies")[0] == lexical_tokens("Movies!")[0]


class TestLexicalTokens:
    def test_query_and_document_meet_across_number(self):
        query = set(lexical_tokens("how fast does an eagle travel"))
        document = set(lexical_tokens("Eagles fly 30 to 55 mph and dive at over 100 mph."))
        assert "eagle" in query & document

    def test_unfolded_tokenizer_would_have_missed_it(self):
        """The regression this fixes, stated as a test."""
        query = set(tokenize("how fast does an eagle travel"))
        document = set(tokenize("Eagles fly 30 to 55 mph."))
        assert not (query & document) & {"eagle", "eagles"}

    def test_keeps_stopwords_for_bm25(self):
        """BM25 weights common terms down by IDF; it does not need them removed."""
        assert "the" in lexical_tokens("the eagle")


class TestFoldedTokens:
    def test_drops_stopwords_and_folds(self):
        assert set(folded_tokens("the eagles and the dogs")) == {"eagle", "dog"}

    def test_coverage_of_query_against_passage(self):
        """The extractive selector's term-coverage score, which was reading 0.00."""
        focus = set(folded_tokens("how fast does an eagle travel"))
        sentence = set(folded_tokens("Eagles fly 30 to 55 mph and dive at over 100 mph."))
        assert focus & sentence == {"eagle"}
        assert not set(content_tokens("how fast does an eagle travel")) & set(
            content_tokens("Eagles fly 30 to 55 mph.")
        )


class TestSplitSentences:
    def test_rejoins_abbreviations(self):
        parts = split_sentences("Redding, Calif. reached 118 F in July. That is the record.")
        assert len(parts) == 2
        assert parts[0].startswith("Redding")

    def test_handles_devanagari_danda(self):
        parts = split_sentences("भारत की राजधानी नई दिल्ली है। यह उत्तर भारत में है।")
        assert len(parts) == 2

    def test_empty_input(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []
