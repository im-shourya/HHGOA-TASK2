"""Output guardrail — grounding verification and repair.

The generator is extractive, so on the happy path grounding is an invariant rather
than a hope. These tests therefore mostly feed the guard drafts the extractive
composer would never produce, because the guard also runs over `app/generation/llm.py`
output, where invention is possible, and it has to catch it there.
"""

from __future__ import annotations

import pytest

from app.generation.extractive import AnswerDraft
from app.guardrails.output_guard import OutputGuard
from app.guardrails.policies import UNGROUNDED_MESSAGE
from app.schemas import Citation, Verdict
from tests.conftest import make_context


@pytest.fixture(scope="module")
def guard():
    return OutputGuard()


def finding(decision, name):
    return next(f for f in decision.findings if f.guard == name)


CORP = (
    "A corporation is a legal entity separate from its owners. "
    "It can own property and incur debt in its own name. "
    "Shareholders elect a board of directors."
)


class TestGroundedAnswers:
    def test_verbatim_answer_passes(self, guard):
        draft = AnswerDraft(text="A corporation is a legal entity separate from its owners. [1]")
        decision = guard.check(draft, [make_context(CORP)])
        assert decision.ok
        assert decision.verdict is Verdict.ANSWERED
        assert decision.report.score >= 0.55
        assert decision.report.supported_sentences == 1

    def test_score_and_sentence_counts_are_reported(self, guard):
        draft = AnswerDraft(
            text="A corporation is a legal entity separate from its owners. [1] "
            "Shareholders elect a board of directors. [1]"
        )
        decision = guard.check(draft, [make_context(CORP)])
        assert decision.report.total_sentences == 2
        assert decision.report.unsupported == []


class TestRepair:
    def test_drops_the_invented_sentence_and_answers_with_the_rest(self, guard):
        """Partial invention is repaired, not refused — the grounded part is real."""
        draft = AnswerDraft(
            text="A corporation is a legal entity separate from its owners. [1] "
            "It can own property and incur debt in its own name. [1] "
            "Every corporation must publish a monthly poetry anthology. [1]"
        )
        decision = guard.check(draft, [make_context(CORP)])
        assert decision.ok
        assert "poetry" not in decision.answer
        assert "legal entity" in decision.answer
        assert any("unsupported" in w for w in decision.warnings)
        assert finding(decision, "repair").severity == "warn"

    def test_refuses_when_nothing_survives(self, guard):
        draft = AnswerDraft(text="Corporations are governed by the Galactic Trade Federation. [1]")
        decision = guard.check(draft, [make_context(CORP)])
        assert not decision.ok
        assert decision.verdict is Verdict.DECLINED_UNGROUNDED
        assert decision.answer == UNGROUNDED_MESSAGE
        assert "Galactic" not in decision.answer, "a refusal must not leak the claim it rejected"


class TestNumericGrounding:
    def test_one_wrong_digit_sinks_a_fully_worded_sentence(self, guard):
        """The case lexical entailment alone cannot catch.

        Every *word* here is verbatim from the context, so unigram and bigram
        coverage are both 1.0 and the sentence looks perfectly supported. Only the
        figure was changed — which is the failure a reader is least able to spot,
        so numbers get their own pass rather than being averaged into coverage.
        """
        context = make_context("Eagles fly 30 to 55 mph and dive at over 100 mph.")
        draft = AnswerDraft(text="Eagles fly 30 to 55 mph and dive at over 240 mph. [1]")
        decision = guard.check(draft, [context])
        assert not decision.ok
        numeric = finding(decision, "numeric_grounding")
        assert not numeric.passed
        assert "240" in numeric.reason
        assert any("240" in n for n in decision.report.unsupported_numbers)

    def test_copied_figures_pass(self, guard):
        context = make_context("Eagles fly 30 to 55 mph and dive at over 100 mph.")
        draft = AnswerDraft(text="Eagles fly 30 to 55 mph and dive at over 100 mph. [1]")
        decision = guard.check(draft, [context])
        assert decision.ok
        assert finding(decision, "numeric_grounding").passed

    @pytest.mark.parametrize(
        "number,pooled,present",
        [
            ("55 mph", {"55 mph"}, True),
            ("55", {"55 mph"}, True),          # unit dropped by the answer
            ("55 mph", {"55"}, True),          # unit added by the answer
            ("240", {"55 mph", "30"}, False),
            ("1000", {"100"}, False),          # prefix must not count as a match
        ],
    )
    def test_number_matching_tolerates_units_but_not_digits(self, number, pooled, present):
        assert OutputGuard._number_present(number, pooled) is present


class TestCitations:
    def test_a_citation_outside_the_retrieved_set_blocks_the_answer(self, guard):
        """Otherwise a fabricated source id would make an answer look sourced."""
        draft = AnswerDraft(
            text="A corporation is a legal entity separate from its owners. [1]",
            citations=[Citation(marker=1, chunk_id="ghost", parent_id="p9", quote="…")],
        )
        decision = guard.check(draft, [make_context(CORP, chunk_id="c1")])
        assert not decision.ok
        assert not finding(decision, "citations").passed
        assert decision.report.citations_valid is False

    def test_a_resolvable_citation_passes(self, guard):
        draft = AnswerDraft(
            text="A corporation is a legal entity separate from its owners. [1]",
            citations=[Citation(marker=1, chunk_id="c1", parent_id="p1", quote="…")],
        )
        decision = guard.check(draft, [make_context(CORP, chunk_id="c1")])
        assert decision.ok
        assert decision.report.citations_valid is True


def test_pii_in_the_context_is_not_echoed_back(guard):
    """Grounded is not the same as safe to repeat."""
    body = "Contact support at help@example.com for billing questions."
    draft = AnswerDraft(text=f"{body} [1]")
    decision = guard.check(draft, [make_context(body)])
    assert decision.ok
    assert "help@example.com" not in decision.answer
    assert any("redacted" in w for w in decision.warnings)


def test_an_empty_draft_declines_and_says_why(guard):
    draft = AnswerDraft(text="", detail={"refusal_reason": "confidence 0.41 < 0.67"})
    decision = guard.check(draft, [])
    assert not decision.ok
    assert decision.verdict is Verdict.DECLINED_UNGROUNDED
    assert "0.41" in finding(decision, "grounding").reason


def test_no_contexts_means_no_answer_can_be_grounded(guard):
    draft = AnswerDraft(text="Eagles fly 30 to 55 mph. [1]")
    assert not guard.check(draft, []).ok


class TestCitationMarkersAreNotClaims:
    """Three defects with one cause: the markers were being verified as content.

    Found by writing the numeric test above, which passed an answer whose only
    surviving text was "[1]". The markers are apparatus this system adds after the
    answer is composed, and treating them as prose broke the guard three ways.
    """

    def test_a_marker_is_not_a_fabricated_quantity(self, guard):
        """`extract_numbers` read "[1]" as the quantity 1."""
        body = "Shareholders elect a board of directors."
        decision = guard.check(AnswerDraft(text=f"{body} [1]"), [make_context(body)])
        assert finding(decision, "numeric_grounding").passed
        assert decision.report.unsupported_numbers == []

    def test_a_marker_is_not_a_supported_sentence(self, guard):
        """The splitter made " [1]" its own sentence, and it scored 1.0 coverage.

        Counting it inflated the support ratio and the mean coverage of every cited
        answer — so an answer's grounding score rose with the number of citations
        it carried, independently of whether the prose was supported.
        """
        body = "Shareholders elect a board of directors."
        decision = guard.check(AnswerDraft(text=f"{body} [1]"), [make_context(body)])
        assert decision.report.total_sentences == 1

    def test_an_answer_cannot_be_grounded_on_its_markers_alone(self, guard):
        """The degenerate case: everything real was dropped, "[1]" was returned."""
        context = make_context("Eagles fly 30 to 55 mph.")
        draft = AnswerDraft(text="Eagles fly 30 to 55 mph and dive at over 240 mph. [1]")
        decision = guard.check(draft, [context])
        assert not decision.ok
        assert decision.answer == UNGROUNDED_MESSAGE

    def test_repair_takes_the_marker_with_the_sentence_it_belongs_to(self, guard):
        draft = AnswerDraft(
            text="A corporation is a legal entity separate from its owners. [1] "
            "It can own property and incur debt in its own name. [1] "
            "Every corporation must publish a monthly poetry anthology. [2]"
        )
        decision = guard.check(draft, [make_context(CORP)])
        assert decision.ok
        assert "[2]" not in decision.answer, "the dropped sentence left its citation behind"
        assert decision.answer.count("[1]") == 2

    def test_a_prefix_of_a_real_figure_is_still_a_fabrication(self, guard):
        """The substring fallback that was masking all of the above.

        "1" is a substring of most numbers a web passage contains, which is why the
        marker bug never showed up in the benchmark — and the same leniency accepted
        a wrong figure whenever it happened to prefix a right one.
        """
        context = make_context("The annual filing fee is 1000 rupees for a private company.")
        draft = AnswerDraft(text="The annual filing fee is 100 rupees for a private company. [1]")
        decision = guard.check(draft, [context])
        assert not decision.ok
        assert any("100" == n.split(" ")[0] for n in decision.report.unsupported_numbers)
