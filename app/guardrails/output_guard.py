"""Output guardrail: verify the answer against the evidence, sentence by sentence.

Retrieval-augmented does not mean grounded. The answer still has to be *checked*,
and checking has to fit the same 200 ms budget, which rules out an NLI model. What
fits is lexical entailment against the cited passages: for each sentence, how much
of its content survives intersection with the context, at unigram and bigram level.

Numbers get their own pass. A paraphrase that drifts is a nuisance; a figure that
drifts is a fabrication, and it is the failure users are least able to catch. So
every quantity in the answer must appear verbatim in the cited text.

When only part of an answer is unsupported the guard repairs rather than refuses —
unsupported sentences are dropped and the remainder is re-verified. Refusal is the
outcome when nothing survives.

**What this guard does not do.** It is not what decides whether to answer at all.
Measured over the two holdout populations (155 queries whose evidence is indexed, 56
whose evidence deliberately is not), the grounding score separates them barely at
all: median 1.000 against 1.000, and the best achievable Youden J over any floor is
0.018. That is expected rather than broken — the extractive generator copies its
sentences verbatim out of the retrieved context, so coverage is ~1.0 by construction
whether or not that context has anything to do with the question. Grounding answers
"is this claim supported by the evidence cited", which is a different question from
"is this evidence an answer". The second question is the retrieval-confidence gate's
job, and *that* is where the discrimination lives: floor 0.67, J = 0.798.

So `min_grounding = 0.55` is a floor against a generator going off-script — it earns
its place on the `app/generation/llm.py` path, where invention is possible, and via
the numeric pass on both paths. It refuses 0/155 answerable drafts, which is the
correct behaviour for a backstop and would be a useless result for a discriminator.
Reading the two roles as one is the mistake this note exists to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.generation.extractive import AnswerDraft
from app.guardrails.input_guard import redact_pii
from app.guardrails.policies import UNGROUNDED_MESSAGE
from app.schemas import Citation, GroundingReport, GuardFinding, RetrievedChunk, Verdict
from app.text import content_tokens, extract_numbers, split_sentences

# Citation markers are apparatus this system adds in `ExtractiveComposer._render`,
# not part of the claim being checked, and leaving them in the text being verified
# broke this guard three ways at once:
#
#   1. `extract_numbers` reads "[1]" as the quantity 1, so every cited answer
#      reported a fabricated figure and every sentence carrying a marker was
#      dropped as numerically unsupported.
#   2. `split_sentences` treats " [1]" after a full stop as a new sentence, because
#      "[" is a legitimate sentence opener. That phantom sentence has no content
#      tokens, so `_coverage` scored it 1.0 and it counted as *supported* —
#      inflating both the support ratio and the mean coverage of every answer, and
#      in the degenerate case letting the guard return "[1]" as a grounded answer.
#   3. (1) was invisible in the benchmark only because `_number_present` fell back
#      to a substring test, and the digit 1 is a substring of most numbers a web
#      passage contains. Two bugs cancelling, which is why the numbers looked fine.
#
# Net effect measured over 155 answerable holdout drafts: mean grounding score rose
# 0.9212 -> 0.9949 once markers stopped being verified as prose. The direction is
# worth noting — (1) was *penalising* correct answers harder than (2) was inflating
# them, so the visible symptom of all this was good answers being trimmed, not bad
# ones getting through.
_MARKER = re.compile(r"\[\d+\]")
_LEADING_MARKERS = re.compile(r"^((?:\[\d+\]\s*)+)(.*)$", re.DOTALL)
# The digits at the head of a quantity, so "55 mph" and "55" compare equal.
_QUANTITY = re.compile(r"^\d+(?:\.\d+)?")


@dataclass
class OutputDecision:
    ok: bool
    answer: str
    verdict: Verdict = Verdict.ANSWERED
    report: GroundingReport = field(default_factory=GroundingReport)
    findings: list[GuardFinding] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _prose(text: str) -> str:
    """The claim without its citation apparatus."""
    return _MARKER.sub(" ", text).strip()


def claims(text: str) -> list[str]:
    """Split an answer into verifiable claims, re-attaching displaced markers.

    A marker is not a sentence, and the splitter puts it at the *head* of the
    following piece — "…owners." / "[1] It can own property…" — because "[" opens a
    sentence as far as the splitter is concerned. But "[1]" cites the sentence
    before it, so each leading marker run is moved back onto the claim it belongs
    to. That is both the correct segmentation and what lets the repair step drop an
    unsupported sentence without orphaning its citation or stealing its neighbour's.
    """
    pieces: list[str] = []
    for raw in split_sentences(text):
        piece = raw.strip()
        match = _LEADING_MARKERS.match(piece)
        if not match:
            pieces.append(piece)
            continue
        markers, rest = match.group(1).strip(), match.group(2).strip()
        if pieces:
            pieces[-1] = f"{pieces[-1]} {markers}"
        elif rest:
            rest = f"{markers} {rest}"  # nothing to attach it to; leave it inline
        else:
            rest = markers
        if rest:
            pieces.append(rest)
    return pieces


class OutputGuard:
    def __init__(
        self,
        min_grounding: float = 0.55,
        sentence_support: float = 0.60,
        require_numbers_in_context: bool = True,
    ) -> None:
        self.min_grounding = min_grounding
        self.sentence_support = sentence_support
        self.require_numbers_in_context = require_numbers_in_context

    # ------------------------------------------------------------------ public
    def check(
        self, draft: AnswerDraft, contexts: Sequence[RetrievedChunk]
    ) -> OutputDecision:
        if not draft.text.strip():
            reason = str(draft.detail.get("refusal_reason") or "generator produced no answer")
            return OutputDecision(
                ok=False,
                answer="",
                verdict=Verdict.DECLINED_UNGROUNDED,
                findings=[
                    GuardFinding(
                        guard="grounding", passed=False, severity="block", reason=reason
                    )
                ],
            )

        haystacks = [
            (context.chunk_id, f"{context.context_text or ''} {context.text}")
            for context in contexts
        ]
        pooled_tokens = set()
        pooled_bigrams: set[tuple[str, str]] = set()
        pooled_numbers: set[str] = set()
        per_chunk: dict[str, tuple[set[str], set[tuple[str, str]]]] = {}
        for chunk_id, body in haystacks:
            tokens = content_tokens(body)
            per_chunk[chunk_id] = (set(tokens), _bigrams(tokens))
            pooled_tokens |= set(tokens)
            pooled_bigrams |= _bigrams(tokens)
            pooled_numbers |= set(extract_numbers(body))

        sentences = claims(draft.text) or [draft.text]
        supported: list[str] = []
        unsupported: list[str] = []
        coverages: list[float] = []
        for sentence in sentences:
            coverage = self._coverage(sentence, pooled_tokens, pooled_bigrams)
            coverages.append(coverage)
            (supported if coverage >= self.sentence_support else unsupported).append(sentence)

        findings: list[GuardFinding] = []
        warnings: list[str] = []

        # 1. Numbers must be copied, not invented ---------------------------
        answer_numbers = set(extract_numbers(_prose(draft.text)))
        stray_numbers = sorted(
            n for n in answer_numbers if not self._number_present(n, pooled_numbers)
        )
        if self.require_numbers_in_context and stray_numbers:
            stray_set = set(stray_numbers)
            tainted = [
                s for s in supported if stray_set & set(extract_numbers(_prose(s)))
            ]
            unsupported.extend(tainted)
            supported = [s for s in supported if s not in tainted]
        findings.append(
            GuardFinding(
                guard="numeric_grounding",
                passed=not stray_numbers,
                severity="block" if stray_numbers else "info",
                reason=f"numbers absent from context: {', '.join(stray_numbers)}"
                if stray_numbers
                else "",
                detail={"answer_numbers": sorted(answer_numbers), "stray": stray_numbers},
            )
        )

        # 2. Citations must resolve to retrieved chunks ----------------------
        known_ids = {context.chunk_id for context in contexts}
        bad_citations = [c.marker for c in draft.citations if c.chunk_id not in known_ids]
        citations_valid = not bad_citations
        findings.append(
            GuardFinding(
                guard="citations",
                passed=citations_valid,
                severity="block" if bad_citations else "info",
                reason=f"citations not in retrieved set: {bad_citations}" if bad_citations else "",
                detail={"count": len(draft.citations)},
            )
        )

        # 3. Repair: drop unsupported sentences, keep the grounded remainder --
        answer = draft.text
        if unsupported and supported:
            answer = self._rebuild(answer, unsupported)
            warnings.append(
                f"removed {len(unsupported)} unsupported sentence(s) before answering"
            )
            findings.append(
                GuardFinding(
                    guard="repair",
                    passed=True,
                    severity="warn",
                    reason="answer trimmed to the grounded portion",
                    detail={"removed": len(unsupported)},
                )
            )

        score = self._score(supported, sentences, coverages, citations_valid)
        report = GroundingReport(
            score=round(score, 4),
            supported_sentences=len(supported),
            total_sentences=len(sentences),
            unsupported=unsupported[:5],
            unsupported_numbers=stray_numbers,
            citations_valid=citations_valid,
        )
        grounded = bool(supported) and score >= self.min_grounding and citations_valid
        findings.append(
            GuardFinding(
                guard="grounding",
                passed=grounded,
                severity="info" if grounded else "block",
                reason="" if grounded else f"grounding {score:.2f} < {self.min_grounding:.2f}",
                detail={"score": round(score, 4), "threshold": self.min_grounding},
            )
        )

        # 4. Never echo PII back out ----------------------------------------
        leaked, pii_types = redact_pii(answer)
        if pii_types:
            answer = leaked
            warnings.append(f"redacted {', '.join(pii_types)} from the answer")
        findings.append(
            GuardFinding(
                guard="output_pii",
                passed=not pii_types,
                severity="warn" if pii_types else "info",
                detail={"types": pii_types},
            )
        )

        if not grounded:
            return OutputDecision(
                ok=False,
                answer=UNGROUNDED_MESSAGE,
                verdict=Verdict.DECLINED_UNGROUNDED,
                report=report,
                findings=findings,
                citations=draft.citations,
                warnings=warnings,
            )
        return OutputDecision(
            ok=True,
            answer=answer.strip(),
            verdict=Verdict.ANSWERED,
            report=report,
            findings=findings,
            citations=draft.citations,
            warnings=warnings,
        )

    # ----------------------------------------------------------------- private
    @staticmethod
    def _coverage(
        sentence: str, pooled_tokens: set[str], pooled_bigrams: set[tuple[str, str]]
    ) -> float:
        """Lexical entailment: how much of this sentence exists in the context."""
        tokens = [t for t in content_tokens(_prose(sentence)) if not t.isdigit()]
        if not tokens:
            return 1.0  # nothing to contradict (bare citation marker, etc.)
        unigram = sum(1 for t in tokens if t in pooled_tokens) / len(tokens)
        bigrams = _bigrams(tokens)
        bigram = (
            sum(1 for b in bigrams if b in pooled_bigrams) / len(bigrams) if bigrams else unigram
        )
        return 0.55 * unigram + 0.45 * bigram

    @staticmethod
    def _number_present(number: str, pooled: set[str]) -> bool:
        """True if this quantity appears in the context, unit differences aside.

        Comparison is on the digits, exactly. The earlier substring fallback
        (`bare in p`) accepted an answer saying "the fee is 100" against a context
        saying "1000" — the answer's figure being a fragment of the evidence's is
        precisely the fabrication this pass exists to catch. Unit mismatches are
        still tolerated in both directions, because dropping or adding "mph" is a
        rendering difference rather than a claim about quantity.
        """
        if number in pooled:
            return True
        digits = _QUANTITY.match(number.strip())
        if not digits:
            return False
        head = digits.group(0)
        return any(
            (match := _QUANTITY.match(candidate.strip())) and match.group(0) == head
            for candidate in pooled
        )

    @staticmethod
    def _rebuild(answer: str, unsupported: Sequence[str]) -> str:
        kept = [
            sentence for sentence in claims(answer) if sentence not in set(unsupported)
        ]
        return " ".join(kept) if kept else answer

    def _score(
        self,
        supported: Sequence[str],
        sentences: Sequence[str],
        coverages: Sequence[float],
        citations_valid: bool,
    ) -> float:
        if not sentences:
            return 0.0
        ratio = len(supported) / len(sentences)
        mean_coverage = sum(coverages) / len(coverages)
        score = 0.6 * ratio + 0.4 * mean_coverage
        return score if citations_valid else score * 0.5
