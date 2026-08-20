"""Extractive answer composition — the default generator.

A hosted LLM cannot answer inside a 200 ms end-to-end budget: the network round
trip alone spends it. So the default generator *composes* rather than invents. It
scores every candidate sentence in the retrieved contexts against the query
(static-embedding cosine + IDF-weighted lexical overlap + intent cues), picks a
non-redundant few, and stitches them into a cited answer.

The property that matters: every word of the answer came verbatim from a retrieved
passage, so it cannot hallucinate — the output guard is verifying an invariant the
generator already holds, not hoping the model behaved. `app/generation/llm.py` adds
fluency when the caller opts into a bigger budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from app.generation.intent import Intent, QueryIntent
from app.retrieval.embedder import Embedder
from app.schemas import Citation, RetrievedChunk
from app.text import (
    content_tokens,
    extract_numbers,
    folded_tokens,
    normalize,
    split_sentences,
    truncate_words,
)

_MAX_CANDIDATES = 40

# MS MARCO passages are web scrapes, and a minority of them are damaged in two
# ways that a sentence-level extractor will happily quote back:
#
#   1. Machine-translation loops — "Fast mode. Fast mode. Nonclustered index..."
#   2. Line-wrap artefacts, where the scrape put a full stop at every line break:
#      "An eagle chick will eat as much as it can at a single. / feeding, storing
#      food in its crop. / The crop, an organ located. / near the base of..."
#
# Both produce text that is perfectly *grounded* — every word is verbatim from a
# retrieved passage — so the output guard cannot catch them. They have to be
# filtered where sentences are chosen. Measured on this corpus: 0.58% of the
# 19,876 candidate sentences fall below the repetition threshold, and the 1st
# percentile of distinct-token ratio is 0.60, so 0.55 sits below anything a
# healthy sentence produces.
_MIN_DISTINCT_RATIO = 0.55

# Function words that cannot end an English sentence. A candidate ending in one
# was cut mid-clause, whatever the full stop claims.
_DANGLING_TAIL = frozenset(
    """a an the this that these those of at in on to for with within without into
    onto from by as per via and or but nor is was are were be been being am has
    have had do does did will would shall should can could may might must my our
    your their its""".split()
)
# Lowercase-initial brand names ("iPhone", "eBay") are not fragments.
_BRANDISH = re.compile(r"^[a-z]+[A-Z]")


@dataclass
class AnswerDraft:
    text: str
    citations: list[Citation] = field(default_factory=list)
    support: list[tuple[str, str]] = field(default_factory=list)
    mode: str = "extractive"
    detail: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class _Candidate:
    text: str
    chunk_id: str
    source_rank: int
    position: int
    retrieval_score: float
    score: float = 0.0


class ExtractiveComposer:
    def __init__(
        self,
        embedder: Embedder,
        max_words: int = 70,
        max_sentences: int = 3,
        redundancy_penalty: float = 0.45,
        # A sentence joins the answer only if it scores within this fraction of the
        # best one. Swept 0.50–0.90 against how much of the dataset's own
        # gold_answer survives into the composed text: 0.50 -> 0.408 recall at 33.5
        # words, 0.70 -> 0.318 at 17.5, and everything above 0.75 collapses to a
        # single sentence at 0.317. Tightening it looked like it would trim the
        # padding; what it actually trims is the answer, because on this corpus the
        # supporting sentences are where the rest of the gold content lives.
        padding_floor: float = 0.5,
        # Optional: require an added sentence to resemble the one it is joining
        # (cosine >= floor), not just the query, giving an admissible band of
        # [floor, 0.86] — related but not repetitive. Off by default because the
        # sweep rejected it: it does clean up individual answers (at 0.40 the
        # "how fast does an eagle travel" answer loses an Amtrak *Texas Eagle*
        # distractor), but gold-answer recall falls monotonically with it —
        # 0.423 off, 0.420 at 0.30, 0.403 at 0.40, 0.357 at 0.60 over 143
        # queries. Entity collisions like Texas Eagle are a retrieval problem;
        # paying two points of recall corpus-wide to hide one is a bad trade.
        coherence_floor: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.max_words = max_words
        self.max_sentences = max_sentences
        self.redundancy_penalty = redundancy_penalty
        self.padding_floor = padding_floor
        self.coherence_floor = coherence_floor

    # ------------------------------------------------------------------ public
    def compose(
        self,
        query: str,
        contexts: Sequence[RetrievedChunk],
        intent: QueryIntent,
        query_vector: np.ndarray | None = None,
    ) -> AnswerDraft:
        candidates = self._candidates(contexts)
        if not candidates:
            return AnswerDraft(text="", mode="extractive", detail={"candidates": 0})

        if query_vector is None:
            query_vector = self.embedder.encode([intent.normalized or query])[0]
        vectors = self.embedder.encode([c.text for c in candidates])
        similarity = vectors @ np.asarray(query_vector, dtype=np.float32).reshape(-1)

        focus = set(intent.focus)
        for i, candidate in enumerate(candidates):
            candidate.score = self._score(candidate, float(similarity[i]), focus, intent)

        picked = self._select(candidates, vectors)
        return self._render(picked, contexts, intent)

    # ----------------------------------------------------------------- private
    @staticmethod
    def _is_degenerate(sentence: str) -> bool:
        """True for machine-translation loops and other repeated-phrase noise."""
        tokens = content_tokens(sentence)
        if len(tokens) < 4:
            return False
        return len(set(tokens)) / len(tokens) < _MIN_DISTINCT_RATIO

    @staticmethod
    def _is_fragment(sentence: str, following: str | None = None) -> bool:
        """True for text cut mid-clause by a scrape artefact, not a full stop."""
        stripped = sentence.strip()
        if not stripped:
            return True
        first = stripped[0]
        if first.isascii() and first.islower() and not _BRANDISH.match(stripped):
            return True
        words = re.findall(r"[^\W_]+", stripped.casefold())
        if words and words[-1] in _DANGLING_TAIL:
            return True
        # A lowercase-initial *next* sentence is not a new sentence at all — it is
        # the rest of this one, orphaned by a full stop the scrape inserted at a
        # line break. "An eagle chick will eat as much as it can at a single." /
        # "feeding, storing food in its crop." is one clause, not two. Dropping
        # only the lowercase half leaves the truncated head looking well-formed,
        # which is how it reached an answer with a full stop after "at a single".
        #
        # Measured cost, by hand-reading 25 of the drops on this corpus: the rule
        # removes 253/19,876 = 1.27% of candidate sentences, and roughly a quarter
        # of those are complete sentences that merely sit before damaged text
        # ("4 GSM aka g/m2 = grams per square meter."), so ~0.35% of sentences are
        # lost unnecessarily. Kept anyway, because the errors are not symmetric: a
        # query has ~14 candidate sentences and losing one rarely changes the
        # answer, whereas quoting visibly broken text is the failure a reader
        # notices first. Gold-answer recall rose from 0.408 to 0.423 with this and
        # the coverage fold in place.
        nxt = (following or "").strip()
        return bool(nxt) and nxt[0].isascii() and nxt[0].islower() and not _BRANDISH.match(nxt)

    @classmethod
    def _candidates(cls, contexts: Sequence[RetrievedChunk]) -> list[_Candidate]:
        """Sentences from every retrieved context, deduplicated, capped."""
        seen: set[str] = set()
        out: list[_Candidate] = []
        for rank, context in enumerate(contexts):
            body = context.context_text or context.text
            sentences = split_sentences(body)
            for position, sentence in enumerate(sentences):
                key = sentence.casefold()[:120]
                if key in seen or len(sentence.split()) < 4:
                    continue
                following = sentences[position + 1] if position + 1 < len(sentences) else None
                if cls._is_degenerate(sentence) or cls._is_fragment(sentence, following):
                    continue
                seen.add(key)
                out.append(
                    _Candidate(
                        text=sentence,
                        chunk_id=context.chunk_id,
                        source_rank=rank,
                        position=position,
                        retrieval_score=context.score,
                    )
                )
                if len(out) >= _MAX_CANDIDATES:
                    return out
        return out

    @staticmethod
    def _score(
        candidate: _Candidate, similarity: float, focus: set[str], intent: QueryIntent
    ) -> float:
        tokens = set(folded_tokens(candidate.text))
        coverage = len(focus & tokens) / len(focus) if focus else 0.0
        score = 0.52 * similarity + 0.28 * coverage + 0.12 * candidate.retrieval_score
        score -= 0.02 * candidate.source_rank      # prefer better-retrieved contexts
        score -= 0.01 * min(candidate.position, 5)  # mild lead-sentence bias

        has_number = bool(extract_numbers(candidate.text))
        if intent.wants_number:
            score += 0.10 if has_number else -0.06
        if intent.intent in (Intent.ENTITY, Intent.LOCATION) and any(
            word[:1].isupper() for word in candidate.text.split()[1:]
        ):
            score += 0.04
        words = len(candidate.text.split())
        if words > 45:
            score -= 0.03  # long sentences dilute the answer
        return score

    def _select(self, candidates: list[_Candidate], vectors: np.ndarray) -> list[_Candidate]:
        """Greedy pick with a redundancy penalty and a hard word budget."""
        order = sorted(range(len(candidates)), key=lambda i: -candidates[i].score)
        chosen: list[int] = []
        words = 0
        for i in order:
            if len(chosen) >= self.max_sentences:
                break
            candidate = candidates[i]
            length = len(candidate.text.split())
            if chosen and words + length > self.max_words:
                continue
            overlap = max((float(vectors[i] @ vectors[j]) for j in chosen), default=0.0)
            if overlap > 0.86:
                continue  # already said
            if chosen and overlap < self.coherence_floor:
                continue  # unrelated to what we are already saying
            adjusted = candidate.score - self.redundancy_penalty * overlap
            if chosen and adjusted < self.padding_floor * candidates[order[0]].score:
                continue  # marginal addition: stop padding the answer
            chosen.append(i)
            words += length
        chosen.sort(key=lambda i: (candidates[i].source_rank, candidates[i].position))
        return [candidates[i] for i in chosen]

    def _render(
        self,
        picked: list[_Candidate],
        contexts: Sequence[RetrievedChunk],
        intent: QueryIntent,
    ) -> AnswerDraft:
        if not picked:
            return AnswerDraft(text="", mode="extractive", detail={"selected": 0})

        by_id = {context.chunk_id: context for context in contexts}
        markers: dict[str, int] = {}
        citations: list[Citation] = []
        pieces: list[str] = []
        support: list[tuple[str, str]] = []

        for candidate in picked:
            if candidate.chunk_id not in markers:
                marker = len(markers) + 1
                markers[candidate.chunk_id] = marker
                context = by_id.get(candidate.chunk_id)
                citations.append(
                    Citation(
                        marker=marker,
                        chunk_id=candidate.chunk_id,
                        parent_id=context.parent_id if context else candidate.chunk_id,
                        quote=truncate_words(candidate.text, 40),
                        strategies=list(context.strategies) if context else [],
                        score=context.score if context else 0.0,
                    )
                )
            sentence = normalize(candidate.text)
            if not sentence.endswith((".", "!", "?", "।")):
                sentence += "."
            pieces.append(f"{sentence} [{markers[candidate.chunk_id]}]")
            support.append((candidate.text, candidate.chunk_id))

        answer = truncate_words(" ".join(pieces), self.max_words + 12)
        return AnswerDraft(
            text=answer,
            citations=citations,
            support=support,
            mode="extractive",
            detail={
                "selected": len(picked),
                "sources": len(citations),
                "intent": intent.intent.value,
                "top_score": round(picked[0].score, 4),
            },
        )
