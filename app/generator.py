"""Adapter: the eval-loop generator contract, over this project's real pipeline.

`rag-local-eval-loop` calls `generate_answer(query, results)` with its OWN
context objects (duck-typed `.text` / `.source`) drawn from its own throwaway
index, and expects back an object with `.text`, `.grounded`, `.generation_ms`
and `.model`.

Two things this deliberately does NOT do:

* It does not re-retrieve. The suite has already chosen the contexts and is
  measuring generation alone; retrieving again would grade a different set of
  passages than the one the suite scores against.
* It does not report `grounded=True` unconditionally. That flag drives the
  suite's "lying factor" reliability check, so it is wired to the same two
  gates the served path uses — the retrieval-confidence floor
  (`min_retrieval_score`) and the output guard's grounding verification. A
  refusal here is a real refusal, produced by the same code as in production.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.config import get_settings
from app.generation.intent import classify
from app.generation.extractive import ExtractiveComposer
from app.guardrails.output_guard import OutputGuard
from app.guardrails.policies import NO_CONTEXT_MESSAGE
from app.retrieval.embedder import get_embedder
from app.schemas import RetrievedChunk, Verdict
from app.text import folded_tokens


@dataclass(slots=True)
class EvalAnswer:
    """The shape TARGET_INTERFACE.md requires back from generate_answer()."""

    text: str
    grounded: bool
    generation_ms: float
    model: str
    detail: dict[str, Any] = field(default_factory=dict)


_composer: ExtractiveComposer | None = None
_guard: OutputGuard | None = None


def _components() -> tuple[ExtractiveComposer, OutputGuard]:
    global _composer, _guard
    settings = get_settings()
    if _composer is None:
        _composer = ExtractiveComposer(
            get_embedder(), max_words=settings.answer_max_words
        )
    if _guard is None:
        _guard = OutputGuard(min_grounding=settings.min_grounding_score)
    return _composer, _guard


def _to_chunks(results: Sequence[Any]) -> list[RetrievedChunk]:
    """The suite's duck-typed contexts -> this project's RetrievedChunk.

    Scores are not carried over because the suite's context objects do not have
    them; confidence is recomputed below from the query/context overlap instead.
    """
    chunks: list[RetrievedChunk] = []
    for rank, item in enumerate(results):
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        source = str(getattr(item, "source", f"eval-{rank}"))
        chunks.append(
            RetrievedChunk(
                chunk_id=f"{source}#{rank}",
                parent_id=source,
                text=text,
                context_text=text,
                score=0.0,
                metadata={"eval_rank": rank, "source": source},
            )
        )
    return chunks


def _confidence(query: str, chunks: Sequence[RetrievedChunk], query_vector: np.ndarray) -> float:
    """Same fusion the served retrieval guard reads: dense agreement + coverage.

    hybrid.py computes this from its own dense/BM25 magnitudes, which do not
    exist here (the suite retrieved these passages itself). Cosine against the
    same embedder plus query-term coverage preserves the two signals that
    actually decide the gate, so the refusal behaviour matches production.
    """
    if not chunks:
        return 0.0
    vectors = get_embedder().encode([c.text for c in chunks])
    cosine = float(np.max(vectors @ np.asarray(query_vector, dtype=np.float32).reshape(-1)))
    dense = max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    query_terms = set(folded_tokens(query))
    if query_terms:
        best = 0.0
        for chunk in chunks:
            overlap = query_terms & set(folded_tokens(chunk.text))
            best = max(best, len(overlap) / len(query_terms))
        coverage = best
    else:
        coverage = 0.0
    return round(0.6 * dense + 0.4 * coverage, 4)


def generate_answer(query: str, results: Sequence[Any]) -> EvalAnswer:
    started = time.perf_counter()
    settings = get_settings()
    composer, guard = _components()
    model = f"extractive/{get_embedder().name}"

    def done(text: str, grounded: bool, **detail: Any) -> EvalAnswer:
        return EvalAnswer(
            text=text,
            grounded=grounded,
            generation_ms=round((time.perf_counter() - started) * 1000, 3),
            model=model,
            detail=detail,
        )

    chunks = _to_chunks(results)
    if not chunks:
        return done(NO_CONTEXT_MESSAGE, False, reason="no_context")

    intent = classify(query)
    query_vector = get_embedder().encode([intent.normalized or query])[0]

    # Gate 1: retrieval confidence — the measured 0.67 floor. This is the gate
    # that actually separates answerable from unanswerable (README: J = 0.798).
    confidence = _confidence(query, chunks, query_vector)
    if confidence < settings.min_retrieval_score:
        return done(
            NO_CONTEXT_MESSAGE,
            False,
            reason="below_confidence_floor",
            confidence=confidence,
            threshold=settings.min_retrieval_score,
        )

    draft = composer.compose(query, chunks, intent, query_vector)
    if not draft.text.strip():
        return done(NO_CONTEXT_MESSAGE, False, reason="empty_draft", confidence=confidence)

    # Gate 2: grounding verification — a backstop, and the source of the
    # repaired-answer path (unsupported claims dropped, remainder kept).
    verification = guard.check(draft, chunks)
    grounded = verification.verdict not in (
        Verdict.DECLINED_UNGROUNDED,
        Verdict.DECLINED_NO_CONTEXT,
    )
    return done(
        verification.answer,
        grounded,
        reason=verification.verdict.value,
        confidence=confidence,
        grounding=verification.report.score,
    )
