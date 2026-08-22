"""The harness: one orchestrated path from audio to a verified answer.

    transcribe → guard_input → embed → retrieve → retrieval_guard
              → generate → verify → finalize

What makes this a harness rather than a function call:

* **Stages are timed and named.** Every request carries a span-level trace, so the
  P50/P70/P100 tables come from the same instrumentation the UI shows live.
* **The deadline is an input, not a hope.** `remaining_ms()` decides whether the
  LLM path is affordable; when it is not, the extractive composer runs instead and
  the span is marked `degraded`.
* **External calls are wrapped.** STT and the LLM get retries with jittered
  backoff and a circuit breaker; both degrade to a working alternative.
* **Refusal is a first-class outcome.** Guards can end the request at four
  different points, each with its own verdict, message and recorded reason.
* **Errors return structure.** An unexpected exception becomes an ERROR verdict
  with a trace attached, never a bare 500.
"""

from __future__ import annotations

import logging
import uuid
from collections import OrderedDict
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.generation.extractive import AnswerDraft, ExtractiveComposer
from app.generation.intent import classify
from app.generation.llm import GroqGenerator, LLMUnavailable
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.guardrails.policies import NO_CONTEXT_MESSAGE
from app.harness.retry import CircuitBreaker, CircuitOpen, RetryPolicy, guarded_call
from app.harness.trace import TraceRecorder
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.index_store import RagIndex
from app.schemas import (
    AskRequest,
    AskResponse,
    GroundingReport,
    GuardFinding,
    StageStatus,
    TranscriptionResult,
    Verdict,
)
from app.stt.base import AudioPayload
from app.stt.registry import STTRegistry

logger = logging.getLogger(__name__)

# Stages that sit outside the graded core budget (network-bound by nature).
OUT_OF_BUDGET = ("transcribe",)
# Below this remaining budget the LLM path is not attempted at all.
LLM_BUDGET_FLOOR_MS = 1_200


class RagPipeline:
    def __init__(
        self,
        index: RagIndex,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index = index
        self.embedder = embedder or get_embedder(self.settings.embedding_model)
        self.retriever = HybridRetriever(
            index,
            self.embedder,
            dense_top_k=self.settings.dense_top_k,
            sparse_top_k=self.settings.sparse_top_k,
            fusion_top_k=self.settings.fusion_top_k,
            context_top_k=self.settings.context_top_k,
            rrf_k=self.settings.rrf_k,
            mmr_lambda=self.settings.mmr_lambda,
        )
        self.composer = ExtractiveComposer(
            self.embedder, max_words=self.settings.answer_max_words
        )
        self.llm = GroqGenerator(self.settings)
        self.input_guard = InputGuard(
            min_chars=self.settings.min_query_chars, max_chars=self.settings.max_query_chars
        )
        self.output_guard = OutputGuard(min_grounding=self.settings.min_grounding_score)
        self.stt = STTRegistry(self.settings)
        self._llm_breaker = CircuitBreaker(name="llm", failure_threshold=3, reset_after_s=45.0)
        self._llm_policy = RetryPolicy(
            attempts=2, base_delay_ms=180.0, retry_on=(LLMUnavailable,)
        )
        self._cache: OrderedDict[str, AskResponse] = OrderedDict()

    # ------------------------------------------------------------------ public
    def warmup(self, rounds: int = 3) -> float:
        """Touch every hot path before serving.

        First-call costs are real but not representative: BLAS picks its kernels,
        numpy allocates its scratch buffers, and the tokeniser builds its caches.
        Measured here, the first query costs ~240 ms and the second ~5 ms — so a
        cold process would put a 240 ms outlier in every P100 it reports. The API
        calls this on startup and the benchmark calls it before timing.
        """
        import time as _time

        started = _time.perf_counter()
        for i in range(max(rounds, 1)):
            probe = f"warmup probe {i} average cost of replacement"
            vector = self.embedder.encode([probe])[0]
            result = self.retriever.retrieve(probe, vector)
            if result.chunks:
                draft = self.composer.compose(probe, result.chunks, classify(probe), vector)
                self.output_guard.check(draft, result.chunks)
        return round((_time.perf_counter() - started) * 1000, 2)

    @property
    def stt_providers(self) -> list[str]:
        return self.stt.available

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": self.index.size,
            "passages": self.index.n_passages,
            "embedding_model": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "vector_backend": (
                self.index.vector_store.backend if self.index.vector_store else "unset"
            ),
            "bm25_vocab": self.index.bm25.vocab_size,
            "manifest": self.index.manifest,
            "stt_providers": self.stt_providers,
            "llm_available": self.llm.available,
            "llm_circuit": self._llm_breaker.state,
            "core_budget_ms": self.settings.core_budget_ms,
        }

    async def answer_audio(
        self, audio: AudioPayload, request: AskRequest, provider: str | None = None
    ) -> AskResponse:
        """Voice entry point: transcribe, then run the text pipeline."""
        recorder = TraceRecorder(budget_ms=self.settings.core_budget_ms)
        transcription: TranscriptionResult | None = None
        with recorder.span("transcribe") as span:
            transcription = await self.stt.transcribe(audio, preferred=provider)
            span.attempts = transcription.attempts
            span.note(
                provider=transcription.provider,
                language=transcription.language,
                chars=len(transcription.text),
                audio_bytes=audio.size,
            )
        request = request.model_copy(update={"query": transcription.text})
        response = await self.answer(request, recorder=recorder)
        response.transcript = transcription.text
        response.stt_latency_ms = transcription.latency_ms
        response.detected_language = transcription.language or response.detected_language
        return response

    async def answer(
        self, request: AskRequest, recorder: TraceRecorder | None = None
    ) -> AskResponse:
        recorder = recorder or TraceRecorder(budget_ms=self.settings.core_budget_ms)
        request_id = uuid.uuid4().hex[:12]
        try:
            return await self._run(request, recorder, request_id)
        except Exception as exc:  # noqa: BLE001 - the API must never leak a traceback
            logger.exception("pipeline failure [%s]", request_id)
            trace = recorder.finish(exclude=OUT_OF_BUDGET)
            return AskResponse(
                verdict=Verdict.ERROR,
                answer="Something broke while answering. The error has been logged.",
                query=request.query,
                trace=trace if request.include_trace else trace.model_copy(update={"spans": []}),
                latency_ms=round(recorder.elapsed_ms, 3),
                core_latency_ms=trace.total_ms,
                request_id=request_id,
                guards=[
                    GuardFinding(
                        guard="pipeline",
                        passed=False,
                        severity="block",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                ],
            )

    # ----------------------------------------------------------------- stages
    async def _run(
        self, request: AskRequest, recorder: TraceRecorder, request_id: str
    ) -> AskResponse:
        mode = request.mode or self.settings.generation_mode
        top_k = request.top_k or self.settings.context_top_k
        cache_key = f"{mode}|{top_k}|{request.query.strip().casefold()}"
        if self.settings.enable_query_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            self._cache.move_to_end(cache_key)
            recorder.record("cache_hit", 0.0)
            return cached.model_copy(update={"request_id": request_id})

        # 1. Input guardrail ------------------------------------------------
        with recorder.span("guard_input") as span:
            decision = self.input_guard.check(request.query)
            span.note(
                allowed=decision.allowed,
                verdict=decision.verdict.value,
                pii=decision.pii_types,
            )
            if not decision.allowed:
                span.mark(StageStatus.SKIPPED, blocked_by=decision.verdict.value)
        if not decision.allowed:
            logger.info("[%s] blocked: %s | %r", request_id, decision.verdict.value, decision.redacted)
            return self._finish(
                recorder,
                request,
                request_id,
                verdict=decision.verdict,
                answer=decision.message,
                guards=decision.findings,
            )

        # 2. Intent + query embedding ---------------------------------------
        with recorder.span("classify") as span:
            intent = classify(decision.query)
            span.note(intent=intent.intent.value, script=intent.script)

        with recorder.span("embed_query") as span:
            query_vector: np.ndarray = self.embedder.encode([decision.query])[0]
            span.note(dim=int(query_vector.shape[0]), model=self.embedder.name)

        # 3. Hybrid retrieval ------------------------------------------------
        with recorder.span("retrieve") as span:
            retrieval = self.retriever.retrieve(decision.query, query_vector, top_k=top_k)
            span.note(
                **{k: v for k, v in retrieval.detail.items()},
                confidence=retrieval.confidence,
                dense_top=retrieval.dense_top,
                sparse_top=retrieval.sparse_top,
            )

        # 4. Retrieval guardrail: is there anything worth answering from? ----
        with recorder.span("retrieval_guard") as span:
            enough = bool(retrieval.chunks) and retrieval.confidence >= self.settings.min_retrieval_score
            span.note(
                confidence=retrieval.confidence,
                threshold=self.settings.min_retrieval_score,
                passed=enough,
            )
        guards = list(decision.findings)
        guards.append(
            GuardFinding(
                guard="retrieval",
                passed=enough,
                severity="info" if enough else "block",
                reason="" if enough else (
                    f"top-context confidence {retrieval.confidence:.2f} below "
                    f"{self.settings.min_retrieval_score:.2f}"
                ),
                detail={
                    "confidence": retrieval.confidence,
                    "candidates": retrieval.candidates,
                    "dense_top": retrieval.dense_top,
                    "sparse_top": retrieval.sparse_top,
                },
            )
        )
        if not enough:
            return self._finish(
                recorder,
                request,
                request_id,
                verdict=Verdict.DECLINED_NO_CONTEXT,
                answer=NO_CONTEXT_MESSAGE,
                guards=guards,
                sources=retrieval.chunks,
                detected_language=intent.script,
            )

        # 5. Generation (budget-aware, degrades to extractive) ---------------
        draft = await self._generate(recorder, mode, decision.query, retrieval.chunks, intent, query_vector)

        # 6. Output guardrail: verify before anyone sees it ------------------
        with recorder.span("verify") as span:
            verification = self.output_guard.check(draft, retrieval.chunks)
            span.note(
                grounding=verification.report.score,
                supported=verification.report.supported_sentences,
                total=verification.report.total_sentences,
                passed=verification.ok,
            )
        guards.extend(verification.findings)

        response = self._finish(
            recorder,
            request,
            request_id,
            verdict=verification.verdict,
            answer=verification.answer,
            guards=guards,
            sources=retrieval.chunks,
            citations=verification.citations if verification.ok else [],
            grounding=verification.report,
            generation_mode=draft.mode,
            warnings=verification.warnings,
            detected_language=intent.script,
        )
        if self.settings.enable_query_cache and response.verdict is Verdict.ANSWERED:
            self._cache[cache_key] = response
            while len(self._cache) > self.settings.query_cache_size:
                self._cache.popitem(last=False)
        return response

    async def _generate(
        self,
        recorder: TraceRecorder,
        mode: str,
        query: str,
        contexts: list,
        intent: Any,
        query_vector: np.ndarray,
    ) -> AnswerDraft:
        """LLM when the caller asked for it and the budget allows; else extractive."""
        remaining = recorder.remaining_ms(exclude=OUT_OF_BUDGET)
        wants_llm = mode == "llm" or (mode == "auto" and remaining >= LLM_BUDGET_FLOOR_MS)
        if wants_llm and self.llm.available:
            with recorder.span("generate") as span:
                span.note(requested="llm", remaining_budget_ms=round(remaining, 1))
                try:
                    async def call() -> AnswerDraft:
                        return await self.llm.generate(query, contexts, intent)

                    draft, attempts = await guarded_call(
                        self._llm_breaker, call, self._llm_policy
                    )
                    span.attempts = attempts
                    span.note(mode="llm", **{k: v for k, v in draft.detail.items()})
                    return draft
                except (LLMUnavailable, CircuitOpen) as exc:
                    span.mark(StageStatus.DEGRADED, fallback="extractive", error=str(exc)[:160])
                    logger.warning("llm unavailable, degrading to extractive: %s", exc)
                    draft = self.composer.compose(query, contexts, intent, query_vector)
                    span.note(mode=draft.mode, **{k: v for k, v in draft.detail.items()})
                    return draft

        with recorder.span("generate") as span:
            draft = self.composer.compose(query, contexts, intent, query_vector)
            status = (
                StageStatus.DEGRADED
                if wants_llm and not self.llm.available
                else StageStatus.OK
            )
            if status is StageStatus.DEGRADED:
                span.mark(status, fallback="extractive", error="llm not configured")
            span.note(
                mode=draft.mode,
                remaining_budget_ms=round(remaining, 1),
                **{k: v for k, v in draft.detail.items()},
            )
            return draft

    # ---------------------------------------------------------------- helpers
    def _finish(
        self,
        recorder: TraceRecorder,
        request: AskRequest,
        request_id: str,
        *,
        verdict: Verdict,
        answer: str,
        guards: list[GuardFinding],
        sources: list | None = None,
        citations: list | None = None,
        grounding: Any = None,
        generation_mode: str = "none",
        warnings: list[str] | None = None,
        detected_language: str | None = None,
    ) -> AskResponse:
        trace = recorder.finish(exclude=OUT_OF_BUDGET)
        return AskResponse(
            verdict=verdict,
            answer=answer,
            query=request.query,
            generation_mode=generation_mode,
            citations=citations or [],
            sources=sources or [],
            grounding=grounding or GroundingReport(),
            guards=guards,
            trace=trace if request.include_trace else trace.model_copy(update={"spans": []}),
            latency_ms=round(recorder.elapsed_ms, 3),
            core_latency_ms=trace.total_ms,
            detected_language=detected_language,
            request_id=request_id,
            warnings=warnings or [],
        )

    async def aclose(self) -> None:
        await self.stt.aclose()
