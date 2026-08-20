"""Structured input/output contracts for the whole pipeline.

Every stage of the harness consumes and produces one of these models, so a
malformed payload fails at the boundary instead of halfway through retrieval.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- verdicts
class Verdict(str, Enum):
    """Terminal outcome of a request. Anything other than ANSWERED is a refusal."""

    ANSWERED = "answered"
    DECLINED_UNSAFE = "declined_unsafe"
    DECLINED_INJECTION = "declined_injection"
    DECLINED_MALFORMED = "declined_malformed"
    DECLINED_NO_CONTEXT = "declined_no_context"
    DECLINED_UNGROUNDED = "declined_ungrounded"
    ERROR = "error"


class StageStatus(str, Enum):
    OK = "ok"
    RETRIED = "retried"
    DEGRADED = "degraded"
    SKIPPED = "skipped"
    FAILED = "failed"


# ------------------------------------------------------------------ trace types
class Span(BaseModel):
    """One timed stage of the pipeline."""

    name: str
    ms: float
    status: StageStatus = StageStatus.OK
    attempts: int = 1
    detail: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    spans: list[Span] = Field(default_factory=list)
    total_ms: float = 0.0
    budget_ms: int = 200
    within_budget: bool = True

    def stage_ms(self, name: str) -> float:
        return sum(s.ms for s in self.spans if s.name == name)


# ------------------------------------------------------------ retrieval types
class RetrievedChunk(BaseModel):
    chunk_id: str
    parent_id: str
    text: str
    context_text: str = ""
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    strategies: list[str] = Field(default_factory=list)
    lang: str = "eng_Latn"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    marker: int
    chunk_id: str
    parent_id: str
    quote: str
    strategies: list[str] = Field(default_factory=list)
    score: float = 0.0


# --------------------------------------------------------------- guardrail types
class GuardFinding(BaseModel):
    guard: str
    passed: bool
    severity: Literal["info", "warn", "block"] = "info"
    reason: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class GroundingReport(BaseModel):
    score: float = 0.0
    supported_sentences: int = 0
    total_sentences: int = 0
    unsupported: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    citations_valid: bool = True


# ------------------------------------------------------------------- API models
class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)
    mode: Literal["extractive", "llm", "auto"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    language: str | None = None
    include_trace: bool = True


class AskResponse(BaseModel):
    verdict: Verdict
    answer: str
    query: str
    transcript: str | None = None
    generation_mode: str = "extractive"
    citations: list[Citation] = Field(default_factory=list)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    grounding: GroundingReport = Field(default_factory=GroundingReport)
    guards: list[GuardFinding] = Field(default_factory=list)
    trace: Trace = Field(default_factory=Trace)
    latency_ms: float = 0.0
    core_latency_ms: float = 0.0
    stt_latency_ms: float | None = None
    detected_language: str | None = None
    request_id: str = ""
    warnings: list[str] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    text: str
    provider: str
    language: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    raw: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_loaded: bool
    chunks: int
    passages: int
    embedding_model: str
    vector_backend: str
    stt_providers: list[str]
    llm_available: bool
    version: str
