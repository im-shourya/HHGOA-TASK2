"""Central configuration.

Every tunable lives here so the pipeline can be re-shaped from the environment
without touching code. Values are validated at import time by pydantic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# `Settings` below reads `.env` through pydantic-settings, which populates the
# model but deliberately does NOT touch os.environ. That is the right default
# for this app — nothing here reads os.environ for a secret — but it means any
# *other* tool that reads os.environ cannot see keys kept in `.env`.
#
# The eval harness (`eval/judge.py`) is exactly that case: it imports this module
# specifically for a load_dotenv() side effect, then looks up its judge
# credential in os.environ. Without this call, a key sitting in `.env` is
# invisible to it and the judge-based checks report SKIPPED.
#
# `override=False` so a variable already exported in the shell always wins over
# the file — the usual precedence, and it keeps CI/container env vars authoritative.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:  # python-dotenv is a dev/eval dependency, not a runtime one
    pass

# Retrieval-only sub-budget, in milliseconds: query embedding + vector search +
# sparse search + fusion. Read as a module constant (not a Settings field) because
# the reference harness shipped with the task imports it directly:
#
#     from app.config import LATENCY_BUDGET_MS
#
# The 200 ms figure in the brief covers the whole core path; retrieval is given a
# quarter of it so guardrails, answer composition and grounding verification all
# have room left. `RETRIEVAL_BUDGET_MS` env var overrides it.
LATENCY_BUDGET_MS = int(os.getenv("RETRIEVAL_BUDGET_MS", "50"))
# End-to-end budget for the graded core path, mirrored into Settings.core_budget_ms.
CORE_BUDGET_MS = int(os.getenv("CORE_BUDGET_MS", "200"))


class Settings(BaseSettings):
    """Runtime settings, populated from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------ paths
    corpus_dir: Path = DATA_DIR / "corpus"
    index_dir: Path = DATA_DIR / "index"

    # -------------------------------------------------------------- embedding
    # Static (token-lookup) embeddings: no transformer forward pass, so a query
    # is encoded in well under a millisecond. This is what makes the <200 ms
    # end-to-end budget reachable on commodity CPU.
    # `potion-retrieval-32M` (512-d, 129 MB) is the default: purpose-built for
    # retrieval and small enough for a 512 MB container. Swap to
    # `minishlab/potion-multilingual-128M` (256-d, 512 MB) for cross-lingual
    # dense retrieval — see README "Embedding profiles".
    embedding_model: str = "minishlab/potion-retrieval-32M"
    embedding_dim: int = 512

    # ------------------------------------------------------------- retrieval
    dense_top_k: int = 40
    sparse_top_k: int = 40
    fusion_top_k: int = 12
    context_top_k: int = 4
    rrf_k: int = 60
    mmr_lambda: float = 0.72
    # Retrieval guardrail: below this fused-confidence floor we decline to answer.
    # 0.67 is measured, not guessed — see reports/benchmark.md "calibration": it is
    # the threshold that best separates answerable queries (median 0.83) from a
    # holdout set whose evidence was never indexed (median 0.55).
    min_retrieval_score: float = 0.67
    # Minimum gap between the best and second-best fused score. Computed and
    # reported by hybrid.py as `margin`, but deliberately NOT gated on: sweeping it
    # over the calibration holdout only ever traded answer rate for nothing (see
    # README "Gates that were measured and rejected"). Kept as an observable.
    min_score_margin: float = 0.02
    vector_backend: Literal["auto", "flat", "hnsw"] = "auto"
    hnsw_threshold: int = 120_000

    # ------------------------------------------------------------- generation
    generation_mode: Literal["extractive", "llm", "auto"] = "extractive"
    llm_model: str = "deepseek-r1-distill-llama-70b"
    llm_max_tokens: int = 800
    llm_timeout_ms: int = 15_000
    groq_api_key: str | None = None
    answer_max_words: int = 70

    # ------------------------------------------------------------- guardrails
    min_grounding_score: float = 0.55
    max_query_chars: int = 512
    min_query_chars: int = 2

    # ----------------------------------------------------------------- budget
    # Deadline for the graded core path (guard -> retrieve -> generate -> verify).
    core_budget_ms: int = CORE_BUDGET_MS
    # Sub-budget for embed + vector search + sparse search + fusion alone.
    retrieval_budget_ms: int = LATENCY_BUDGET_MS
    stage_timeout_ms: int = 120

    # -------------------------------------------------------------------- stt
    stt_provider: Literal["sarvam", "elevenlabs", "auto", "none"] = "auto"
    sarvam_api_key: str | None = None
    sarvam_stt_model: str = "saaras:v3"
    sarvam_base_url: str = "https://api.sarvam.ai"
    elevenlabs_api_key: str | None = None
    elevenlabs_stt_model: str = "scribe_v1"
    elevenlabs_base_url: str = "https://api.elevenlabs.io"
    stt_timeout_ms: int = 15_000
    stt_max_audio_bytes: int = 12 * 1024 * 1024

    # ------------------------------------------------------------------ misc
    enable_query_cache: bool = False
    query_cache_size: int = 512
    log_level: str = "INFO"
    cors_origins: str = "*"

    @field_validator("corpus_dir", "index_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so settings are parsed exactly once per process."""
    return Settings()
