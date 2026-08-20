"""FastAPI application: the HTTP surface over the harness.

Endpoints:

    POST /api/ask        text question -> verified answer + trace
    POST /api/voice      audio upload  -> transcript + verified answer + trace
    GET  /api/health     readiness, index size, which providers are live
    GET  /api/ready      deploy gate: 503 until the index is loaded (see below)
    GET  /api/metrics    rolling P50/P70/P100 measured from real traffic
    GET  /api/config     what the browser needs to know (voice mode, languages)
    GET  /api/report     the committed benchmark report, if present
    GET  /                the demo UI (the built web/ bundle, no dev server needed)

The index is loaded once at startup and warmed before the first request, because a
cold process answers its first query in ~240 ms and every subsequent one in ~6 ms —
serving that first request unwarmed would report a latency the system does not have.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import REPO_ROOT, get_settings
from app.harness.gc_tuning import tune_gc
from app.harness.orchestrator import RagPipeline
from app.retrieval.index_store import RagIndex
from app.schemas import AskRequest, AskResponse, HealthResponse
from app.stt.base import AudioPayload, STTError
from app.stt.registry import NoProviderConfigured

logger = logging.getLogger(__name__)
# The UI is the static export of the Next.js app (web/out). app/static is kept as
# a hand-written fallback for environments where the web bundle was never built.
WEB_OUT_DIR = Path(__file__).resolve().parent.parent / "web" / "out"
STATIC_DIR = Path(__file__).parent / "static"
MAX_LATENCY_SAMPLES = 2_000


class AppState:
    """Process-wide singletons plus a rolling latency window for /api/metrics."""

    pipeline: RagPipeline | None = None
    index_error: str | None = None
    warmup_ms: float = 0.0
    gc_report: dict[str, Any] = {}
    started_at: float = time.time()
    core_latencies: deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
    verdicts: dict[str, int] = {}

    def record(self, response: AskResponse) -> None:
        self.core_latencies.append(response.core_latency_ms)
        key = response.verdict.value
        self.verdicts[key] = self.verdicts.get(key, 0) + 1


state = AppState()


def _percentile(ordered: list[float], p: float) -> float:
    if not ordered:
        return 0.0
    if p >= 100:
        return ordered[-1]
    position = (len(ordered) - 1) * (p / 100)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    try:
        index = RagIndex.load(settings.index_dir)
        pipeline = RagPipeline(index, settings)
        state.warmup_ms = pipeline.warmup(rounds=3)
        # After the index is resident and every hot path has been touched: freeze the
        # long-lived heap. An untuned gen-2 collection traverses ~111k index objects
        # and costs 51 ms median / 72 ms worst — a quarter of the 200 ms budget,
        # landing in whichever unlucky request triggers it. See app/harness/gc_tuning.py.
        state.gc_report = tune_gc()
        state.pipeline = pipeline
        logger.info(
            "ready: %d chunks / %d passages, warmed in %.0f ms, stt=%s, llm=%s",
            index.size,
            index.n_passages,
            state.warmup_ms,
            pipeline.stt_providers or ["browser-only"],
            pipeline.llm.available,
        )
    except Exception as exc:  # noqa: BLE001 - surface via /api/health instead of crashing
        state.index_error = str(exc)
        logger.error("index unavailable: %s", exc)
    yield
    if state.pipeline is not None:
        await state.pipeline.aclose()


app = FastAPI(
    title="Voice RAG over MSMARCO-XI",
    version=__version__,
    description="Voice → STT → hybrid retrieval → grounded answer, under 200 ms.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def require_pipeline() -> RagPipeline:
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"index not loaded: {state.index_error}. Run scripts/ingest.py then "
                "scripts/build_index.py."
            ),
        )
    return state.pipeline


# ------------------------------------------------------------------- endpoints
@app.post("/api/ask", response_model=AskResponse)
async def ask(
    request: AskRequest, pipeline: RagPipeline = Depends(require_pipeline)
) -> AskResponse:
    """Text question. Also the endpoint the browser uses after local transcription."""
    response = await pipeline.answer(request)
    state.record(response)
    return response


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    if state.pipeline is None:
        return HealthResponse(
            status="degraded",
            index_loaded=False,
            chunks=0,
            passages=0,
            embedding_model=settings.embedding_model,
            vector_backend="none",
            stt_providers=[],
            llm_available=False,
            version=__version__,
        )
    stats = state.pipeline.stats()
    return HealthResponse(
        status="ok",
        index_loaded=True,
        chunks=stats["chunks"],
        passages=stats["passages"],
        embedding_model=stats["embedding_model"],
        vector_backend=stats["vector_backend"],
        stt_providers=stats["stt_providers"],
        llm_available=stats["llm_available"],
        version=__version__,
    )


@app.get("/api/ready")
async def ready() -> JSONResponse:
    """Readiness for the platform's health check — 503 until the index is usable.

    Deliberately separate from `/api/health`, which always answers 200 so the UI can
    render *why* it is broken (`index_loaded: false` drives a red indicator in
    PipelineHeader). Two different consumers want two different things from the same
    fact:

    * the browser wants the diagnosis, and a non-2xx would make its client throw and
      leave the banner blank exactly when it matters;
    * the deploy platform wants a gate. If a health check returns 200 while the index
      is missing, the bad release gets promoted and every /api/ask answers 503 — a URL
      that loads but cannot answer, which is the worst way for this to fail.

    So Render and the container HEALTHCHECK point here, the UI points at /api/health,
    and a broken build keeps the last good deploy live instead of replacing it.
    """
    if state.pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": state.index_error or "index not loaded"},
        )
    return JSONResponse({"ready": True, "chunks": state.pipeline.index.size})


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
    """Percentiles measured from this process's real traffic, not a fixture."""
    ordered = sorted(state.core_latencies)
    return {
        "requests": len(ordered),
        "core_latency_ms": {
            "p50": round(_percentile(ordered, 50), 2),
            "p70": round(_percentile(ordered, 70), 2),
            "p90": round(_percentile(ordered, 90), 2),
            "p95": round(_percentile(ordered, 95), 2),
            "p100": round(_percentile(ordered, 100), 2),
        },
        "budget_ms": get_settings().core_budget_ms,
        "within_budget_pct": round(
            100
            * sum(1 for v in ordered if v <= get_settings().core_budget_ms)
            / max(len(ordered), 1),
            2,
        ),
        "verdicts": dict(sorted(state.verdicts.items())),
        "uptime_s": round(time.time() - state.started_at, 1),
        "warmup_ms": state.warmup_ms,
        "gc": state.gc_report,
        "index": state.pipeline.stats() if state.pipeline else {},
    }


@app.get("/api/config")
async def client_config() -> dict[str, Any]:
    settings = get_settings()
    providers = state.pipeline.stt_providers if state.pipeline else []
    return {
        "stt_providers": providers,
        "voice_mode": "server" if providers else "browser",
        "llm_available": bool(state.pipeline and state.pipeline.llm.available),
        "default_mode": settings.generation_mode,
        "budget_ms": settings.core_budget_ms,
        "languages": [
            {"flores": "eng_Latn", "bcp47": "en-IN", "label": "English"},
            {"flores": "hin_Deva", "bcp47": "hi-IN", "label": "हिन्दी"},
            {"flores": "ben_Beng", "bcp47": "bn-IN", "label": "বাংলা"},
            {"flores": "tam_Taml", "bcp47": "ta-IN", "label": "தமிழ்"},
            {"flores": "tel_Telu", "bcp47": "te-IN", "label": "తెలుగు"},
            {"flores": "mar_Deva", "bcp47": "mr-IN", "label": "मराठी"},
        ],
    }


@app.get("/api/report")
async def report() -> JSONResponse:
    path = REPO_ROOT / "reports" / "benchmark.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no benchmark report committed")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.post("/api/voice", response_model=AskResponse)
async def voice(
    audio: UploadFile = File(..., description="recorded audio (webm/opus, wav, mp3 …)"),
    language: str | None = Form(default=None, description="FLORES tag, e.g. hin_Deva"),
    mode: str | None = Form(default=None),
    provider: str | None = Form(default=None, description="sarvam | elevenlabs"),
    include_trace: bool = Form(default=True),
    pipeline: RagPipeline = Depends(require_pipeline),
) -> AskResponse:
    """Voice question: transcribe with Sarvam/ElevenLabs, then run the text path."""
    settings = get_settings()
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio upload")
    if len(data) > settings.stt_max_audio_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"audio too large ({len(data)} bytes, limit {settings.stt_max_audio_bytes})",
        )
    payload = AudioPayload(
        data=data,
        filename=audio.filename or "audio.webm",
        content_type=audio.content_type or "audio/webm",
        language=language,
    )
    if mode is not None and mode not in ("extractive", "llm", "auto"):
        raise HTTPException(status_code=400, detail=f"unknown mode '{mode}'")
    if provider is not None and provider not in ("sarvam", "elevenlabs", "auto"):
        raise HTTPException(status_code=400, detail=f"unknown provider '{provider}'")
    ask_request = AskRequest(
        query="(pending transcription)", mode=mode, include_trace=include_trace
    )
    try:
        response = await pipeline.answer_audio(payload, ask_request, provider=provider)
    except NoProviderConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except STTError as exc:
        raise HTTPException(status_code=502, detail=f"transcription failed: {exc}") from exc
    state.record(response)
    return response


# --------------------------------------------------------------------- static UI
# The UI is the static export of the Next.js app (web/out), mounted at the root
# so its absolute asset paths (/_next/…) resolve against the same origin as /api.
# app/static is a hand-written fallback for environments without a web build.
@app.get("/")
async def index_page() -> FileResponse:
    """Serve the built web bundle when present, else the hand-written fallback."""
    target = WEB_OUT_DIR if WEB_OUT_DIR.exists() else STATIC_DIR
    index = target / "index.html"
    if not index.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "UI not built: run `npm run build` in web/ (or drop an index.html "
                "into app/static/)."
            ),
        )
    return FileResponse(index)


# Registered last, so /api/* still wins. html=True serves index.html for "/".
if WEB_OUT_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_OUT_DIR, html=True), name="ui")
elif STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
