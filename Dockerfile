# Multi-stage: build the UI with Node, serve everything from one Python process.
#
# The result is a single container that answers both /api/* and / — no CORS, no
# second deploy, no static host to keep in sync with the API.
#
# Two things happen at build time rather than at boot, because a cold container
# that downloads a model and builds an index takes ~90 s to become useful and
# free-tier hosts restart often:
#
#   1. The embedding model is fetched and baked into the image.
#   2. The vector + BM25 index is built from the committed corpus.
#
# So the deployed process starts, mmaps an index that is already there, warms up,
# freezes the GC, and serves. `data/index/` is deliberately *not* copied from the
# build context (see .dockerignore) — it is derived data and is rebuilt here, so
# the image can never ship an index that disagrees with the corpus beside it.

# ----------------------------------------------------------- stage 1: the UI
FROM node:22-alpine AS web
WORKDIR /web

# Copy manifests first: this layer is cached unless the dependency set changes.
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
# `output: "export"` in next.config.ts -> a static bundle in /web/out.
RUN npm run build


# ------------------------------------------------------- stage 2: the service
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keep the HF cache on a fixed path so the model baked in below is the one
    # found at runtime, whatever HOME ends up being.
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    # Static embeddings are memory-bandwidth bound, not compute bound. Letting
    # BLAS spin up a thread per core adds contention and context switches to a
    # 0.2 ms operation; one thread is measurably faster here and keeps latency
    # predictable on a shared vCPU.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Source and corpus. The corpus is committed; the index is not.
COPY app/ app/
COPY scripts/ scripts/
COPY data/corpus/ data/corpus/

# Build the index and bake the embedding model into the image layer.
RUN python scripts/build_index.py && \
    python -c "from app.retrieval.embedder import get_embedder; get_embedder()" && \
    python -c "from app.retrieval.index_store import RagIndex; from app.config import get_settings; \
i = RagIndex.load(get_settings().index_dir); print(f'index ok: {i.size} chunks / {i.n_passages} passages')"

# The built UI, and the committed benchmark report that /api/report serves.
COPY --from=web /web/out web/out
COPY reports/ reports/

# Drop privileges. Done after the build steps so they can write data/index.
RUN useradd --create-home --uid 10001 app && chown -R app:app /srv /opt/hf
USER app

EXPOSE 8000
# Gate on /api/ready, which returns 503 until the index is loaded, so an image that
# boots but cannot answer reports unhealthy instead of pretending to be up.
# /api/health is the informational endpoint and always answers 200 — the wrong thing
# to gate on. urlopen raises on 503, and any exception here is already a non-zero
# exit, which is the unhealthy signal; the try/except only keeps the log readable.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys,os; \
url=f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/api/ready'; \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)" \
    || exit 1

# One worker on purpose: each would load its own copy of the index and the
# embedding model. Concurrency comes from asyncio, and the core path is CPU-bound
# for ~6 ms, so a second worker buys contention rather than throughput.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log"]
