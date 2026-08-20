"""Measure end-to-end retrieval latency (embed + vector search) against the
budget defined in app/config.py.

Usage:
    python -m app.benchmark [n_queries]

This is the harness shipped with the task, kept runnable as-is. Two things were
added rather than changed:

* **P70 and P100 columns.** The brief asks for P50 / P70 / P100; the original
  table printed P50 / P95 / P99. Both are shown now — nothing was removed.
* **A second section for the full core path.** The original measures retrieval
  only (embed + search) against a 50 ms sub-budget. The 200 ms figure in the brief
  covers everything through to final output, so section 2 runs the same queries
  through `RagPipeline` — input guardrails, retrieval, answer composition,
  grounding verification — and checks that. A retrieval-only PASS says nothing
  about the number that is actually graded.

Queries come from the ingested MSMARCO-XI slice when `data/corpus/` is present, so
the table reflects the corpus being served. `--reference-queries` forces the
original hard-coded list instead. Both are warmed first: cold-start cost is real,
but averaging one 240 ms process-start into a P100 describes the interpreter, not
the pipeline.

For retrieval quality, guardrail accuracy and threshold calibration, see
`scripts/benchmark.py`, which writes reports/benchmark.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys

from app.config import LATENCY_BUDGET_MS, get_settings
from app.harness.gc_tuning import tune_gc
from app.retriever import search, stats, warmup

QUERIES = [
    "What is FAISS used for?",
    "How does HNSW indexing work?",
    "What is retrieval augmented generation?",
    "Which embedding model is fast on CPU?",
    "How do you reduce RAG latency?",
    "What does efSearch control?",
    "Why normalize embeddings before indexing?",
    "What are the stages of a RAG pipeline?",
]

PERCENTILES = (50, 70, 95, 99, 100)


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def _safe(text: str, limit: int = 64) -> str:
    """Truncate and drop characters the active console cannot encode.

    Hindi queries are part of the corpus and a Windows cp1252 console cannot print
    them; a benchmark should not die on its own progress output.
    """
    clipped = text if len(text) <= limit else text[: limit - 1] + "…"
    encoding = sys.stdout.encoding or "utf-8"
    return clipped.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _table(rows: list[tuple[str, list[float]]]) -> None:
    header = f"{'stage':<12}{'avg':>8}" + "".join(f"{'p' + str(p):>8}" for p in PERCENTILES)
    print(f"{header}   (ms)")
    for name, values in rows:
        line = f"{name:<12}{statistics.mean(values):>8.2f}"
        line += "".join(f"{percentile(values, p):>8.2f}" for p in PERCENTILES)
        print(line)


def load_queries(reference_only: bool = False) -> tuple[list[str], str]:
    """Real corpus queries when available, else the reference list."""
    if reference_only:
        return list(QUERIES), "reference list (hard-coded)"
    corpus = get_settings().corpus_dir / "eval_queries.jsonl"
    if not corpus.exists():
        return list(QUERIES), f"reference list ({corpus} not found)"
    queries: list[str] = []
    with corpus.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("eng_query", "query"):
                value = (row.get(key) or "").strip()
                # A handful of dataset translations are degenerate MT loops of
                # several thousand characters; they are not questions.
                if 0 < len(value) <= 400:
                    queries.append(value)
    if not queries:
        return list(QUERIES), "reference list (corpus held no usable queries)"
    return queries, f"{corpus.name} ({len(queries)} queries from MSMARCO-XI)"


def measure_retrieval(queries: list[str], n: int, top_k: int) -> dict[str, list[float]]:
    total_ms: list[float] = []
    embed_ms: list[float] = []
    search_ms: list[float] = []
    worst = (0.0, "")
    for i in range(n):
        query = queries[i % len(queries)]
        resp = search(query, top_k=top_k)
        total_ms.append(resp.total_ms)
        embed_ms.append(resp.embed_ms)
        search_ms.append(resp.search_ms)
        if resp.total_ms > worst[0]:
            worst = (resp.total_ms, query)
    return {
        "embed": embed_ms,
        "search": search_ms,
        "total": total_ms,
        "_worst": worst,  # type: ignore[dict-item]
    }


async def measure_core(queries: list[str], n: int) -> dict[str, list[float]]:
    """Full graded path: guardrails → retrieve → generate → verify."""
    from app.harness.orchestrator import RagPipeline
    from app.retriever import get_retriever
    from app.schemas import AskRequest

    retriever = get_retriever()
    settings = get_settings().model_copy(update={"enable_query_cache": False})
    pipeline = RagPipeline(retriever.index, settings, embedder=retriever.embedder)
    pipeline.warmup(rounds=3)

    core_ms: list[float] = []
    wall_ms: list[float] = []
    verdicts: dict[str, int] = {}
    worst = (0.0, "")
    try:
        for i in range(n):
            query = queries[i % len(queries)]
            response = await pipeline.answer(
                AskRequest(query=query, mode="extractive", include_trace=True)
            )
            core_ms.append(response.core_latency_ms)
            wall_ms.append(response.latency_ms)
            verdicts[response.verdict.value] = verdicts.get(response.verdict.value, 0) + 1
            if response.core_latency_ms > worst[0]:
                worst = (response.core_latency_ms, query)
    finally:
        await pipeline.aclose()
    return {
        "core": core_ms,
        "wall": wall_ms,
        "_verdicts": verdicts,  # type: ignore[dict-item]
        "_worst": worst,  # type: ignore[dict-item]
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Latency benchmark against the configured budgets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("n", nargs="?", type=int, default=50, help="number of queries to run")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reference-queries", action="store_true",
                        help="use the hard-coded query list instead of the corpus")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip the full-pipeline section")
    args = parser.parse_args()
    n = max(args.n, 1)

    print("Warming up (index load + model load + first inference)...")
    try:
        warm_ms = warmup(rounds=5)
    except FileNotFoundError as exc:
        print(f"\nno index: {exc}", file=sys.stderr)
        print("Build one first:\n"
              "  python scripts/ingest.py --languages hin_Deva --queries 300\n"
              "  python scripts/build_index.py", file=sys.stderr)
        return 2
    # Same tuning the API applies at startup, for the same reason: an untraversed
    # gen-2 collection over the resident index costs ~51 ms and would show up here
    # as an unexplained P100 outlier. Benchmarking an untuned process would report
    # a tail the deployed service does not have.
    gc_report = tune_gc()

    index = stats()
    queries, source = load_queries(args.reference_queries)
    print(f"warmed in {warm_ms} ms | {index['chunks']} chunks / {index['passages']} passages "
          f"| {index['vector_backend']} | {index['embedding_model']}")
    print(f"gc: froze {gc_report['frozen_objects']:,} long-lived objects "
          f"(thresholds {gc_report['threshold_before']} -> {gc_report['threshold_after']})")
    print(f"queries: {source}")

    # ------------------------------------------------- 1. retrieval sub-budget
    retrieval = measure_retrieval(queries, n, args.top_k)
    worst_total, worst_query = retrieval.pop("_worst")  # type: ignore[misc]
    print(f"\nSection 1 — retrieval only (embed + hybrid search), {n} queries\n")
    _table([(name, retrieval[name]) for name in ("embed", "search", "total")])

    p95_total = percentile(retrieval["total"], 95)
    p100_total = percentile(retrieval["total"], 100)
    print(f"\nRetrieval budget: {LATENCY_BUDGET_MS}ms | p95 total: {p95_total:.2f}ms "
          f"| p100 total: {p100_total:.2f}ms")
    print(f"slowest: {worst_total:.2f}ms on {_safe(worst_query)!r}")
    retrieval_ok = p95_total <= LATENCY_BUDGET_MS
    print("PASS: within budget" if retrieval_ok
          else "FAIL: over budget -- see README 'Tuning latency' section")

    # ------------------------------------------------------ 2. full core path
    core_ok = True
    if not args.retrieval_only:
        core_budget = get_settings().core_budget_ms
        core = asyncio.run(measure_core(queries, n))
        verdicts = core.pop("_verdicts")  # type: ignore[misc]
        worst_core, worst_core_query = core.pop("_worst")  # type: ignore[misc]
        print(f"\nSection 2 — full core path (guard -> retrieve -> generate -> verify), "
              f"{n} queries\n")
        _table([("core", core["core"]), ("wall", core["wall"])])

        p100_core = percentile(core["core"], 100)
        within = 100 * sum(1 for v in core["core"] if v <= core_budget) / len(core["core"])
        print(f"\nCore budget: {core_budget}ms | p50 {percentile(core['core'], 50):.2f}ms "
              f"| p70 {percentile(core['core'], 70):.2f}ms | p100 {p100_core:.2f}ms "
              f"| within budget {within:.1f}%")
        print(f"slowest: {worst_core:.2f}ms on {_safe(worst_core_query)!r}")
        print("verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
        # P100, not P95: the brief asks for the worst case, so that is what is gated.
        core_ok = p100_core <= core_budget
        print("PASS: within budget" if core_ok
              else "FAIL: over budget -- see README 'Tuning latency' section")

    print("\nSpeech-to-text is a network call to Sarvam/ElevenLabs and is excluded "
          "from both budgets by design; it is reported separately in reports/benchmark.md.")
    return 0 if (retrieval_ok and core_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
