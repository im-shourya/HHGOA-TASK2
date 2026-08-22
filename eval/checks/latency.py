"""Aggregates the timings already collected during eval/pipeline.py's real
retrieval + generation calls -- makes no calls of its own. Retrieval is
graded against the target's OPTIONAL app.config.LATENCY_BUDGET_MS if it
declares one (see eval/target.py's interface contract); if it doesn't,
falls back to this suite's own default (50ms, this suite's original
target project's own value, chosen as a reasonable retrieval-latency bar
in general -- override via EVAL_RETRIEVAL_LATENCY_BUDGET_MS). Generation
has no equivalent budget expectation on principle: a target calling a
hosted API is bound by real network latency no config value can shrink,
so GENERATION_LATENCY_TARGET_MS below is purely this suite's own
reference point for the report, never a constraint on the target itself
-- override with EVAL_GENERATION_LATENCY_TARGET_MS if 1500ms isn't the
right bar for your use case.
"""
import os
import statistics

from eval import target
from eval.pipeline import ExampleResult

GENERATION_LATENCY_TARGET_MS = float(os.environ.get("EVAL_GENERATION_LATENCY_TARGET_MS", 1500))
DEFAULT_RETRIEVAL_LATENCY_BUDGET_MS = float(os.environ.get("EVAL_RETRIEVAL_LATENCY_BUDGET_MS", 50))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    return values[f] if f == c else values[f] + (k - f) * (values[c] - values[f])


def _block(values: list[float]) -> dict:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "avg_ms": round(statistics.mean(values), 2),
        "p50_ms": round(_percentile(values, 50), 2),
        "p95_ms": round(_percentile(values, 95), 2),
        "p99_ms": round(_percentile(values, 99), 2),
    }


def run(results: list[ExampleResult]) -> dict:
    latency_budget_ms = target.optional_config("LATENCY_BUDGET_MS", default=DEFAULT_RETRIEVAL_LATENCY_BUDGET_MS)

    usable = [r for r in results if r.error is None]

    embed_ms = [r.embed_ms_en for r in usable] + [r.embed_ms_hi for r in usable]
    search_ms = [r.search_ms_en for r in usable] + [r.search_ms_hi for r in usable]
    retrieval_total_ms = [r.embed_ms_en + r.search_ms_en for r in usable]
    generation_ms = [r.generation_ms for r in usable if r.generation_ms > 0]

    retrieval_p95 = _percentile(retrieval_total_ms, 95)
    generation_p95 = _percentile(generation_ms, 95)

    return {
        "check": "latency",
        "num_evaluated": len(usable),
        "embed": _block(embed_ms),
        "search": _block(search_ms),
        "retrieval_total": _block(retrieval_total_ms),
        "generation": _block(generation_ms),
        "retrieval_latency_budget_ms": latency_budget_ms,
        "retrieval_within_budget": retrieval_p95 <= latency_budget_ms,
        "generation_latency_target_ms": GENERATION_LATENCY_TARGET_MS,
        "generation_within_target": generation_p95 <= GENERATION_LATENCY_TARGET_MS,
    }
