"""Garbage-collector tuning for a latency-sensitive request path.

The problem, measured rather than assumed. The loaded index is ~18k `Chunk`
objects plus their dicts and interned strings — all long-lived, none of it garbage.
CPython's generational collector does not know that: every gen-2 pass traverses the
whole graph. On this index that pass costs 50–65 ms, and it lands wherever it lands:

    query 'what is basic? uni', n=300, no tuning
        p50   4.30 ms
        p99  10.69 ms
        p100 66.41 ms      <- a gen-2 collection inside the request
        2 requests over 20 ms

That is a quarter of the 200 ms budget consumed by bookkeeping for objects that
will live until the process exits.

`gc.disable()` "fixes" it and is the wrong fix: a long-running server that never
collects will leak reference cycles until it is OOM-killed. The right fix is
`gc.freeze()` (CPython 3.7+, the trick Instagram published): move everything
allocated so far into a permanent generation that is never traversed again, then
let the collector go on working normally for whatever the request path allocates.
Per-request garbage is still collected; the index is simply no longer re-scanned.

Call `tune_gc()` once, after the index is loaded and warmed and before serving.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Defaults are (700, 10, 10). Gen-0 fires every 700 net allocations, which for a
# path that allocates numpy scratch and a few hundred dicts per request means
# several collections per second. After the freeze the young generations hold only
# per-request garbage, so collecting them less often is nearly free.
GEN0_THRESHOLD = 20_000
GEN1_THRESHOLD = 25
GEN2_THRESHOLD = 25


def tune_gc(freeze: bool = True) -> dict[str, Any]:
    """Collect once, freeze the long-lived heap, and relax the thresholds.

    Returns what it did, so startup logs and /api/metrics can show it rather than
    leaving a reader to guess whether it ran.
    """
    before = gc.get_threshold()
    collected = gc.collect(2)
    frozen = 0
    if freeze:
        gc.freeze()
        frozen = gc.get_freeze_count()
    gc.set_threshold(GEN0_THRESHOLD, GEN1_THRESHOLD, GEN2_THRESHOLD)
    report = {
        "collected": collected,
        "frozen_objects": frozen,
        "threshold_before": list(before),
        "threshold_after": list(gc.get_threshold()),
        "enabled": gc.isenabled(),
    }
    logger.info(
        "gc tuned: froze %d objects, collected %d, thresholds %s -> %s",
        frozen,
        collected,
        before,
        gc.get_threshold(),
    )
    return report


def untune_gc() -> None:
    """Undo `tune_gc`. Exists so tests can assert against an untuned baseline."""
    gc.unfreeze()
    gc.set_threshold(700, 10, 10)
