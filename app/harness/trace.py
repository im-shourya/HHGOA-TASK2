"""Timing, spans and the deadline budget.

Every stage runs inside `recorder.span(...)`, which is what makes the latency
numbers in the README a property of the system rather than a stopwatch held by
hand. `remaining_ms()` turns the same data into control flow: a stage can ask how
much of the 200 ms is left and choose a cheaper path instead of blowing it.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.schemas import Span, StageStatus, Trace


@dataclass
class SpanHandle:
    """Mutable handle a stage uses to annotate its own span."""

    name: str
    status: StageStatus = StageStatus.OK
    attempts: int = 1
    detail: dict[str, Any] = field(default_factory=dict)

    def mark(self, status: StageStatus, **detail: Any) -> None:
        self.status = status
        self.detail.update(detail)

    def note(self, **detail: Any) -> None:
        self.detail.update(detail)


class TraceRecorder:
    def __init__(self, budget_ms: int = 200) -> None:
        self.budget_ms = budget_ms
        self.spans: list[Span] = []
        self._start = time.perf_counter()

    @contextmanager
    def span(self, name: str) -> Iterator[SpanHandle]:
        handle = SpanHandle(name=name)
        started = time.perf_counter()
        try:
            yield handle
        except Exception:
            handle.status = StageStatus.FAILED
            raise
        finally:
            self.spans.append(
                Span(
                    name=name,
                    ms=round((time.perf_counter() - started) * 1000, 3),
                    status=handle.status,
                    attempts=handle.attempts,
                    detail=handle.detail,
                )
            )

    def record(self, name: str, ms: float, status: StageStatus = StageStatus.OK, **detail: Any) -> None:
        """Record a span measured elsewhere (e.g. an external STT call)."""
        self.spans.append(Span(name=name, ms=round(ms, 3), status=status, detail=detail))

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def remaining_ms(self, exclude: tuple[str, ...] = ()) -> float:
        """Budget left, optionally ignoring stages that are outside the budget."""
        spent = self.elapsed_ms - sum(
            span.ms for span in self.spans if span.name in exclude
        )
        return self.budget_ms - spent

    def finish(self, exclude: tuple[str, ...] = ()) -> Trace:
        core = sum(span.ms for span in self.spans if span.name not in exclude)
        return Trace(
            spans=list(self.spans),
            total_ms=round(core, 3),
            budget_ms=self.budget_ms,
            within_budget=core <= self.budget_ms,
        )
