"""Retries and circuit breaking for the calls that leave the process.

Retrying is easy to get wrong in two directions: retrying what cannot succeed
(a 400, a malformed key) wastes the user's time, and hammering a service that is
already failing makes the outage worse. So `retry_async` only retries what the
caller declares retryable, backs off exponentially with jitter, and a
`CircuitBreaker` in front of each provider fails fast once a service is clearly
down — then lets a single probe through after a cooldown to notice recovery.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitOpen(RuntimeError):
    """Raised instead of calling a provider that is currently failing."""


@dataclass
class RetryPolicy:
    attempts: int = 3
    base_delay_ms: float = 60.0
    max_delay_ms: float = 800.0
    jitter: float = 0.3
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    give_up_on: tuple[type[BaseException], ...] = (asyncio.CancelledError, CircuitOpen)
    # Optional finer-grained test: even a `retry_on` type is abandoned when this
    # returns False (e.g. an HTTP 401 that will never start succeeding).
    retry_if: Callable[[BaseException], bool] | None = None

    def delay_for(self, attempt: int) -> float:
        raw = min(self.base_delay_ms * (2 ** (attempt - 1)), self.max_delay_ms)
        spread = raw * self.jitter
        return max(0.0, raw + random.uniform(-spread, spread)) / 1000.0

    def should_retry(self, exc: BaseException) -> bool:
        return self.retry_if is None or bool(self.retry_if(exc))


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    *,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> tuple[T, int]:
    """Run `operation`, retrying per policy. Returns `(result, attempts_used)`."""
    policy = policy or RetryPolicy()
    last: BaseException | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            return await operation(), attempt
        except policy.give_up_on:
            raise
        except policy.retry_on as exc:
            last = exc
            if attempt >= policy.attempts or not policy.should_retry(exc):
                break
            if on_retry:
                on_retry(attempt, exc)
            logger.warning(
                "attempt %d/%d failed (%s); retrying", attempt, policy.attempts, exc
            )
            await asyncio.sleep(policy.delay_for(attempt))
    assert last is not None
    raise last


@dataclass
class CircuitBreaker:
    """Closed -> open after `failure_threshold`, half-open after `reset_after_s`."""

    name: str = "provider"
    failure_threshold: int = 3
    reset_after_s: float = 30.0
    _failures: int = field(default=0, repr=False)
    _opened_at: float = field(default=0.0, repr=False)

    @property
    def state(self) -> str:
        if self._failures < self.failure_threshold:
            return "closed"
        if time.monotonic() - self._opened_at >= self.reset_after_s:
            return "half_open"
        return "open"

    def ensure_closed(self) -> None:
        if self.state == "open":
            waited = time.monotonic() - self._opened_at
            raise CircuitOpen(
                f"{self.name} circuit open ({self._failures} failures, "
                f"retry in {self.reset_after_s - waited:.0f}s)"
            )

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = 0.0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()
            logger.error("%s circuit opened after %d failures", self.name, self._failures)


async def guarded_call(
    breaker: CircuitBreaker,
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
) -> tuple[T, int]:
    """Circuit-breaker + retry, the combination every external call here uses."""
    breaker.ensure_closed()
    try:
        result, attempts = await retry_async(operation, policy)
    except CircuitOpen:
        raise
    except BaseException:
        breaker.record_failure()
        raise
    breaker.record_success()
    return result, attempts


def first_available(candidates: Sequence[tuple[str, bool]]) -> str | None:
    """Pick the first configured provider from an ordered preference list."""
    return next((name for name, configured in candidates if configured), None)
