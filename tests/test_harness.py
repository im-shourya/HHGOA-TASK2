"""The harness: retries, circuit breaking, provider selection, GC tuning.

`retry_async` is deliberately narrow — it retries only what the caller declares
retryable — because the two ways to get retries wrong are opposites: retrying a 401
wastes the user's whole budget on a call that cannot succeed, and hammering a service
that is already failing makes the outage worse. Both directions are tested here.

Async cases use `asyncio.run` inside sync tests rather than pytest-asyncio, so the
suite has one fewer plugin to install in CI.
"""

from __future__ import annotations

import asyncio

import pytest

from app.harness.retry import (
    CircuitBreaker,
    CircuitOpen,
    RetryPolicy,
    first_available,
    guarded_call,
    retry_async,
)

FAST = RetryPolicy(attempts=3, base_delay_ms=0.0, max_delay_ms=0.0, jitter=0.0)


def flaky(fail_times: int, exc=RuntimeError("boom")):
    """An operation that fails `fail_times` times, then succeeds."""
    state = {"calls": 0}

    async def operation():
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc
        return "ok"

    return operation, state


class TestRetryAsync:
    def test_returns_immediately_when_the_call_succeeds(self):
        operation, state = flaky(0)
        result, attempts = asyncio.run(retry_async(operation, FAST))
        assert (result, attempts, state["calls"]) == ("ok", 1, 1)

    def test_retries_until_success(self):
        operation, state = flaky(2)
        result, attempts = asyncio.run(retry_async(operation, FAST))
        assert (result, attempts, state["calls"]) == ("ok", 3, 3)

    def test_raises_the_last_error_after_exhausting_attempts(self):
        operation, state = flaky(99)
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(retry_async(operation, FAST))
        assert state["calls"] == 3, "must not exceed policy.attempts"

    def test_does_not_retry_an_error_outside_retry_on(self):
        policy = RetryPolicy(attempts=3, base_delay_ms=0.0, retry_on=(TimeoutError,))
        operation, state = flaky(99, ValueError("bad request"))
        with pytest.raises(ValueError):
            asyncio.run(retry_async(operation, policy))
        assert state["calls"] == 1

    def test_retry_if_abandons_an_error_that_cannot_start_succeeding(self):
        """A 401 is the shape of failure retries make worse, not better."""
        policy = RetryPolicy(
            attempts=5,
            base_delay_ms=0.0,
            retry_on=(RuntimeError,),
            retry_if=lambda exc: "401" not in str(exc),
        )
        operation, state = flaky(99, RuntimeError("401 unauthorized"))
        with pytest.raises(RuntimeError):
            asyncio.run(retry_async(operation, policy))
        assert state["calls"] == 1

    def test_cancellation_is_never_retried(self):
        operation, state = flaky(99, asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(retry_async(operation, FAST))
        assert state["calls"] == 1

    def test_on_retry_is_notified_once_per_retry_not_per_attempt(self):
        seen: list[int] = []
        operation, _ = flaky(2)
        asyncio.run(retry_async(operation, FAST, on_retry=lambda n, exc: seen.append(n)))
        assert seen == [1, 2], "the successful attempt is not a retry"


class TestBackoff:
    def test_delay_grows_exponentially_and_is_capped(self):
        policy = RetryPolicy(base_delay_ms=100.0, max_delay_ms=250.0, jitter=0.0)
        assert [policy.delay_for(n) for n in (1, 2, 3, 4)] == [0.1, 0.2, 0.25, 0.25]

    def test_jitter_stays_within_its_band_and_never_goes_negative(self):
        policy = RetryPolicy(base_delay_ms=100.0, max_delay_ms=100.0, jitter=0.3)
        samples = [policy.delay_for(1) for _ in range(200)]
        assert all(0.07 - 1e-9 <= s <= 0.13 + 1e-9 for s in samples)
        assert len(set(samples)) > 1, "jitter must actually vary, or retries synchronise"


class TestCircuitBreaker:
    def test_starts_closed(self):
        assert CircuitBreaker().state == "closed"

    def test_opens_on_the_threshold_failure(self):
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            breaker.record_failure()
        assert breaker.state == "closed"
        breaker.record_failure()
        assert breaker.state == "open"

    def test_an_open_circuit_fails_fast_with_a_recovery_hint(self):
        breaker = CircuitBreaker(name="sarvam", failure_threshold=1, reset_after_s=30.0)
        breaker.record_failure()
        with pytest.raises(CircuitOpen, match="sarvam"):
            breaker.ensure_closed()

    def test_half_opens_after_the_cooldown_so_recovery_is_noticed(self):
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=0.0)
        breaker.record_failure()
        assert breaker.state == "half_open"
        breaker.ensure_closed()  # a single probe is allowed through

    def test_success_closes_it_again(self):
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=0.0)
        breaker.record_failure()
        breaker.record_success()
        assert breaker.state == "closed"

    def test_a_partial_run_of_failures_does_not_accumulate_into_an_outage(self):
        """Intermittent failures must not open the circuit over hours of uptime."""
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(10):
            breaker.record_failure()
            breaker.record_failure()
            breaker.record_success()
        assert breaker.state == "closed"


class TestGuardedCall:
    def test_records_success_and_returns_the_attempt_count(self):
        breaker = CircuitBreaker(failure_threshold=2)
        operation, _ = flaky(1)
        result, attempts = asyncio.run(guarded_call(breaker, operation, FAST))
        assert (result, attempts) == ("ok", 2)
        assert breaker.state == "closed"

    def test_a_fully_failed_call_counts_as_one_failure_not_one_per_attempt(self):
        """Otherwise a single flaky call trips a threshold-3 breaker on its own."""
        breaker = CircuitBreaker(failure_threshold=3)
        operation, state = flaky(99)
        with pytest.raises(RuntimeError):
            asyncio.run(guarded_call(breaker, operation, FAST))
        assert state["calls"] == 3
        assert breaker.state == "closed"

    def test_stops_calling_the_provider_once_the_circuit_is_open(self):
        breaker = CircuitBreaker(failure_threshold=1, reset_after_s=30.0)
        breaker.record_failure()
        operation, state = flaky(0)
        with pytest.raises(CircuitOpen):
            asyncio.run(guarded_call(breaker, operation, FAST))
        assert state["calls"] == 0, "an open circuit must not reach the network"


class TestFirstAvailable:
    def test_prefers_the_earliest_configured_provider(self):
        assert first_available([("sarvam", False), ("elevenlabs", True)]) == "elevenlabs"

    def test_respects_the_declared_order(self):
        assert first_available([("sarvam", True), ("elevenlabs", True)]) == "sarvam"

    def test_returns_none_when_nothing_is_configured(self):
        assert first_available([("sarvam", False), ("elevenlabs", False)]) is None
        assert first_available([]) is None


class TestGcTuning:
    """Gen-2 collections were a measured source of P100 latency, not a guess."""

    def test_freezing_reports_what_it_moved_and_is_reversible(self):
        from app.harness.gc_tuning import tune_gc, untune_gc

        report = tune_gc()
        try:
            assert report["frozen_objects"] > 0
            assert report["enabled"] in (True, False)
        finally:
            untune_gc()

    def test_tuning_twice_is_safe(self):
        from app.harness.gc_tuning import tune_gc, untune_gc

        try:
            tune_gc()
            tune_gc()
        finally:
            untune_gc()
            untune_gc()
