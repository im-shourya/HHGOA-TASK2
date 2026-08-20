"""Provider registry: selection, retry, failover, circuit breaking.

Order of preference comes from `STT_PROVIDER` (`auto` = Sarvam first, since the
corpus queries are Indic, then ElevenLabs). Within a provider, transient failures
are retried with backoff; a permanent one (bad key, unsupported codec) skips
straight to the next provider instead of burning the retry budget.

If nothing is configured the caller gets `NoProviderConfigured`, which the API
turns into an actionable 503 telling the client to use browser transcription — a
missing key should degrade the demo, not break it.
"""

from __future__ import annotations

import logging
import time

from app.config import Settings
from app.harness.retry import CircuitBreaker, CircuitOpen, RetryPolicy, guarded_call
from app.schemas import TranscriptionResult
from app.stt.base import AudioPayload, STTError, STTProvider
from app.stt.elevenlabs import ElevenLabsSTT
from app.stt.sarvam import SarvamSTT

logger = logging.getLogger(__name__)


class NoProviderConfigured(RuntimeError):
    pass


class STTRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._providers: dict[str, STTProvider] = {
            "sarvam": SarvamSTT(settings),
            "elevenlabs": ElevenLabsSTT(settings),
        }
        self._breakers = {
            name: CircuitBreaker(name=f"stt:{name}", failure_threshold=3, reset_after_s=30.0)
            for name in self._providers
        }
        self._policy = RetryPolicy(
            attempts=2,
            base_delay_ms=120.0,
            max_delay_ms=600.0,
            retry_on=(STTError,),
            # A permanent failure (bad key, unsupported codec) is not worth a
            # second attempt — fail over to the next provider instead.
            retry_if=lambda exc: isinstance(exc, STTError) and not exc.permanent,
        )

    # ------------------------------------------------------------------ public
    @property
    def available(self) -> list[str]:
        return [name for name, p in self._providers.items() if p.configured]

    def order(self, preferred: str | None = None) -> list[str]:
        wanted = preferred or self.settings.stt_provider
        ranked = ["sarvam", "elevenlabs"] if wanted in ("auto", "none", None) else [
            wanted,
            *[n for n in ("sarvam", "elevenlabs") if n != wanted],
        ]
        return [name for name in ranked if self._providers[name].configured]

    async def transcribe(
        self, audio: AudioPayload, preferred: str | None = None
    ) -> TranscriptionResult:
        candidates = self.order(preferred)
        if not candidates:
            raise NoProviderConfigured(
                "no speech-to-text provider configured: set SARVAM_API_KEY or "
                "ELEVENLABS_API_KEY, or transcribe in the browser and POST /api/ask"
            )
        if audio.size > self.settings.stt_max_audio_bytes:
            raise STTError(
                f"audio too large: {audio.size} bytes > {self.settings.stt_max_audio_bytes}",
                permanent=True,
            )

        errors: list[str] = []
        for name in candidates:
            provider = self._providers[name]
            started = time.perf_counter()

            async def call() -> TranscriptionResult:
                return await provider.transcribe(audio)

            try:
                result, attempts = await guarded_call(
                    self._breakers[name], call, self._policy
                )
            except CircuitOpen as exc:
                errors.append(str(exc))
                continue
            except STTError as exc:
                errors.append(f"{name}: {exc}")
                logger.warning("stt provider %s failed: %s", name, exc)
                continue
            result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            result.attempts = attempts
            return result

        raise STTError("all speech-to-text providers failed: " + " | ".join(errors))

    async def aclose(self) -> None:
        for provider in self._providers.values():
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                await closer()
