"""Speech-to-text provider interface.

Two hosted providers are wired in behind one interface so the pipeline can fail
over between them (and so switching costs an env var, not a rewrite):

* **Sarvam** `saaras:v3` — built for Indian languages, which is the right default
  for a corpus whose queries arrive in Hindi.
* **ElevenLabs** `scribe_v1` — strong on English and broadly multilingual.

A third path needs no provider at all: the browser's Web Speech API transcribes
locally and posts text, which keeps the demo working with zero keys and zero
network cost. See `app/static/app.js`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.schemas import TranscriptionResult


class STTError(RuntimeError):
    """Provider-level failure. Retryable unless `permanent` is set."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


@dataclass
class AudioPayload:
    data: bytes
    filename: str = "audio.webm"
    content_type: str = "audio/webm"
    language: str | None = None

    @property
    def size(self) -> int:
        return len(self.data)


class STTProvider(Protocol):
    name: str

    @property
    def configured(self) -> bool: ...

    async def transcribe(self, audio: AudioPayload) -> TranscriptionResult: ...
