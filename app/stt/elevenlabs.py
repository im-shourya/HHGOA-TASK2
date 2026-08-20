"""ElevenLabs speech-to-text (`POST /v1/speech-to-text`).

Contract per elevenlabs.io/docs: `xi-api-key` header, multipart `file` plus a
required `model_id`; `language_code` is ISO-639 and auto-detected when omitted.
Response: `{text, language_code, language_probability, words, ...}`.
"""

from __future__ import annotations

import logging

import httpx

from app.config import Settings
from app.schemas import TranscriptionResult
from app.stt.base import AudioPayload, STTError

logger = logging.getLogger(__name__)

FLORES_TO_ISO: dict[str, str] = {
    "hin_Deva": "hin", "ben_Beng": "ben", "guj_Gujr": "guj", "kan_Knda": "kan",
    "mal_Mlym": "mal", "mar_Deva": "mar", "ory_Orya": "ory", "pan_Guru": "pan",
    "tam_Taml": "tam", "tel_Telu": "tel", "urd_Arab": "urd", "asm_Beng": "asm",
    "npi_Deva": "nep", "san_Deva": "san", "eng_Latn": "eng",
}


def to_iso_language(language: str | None) -> str | None:
    if not language:
        return None
    return FLORES_TO_ISO.get(language, language if len(language) in (2, 3) else None)


class ElevenLabsSTT:
    name = "elevenlabs"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self.settings.elevenlabs_api_key)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.elevenlabs_base_url,
                timeout=httpx.Timeout(self.settings.stt_timeout_ms / 1000),
                headers={"xi-api-key": self.settings.elevenlabs_api_key or ""},
            )
        return self._client

    async def transcribe(self, audio: AudioPayload) -> TranscriptionResult:
        if not self.configured:
            raise STTError("ELEVENLABS_API_KEY not set", permanent=True)
        files = {"file": (audio.filename, audio.data, audio.content_type)}
        data: dict[str, str] = {"model_id": self.settings.elevenlabs_stt_model}
        iso = to_iso_language(audio.language)
        if iso:
            data["language_code"] = iso
        try:
            response = await self._get_client().post(
                "/v1/speech-to-text", files=files, data=data
            )
        except httpx.TimeoutException as exc:
            raise STTError(f"elevenlabs timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise STTError(f"elevenlabs transport error: {exc}") from exc

        if response.status_code >= 400:
            permanent = 400 <= response.status_code < 500 and response.status_code != 429
            raise STTError(
                f"elevenlabs HTTP {response.status_code}: {response.text[:200]}",
                permanent=permanent,
            )
        payload = response.json()
        text = (payload.get("text") or "").strip()
        if not text:
            raise STTError("elevenlabs returned an empty transcript", permanent=True)
        return TranscriptionResult(
            text=text,
            provider=self.name,
            language=payload.get("language_code"),
            raw={
                "language_probability": payload.get("language_probability"),
                "audio_duration_secs": payload.get("audio_duration_secs"),
                "model": data["model_id"],
            },
        )
