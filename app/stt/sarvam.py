"""Sarvam AI speech-to-text (`POST /speech-to-text`).

Contract per docs.sarvam.ai: `api-subscription-key` header, multipart `file`,
optional `model` (`saaras:v3` | `saaras:v4`) and `language_code` (`unknown` for
auto-detection). Response: `{request_id, transcript, language_code, ...}`.

HTTP status drives the retry decision — 4xx other than 429 is permanent (bad key,
unsupported codec), so the harness stops instead of retrying a request that cannot
start succeeding.
"""

from __future__ import annotations

import logging

import httpx

from app.config import Settings
from app.schemas import TranscriptionResult
from app.stt.base import AudioPayload, STTError

logger = logging.getLogger(__name__)

# Sarvam speaks BCP-47-ish codes; MSMARCO-XI uses FLORES tags.
FLORES_TO_SARVAM: dict[str, str] = {
    "hin_Deva": "hi-IN", "ben_Beng": "bn-IN", "guj_Gujr": "gu-IN", "kan_Knda": "kn-IN",
    "mal_Mlym": "ml-IN", "mar_Deva": "mr-IN", "ory_Orya": "od-IN", "pan_Guru": "pa-IN",
    "tam_Taml": "ta-IN", "tel_Telu": "te-IN", "urd_Arab": "ur-IN", "asm_Beng": "as-IN",
    "npi_Deva": "ne-IN", "san_Deva": "sa-IN", "eng_Latn": "en-IN",
}


def to_sarvam_language(language: str | None) -> str:
    if not language:
        return "unknown"
    if language in FLORES_TO_SARVAM:
        return FLORES_TO_SARVAM[language]
    return language if "-" in language else "unknown"


class SarvamSTT:
    name = "sarvam"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self.settings.sarvam_api_key)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.sarvam_base_url,
                timeout=httpx.Timeout(self.settings.stt_timeout_ms / 1000),
                headers={"api-subscription-key": self.settings.sarvam_api_key or ""},
            )
        return self._client

    async def transcribe(self, audio: AudioPayload) -> TranscriptionResult:
        if not self.configured:
            raise STTError("SARVAM_API_KEY not set", permanent=True)
        files = {"file": (audio.filename, audio.data, audio.content_type)}
        data = {
            "model": self.settings.sarvam_stt_model,
            "language_code": to_sarvam_language(audio.language),
        }
        try:
            response = await self._get_client().post("/speech-to-text", files=files, data=data)
        except httpx.TimeoutException as exc:
            raise STTError(f"sarvam timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise STTError(f"sarvam transport error: {exc}") from exc

        if response.status_code >= 400:
            permanent = 400 <= response.status_code < 500 and response.status_code != 429
            raise STTError(
                f"sarvam HTTP {response.status_code}: {response.text[:200]}",
                permanent=permanent,
            )
        payload = response.json()
        transcript = (payload.get("transcript") or "").strip()
        if not transcript:
            raise STTError("sarvam returned an empty transcript", permanent=True)
        return TranscriptionResult(
            text=transcript,
            provider=self.name,
            language=payload.get("language_code"),
            raw={
                "request_id": payload.get("request_id"),
                "language_probability": payload.get("language_probability"),
                "model": data["model"],
            },
        )
