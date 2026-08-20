"""Speech-to-text: Sarvam and ElevenLabs behind one failover-capable registry."""

from app.stt.base import AudioPayload, STTError, STTProvider
from app.stt.elevenlabs import ElevenLabsSTT
from app.stt.registry import NoProviderConfigured, STTRegistry
from app.stt.sarvam import SarvamSTT

__all__ = [
    "AudioPayload",
    "ElevenLabsSTT",
    "NoProviderConfigured",
    "STTError",
    "STTProvider",
    "STTRegistry",
    "SarvamSTT",
]
