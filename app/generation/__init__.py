"""Answer generation: extractive by default, Claude when the budget allows."""

from app.generation.extractive import AnswerDraft, ExtractiveComposer
from app.generation.intent import Intent, QueryIntent, classify
from app.generation.llm import GroqGenerator, LLMUnavailable

__all__ = [
    "AnswerDraft",
    "GroqGenerator",
    "ExtractiveComposer",
    "Intent",
    "LLMUnavailable",
    "QueryIntent",
    "classify",
]
