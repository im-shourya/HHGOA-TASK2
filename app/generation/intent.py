"""Lightweight query intent classification (regex + token cues, ~10 µs).

Intent drives three things: which sentence wins during extraction, how the answer
is shaped, and how strict the output guard is. A NUMERIC question that produces an
answer with no number in it is a strong signal the pipeline missed, so intent also
feeds verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.text import detect_script, folded_tokens, normalize


class Intent(str, Enum):
    NUMERIC = "numeric"
    ENTITY = "entity"
    LOCATION = "location"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    PROCEDURAL = "procedural"
    DESCRIPTION = "description"


_PATTERNS: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (Intent.NUMERIC, re.compile(
        r"\b(how (much|many|long|old|far|tall|big|often)|what (is the )?(cost|price|"
        r"salary|average|percentage|number|temperature|weight|size)|कितन[ाीे])\b", re.I)),
    (Intent.TEMPORAL, re.compile(
        r"\b(when|what year|what time|which year|how long does it take|कब)\b", re.I)),
    (Intent.LOCATION, re.compile(r"\b(where|which (country|state|city|place)|कहाँ|कहां)\b", re.I)),
    (Intent.ENTITY, re.compile(r"\b(who|whose|which (company|person|team|brand)|कौन)\b", re.I)),
    (Intent.BOOLEAN, re.compile(
        r"^\s*(is|are|was|were|does|do|did|can|could|should|will|would|has|have|had)\b", re.I)),
    (Intent.PROCEDURAL, re.compile(r"\b(how (do|to|does|can)|steps to|कैसे)\b", re.I)),
)


@dataclass(slots=True)
class QueryIntent:
    intent: Intent
    focus: list[str]
    script: str
    normalized: str
    is_question: bool

    @property
    def wants_number(self) -> bool:
        return self.intent in (Intent.NUMERIC, Intent.TEMPORAL)


def classify(query: str) -> QueryIntent:
    text = normalize(query)
    intent = Intent.DESCRIPTION
    for candidate, pattern in _PATTERNS:
        if pattern.search(text):
            intent = candidate
            break
    return QueryIntent(
        intent=intent,
        # Folded, so a query's "eagle" matches a passage's "eagles" — `focus` is
        # only ever compared against corpus text, never shown to the user.
        focus=folded_tokens(text),
        script=detect_script(text),
        normalized=text,
        is_question=text.endswith("?") or intent is not Intent.DESCRIPTION,
    )
