"""Input guardrail: decide whether this request should reach retrieval at all.

Ordered cheapest-first — shape, then safety, then injection, then PII — so a
malformed request costs microseconds and never touches the index. Every check
records a `GuardFinding` whether it passed or not, because "which guards ran" is
as important as "which guard fired" when you have to explain a refusal.

Off-topic detection deliberately lives elsewhere. A keyword list cannot know what
this corpus covers; the retrieval scores can, so that decision is made in
`retrieval_guard` once we have evidence.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from app.guardrails.policies import (
    INJECTION_RULES,
    MALFORMED_MESSAGE,
    PII_PATTERNS,
    UNSAFE_RULES,
    is_allowlisted,
)
from app.schemas import GuardFinding, Verdict
from app.text import content_tokens, normalize


@dataclass
class InputDecision:
    allowed: bool
    query: str
    verdict: Verdict = Verdict.ANSWERED
    message: str = ""
    findings: list[GuardFinding] = field(default_factory=list)
    redacted: str = ""
    pii_types: list[str] = field(default_factory=list)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


_VOWELS = frozenset("aeiouy")


def looks_like_mash(token: str) -> bool:
    """Keyboard mash detector for Latin-script tokens.

    Real long words keep a workable vowel rhythm ("immunohistochemistry" is 40%
    vowels); mashed keys do not ("asdkjhaskdjhaksjdh" is 17% and has five-consonant
    runs). Non-Latin scripts are left alone — the vowel heuristic does not transfer
    to abugidas, where matra vowels are combining marks.
    """
    token = token.casefold()
    if len(token) < 15 or not token.isascii() or not token.isalpha():
        return False
    vowel_ratio = sum(1 for c in token if c in _VOWELS) / len(token)
    longest_run, run = 0, 0
    for char in token:
        run = 0 if char in _VOWELS else run + 1
        longest_run = max(longest_run, run)
    return vowel_ratio < 0.28 or longest_run >= 5


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Mask PII before anything gets logged. Returns (masked, types_found)."""
    found: list[str] = []
    masked = text
    for name, pattern in PII_PATTERNS:
        if pattern.search(masked):
            found.append(name)
            masked = pattern.sub(f"[{name}_redacted]", masked)
    return masked, found


class InputGuard:
    def __init__(self, min_chars: int = 2, max_chars: int = 512) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    def check(self, raw_query: str) -> InputDecision:
        findings: list[GuardFinding] = []
        query = normalize(raw_query)

        # 1. Shape --------------------------------------------------------
        if len(query) < self.min_chars or not content_tokens(query):
            findings.append(
                GuardFinding(
                    guard="shape",
                    passed=False,
                    severity="block",
                    reason="empty or contentless query",
                    detail={"chars": len(query)},
                )
            )
            return InputDecision(
                allowed=False,
                query=query,
                verdict=Verdict.DECLINED_MALFORMED,
                message=MALFORMED_MESSAGE,
                findings=findings,
                redacted=query[:120],
            )
        truncated = len(query) > self.max_chars
        if truncated:
            query = query[: self.max_chars].rsplit(" ", 1)[0]
        findings.append(
            GuardFinding(
                guard="shape",
                passed=True,
                severity="warn" if truncated else "info",
                reason="truncated to max length" if truncated else "",
                detail={"chars": len(query), "truncated": truncated},
            )
        )

        # 2. Gibberish ----------------------------------------------------
        # Either signal is enough: a token that looks like keyboard mash, or
        # unusually low character entropy across a long query. Both are
        # length-gated so short real questions are never caught.
        entropy = _entropy(query.casefold())
        tokens = content_tokens(query)
        longest = max((len(t) for t in tokens), default=0)
        mashed = any(looks_like_mash(t) for t in tokens)
        gibberish = len(query) >= 12 and (
            mashed or (entropy < 2.4 and longest > 14)
        )
        findings.append(
            GuardFinding(
                guard="gibberish",
                passed=not gibberish,
                severity="block" if gibberish else "info",
                reason="low-entropy keyboard mash" if gibberish else "",
                detail={
                    "entropy": round(entropy, 2),
                    "longest_token": longest,
                    "mashed_token": mashed,
                },
            )
        )
        if gibberish:
            return InputDecision(
                allowed=False,
                query=query,
                verdict=Verdict.DECLINED_MALFORMED,
                message=MALFORMED_MESSAGE,
                findings=findings,
                redacted=query[:120],
            )

        # 3. Safety -------------------------------------------------------
        allowlisted = is_allowlisted(query)
        for rule in UNSAFE_RULES:
            if rule.pattern.search(query) and not allowlisted:
                findings.append(
                    GuardFinding(
                        guard="safety",
                        passed=False,
                        severity="block",
                        reason=f"matched {rule.category.value}",
                        detail={"category": rule.category.value},
                    )
                )
                return InputDecision(
                    allowed=False,
                    query=query,
                    verdict=Verdict.DECLINED_UNSAFE,
                    message=rule.message,
                    findings=findings,
                    redacted=redact_pii(query)[0][:120],
                )
        findings.append(
            GuardFinding(
                guard="safety",
                passed=True,
                detail={"rules": len(UNSAFE_RULES), "allowlisted": allowlisted},
            )
        )

        # 4. Prompt injection ---------------------------------------------
        for rule in INJECTION_RULES:
            if rule.pattern.search(query):
                findings.append(
                    GuardFinding(
                        guard="injection",
                        passed=False,
                        severity="block",
                        reason="instruction-override attempt",
                    )
                )
                return InputDecision(
                    allowed=False,
                    query=query,
                    verdict=Verdict.DECLINED_INJECTION,
                    message=rule.message,
                    findings=findings,
                    redacted=query[:120],
                )
        findings.append(GuardFinding(guard="injection", passed=True))

        # 5. PII (redact, never block: it may be part of a legitimate question)
        redacted, pii_types = redact_pii(query)
        findings.append(
            GuardFinding(
                guard="pii",
                passed=not pii_types,
                severity="warn" if pii_types else "info",
                reason=f"redacted {', '.join(pii_types)} before logging" if pii_types else "",
                detail={"types": pii_types},
            )
        )
        return InputDecision(
            allowed=True,
            query=query,
            findings=findings,
            redacted=redacted,
            pii_types=pii_types,
        )
