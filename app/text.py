"""Script-aware text utilities shared by chunking, retrieval and guardrails.

MSMARCO-XI spans Latin and Indic scripts, so nothing here may assume ASCII:
sentence boundaries include the Devanagari danda, and tokenisation is driven by
the Unicode word-character class rather than `[a-z]`.
"""

from __future__ import annotations

import re
import unicodedata

# Sentence terminators across the scripts present in MSMARCO-XI.
_TERMINATORS = ".!?।॥۔؟。！？"
# Split after a terminator + whitespace, but not inside decimals ("3.5"),
# initials ("U.S.") or common abbreviations.
_SENTENCE_SPLIT = re.compile(
    rf"(?<=[{re.escape(_TERMINATORS)}])['\"’”\)\]]*\s+"
    r"(?=[^\s\)\]\.,;:])"
)
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "no", "vol", "fig", "approx", "inc", "ltd", "co", "u.s", "u.k",
}
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(
    r"(?<![\w.])"
    r"(?:\d{1,3}(?:,\d{2,3})+|\d+)"       # 1,234,567 or 1234
    r"(?:\.\d+)?"                          # optional decimal part
    r"\s*(?:%|percent|million|billion|thousand|crore|lakh|kg|km|cm|mm|lb|lbs|"
    r"oz|ft|in|mi|ml|l|g|mg|hours?|hrs?|mins?|minutes?|seconds?|days?|weeks?|"
    r"months?|years?)?",
    re.IGNORECASE,
)
_DIGITS = re.compile(r"\d")

# Small, high-frequency stoplist. Kept deliberately short: aggressive stopword
# removal hurts BM25 on short MS MARCO queries ("what is the *cost of* ...").
STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that
    the to was were what when where which who why will with does do did their
    they this these those there can could should would about into than then
    them you your i me my we our us not no if but so such also""".split()
)

SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0041, 0x024F, "Latin"),
)


def normalize(text: str) -> str:
    """NFKC-normalise and collapse whitespace without touching case."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text or "")).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokens (used by BM25 and lexical overlap)."""
    return _WORD.findall(unicodedata.normalize("NFKC", text or "").casefold())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed — the units a grounding check cares about."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


# --- English morphology for the lexical index --------------------------------
# BM25 matches surface forms, so without folding "eagle" cannot match "eagles"
# and a passage scores exactly 0.0 on a query it literally answers. Measured on
# this corpus: for "how fast does an eagle travel", the passage "Eagles fly 30 to
# 55 mph" ranked #0 of 18,416 by dense similarity and 871st by BM25 — and because
# RRF fuses ranks rather than scores, a travel-booking passage that scored on both
# lists displaced it out of the context set entirely. The answer was assembled
# from three unrelated passages instead.
#
# The fold is deliberately plurals-only: no verb stemming, no suffix stripping
# beyond the noun-plural cases below. It is applied to the postings and to the
# query through the same function, so the two can never disagree. That symmetry
# is what makes an imperfect fold safe — "movies" → "movy" is odd in isolation
# but matches itself on both sides, whereas a *missing* fold loses documents.
_KEEP_TRAILING_S = ("ss", "us", "is")  # class, virus, analysis — real singulars
_STRIP_ES = ("sses", "ches", "shes", "xes", "zes")  # glasses, watches, boxes


def fold_plural(token: str) -> str:
    """Map a probable English plural onto its singular surface form."""
    if len(token) < 4 or not token.isascii() or not token.isalpha():
        return token
    if token in STOPWORDS or token.endswith(_KEEP_TRAILING_S):
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"  # cities -> city, companies -> company
    if token.endswith(_STRIP_ES):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def lexical_tokens(text: str) -> list[str]:
    """Tokens for BM25 — `tokenize` plus plural folding, index and query alike."""
    return [fold_plural(t) for t in tokenize(text)]


def folded_tokens(text: str) -> list[str]:
    """Content tokens with plural folding — for term-overlap between two texts.

    Use this wherever the two sides were written by different people: a user's
    query against a corpus passage. The same "eagle"/"eagles" mismatch that cost
    BM25 the passage also zeroed the *sentence selector's* term-coverage score,
    so the sentence that answered the question ("Eagles fly 30 to 55 mph")
    scored 0.00 coverage and lost to a scrape fragment that happened to spell it
    in the singular.

    Deliberately *not* used for the grounding check, where the answer is copied
    verbatim from the context: both sides are the same text there, so there is no
    asymmetry to correct, and the thresholds were calibrated on surface forms.
    """
    return [fold_plural(t) for t in content_tokens(text)]


def split_sentences(text: str, min_chars: int = 24) -> list[str]:
    """Split into sentences, re-joining fragments left by abbreviations."""
    text = normalize(text)
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and (
            len(merged[-1]) < min_chars
            or merged[-1].rstrip(".").split(" ")[-1].casefold() in _ABBREVIATIONS
        ):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def word_count(text: str) -> int:
    return len(text.split())


def detect_script(text: str) -> str:
    """Majority Unicode block of the alphabetic characters — a cheap language cue."""
    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        cp = ord(char)
        for lo, hi, name in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return max(counts, key=counts.__getitem__) if counts else "Unknown"


def extract_numbers(text: str) -> list[str]:
    """Numeric/quantity mentions, normalised for comparison against context.

    Hallucinated figures are the highest-risk failure mode in RAG, so the output
    guard verifies each of these appears in the retrieved text.
    """
    out: list[str] = []
    for match in _NUMBER.finditer(text or ""):
        raw = match.group(0).strip()
        if not _DIGITS.search(raw):
            continue
        out.append(_WHITESPACE.sub(" ", raw.casefold().replace(",", "")))
    return out


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + " …"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
