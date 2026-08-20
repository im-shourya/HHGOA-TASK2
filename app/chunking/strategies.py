"""The chunking library — five complementary strategies over one corpus.

No single splitter wins on every query. Fixed windows give uniform recall but cut
sentences in half; sentence windows respect meaning but drift in length; semantic
splitting finds topic edges but is unstable on short passages. So this project
indexes several strategies side by side, deduplicates the overlap, and lets
hybrid retrieval pick whichever view of the text answers the question.

Strategy               | Unit                     | Best at
-----------------------|--------------------------|--------------------------------
passage                | whole passage            | broad "explain X" questions
fixed_window           | 90 words, 24 overlap     | uniform recall, long passages
sentence_window        | 3 sentences, stride 1    | precise attribution / quoting
recursive_char         | 420 chars, separator     | hard length cap, any script
                       |   cascade                |
semantic               | embedding breakpoints    | topic-shift boundaries
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

from app.chunking.base import Chunk, Document, make_chunk_id
from app.text import normalize, split_sentences

_WORD_SPAN = re.compile(r"\S+")

EncodeFn = Callable[[Sequence[str]], np.ndarray]


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD_SPAN.finditer(text)]


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Sentences with their char offsets in `text`."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        if start < 0:  # normalisation moved things; fall back to the cursor
            start = cursor
        end = start + len(sentence)
        spans.append((start, end, sentence))
        cursor = end
    return spans


class PassageChunker:
    """The whole passage as one chunk — also the parent for finer strategies."""

    name = "passage"

    def __init__(self, max_words: int = 220) -> None:
        self.max_words = max_words

    def split(self, doc: Document) -> list[Chunk]:
        text = normalize(doc.text)
        if not text:
            return []
        spans = _word_spans(text)
        if len(spans) <= self.max_words:
            windows = [(0, len(spans))]
        else:  # very long passage: hard-split so a parent never blows the budget
            windows = [
                (i, min(i + self.max_words, len(spans)))
                for i in range(0, len(spans), self.max_words)
            ]
        chunks: list[Chunk] = []
        for ordinal, (lo, hi) in enumerate(windows):
            start, end = spans[lo][0], spans[hi - 1][1]
            body = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(self.name, doc.doc_id, ordinal, body),
                    doc_id=doc.doc_id,
                    text=body,
                    strategy=self.name,
                    start=start,
                    end=end,
                    lang=doc.lang,
                    metadata={"is_parent": True},
                )
            )
        return chunks


class FixedWindowChunker:
    """Sliding fixed-width word windows with explicit overlap.

    Overlap is the knob that stops an answer-bearing sentence from being sliced
    in half: consecutive windows share `overlap` words, so every span of text is
    seen whole by at least one chunk.
    """

    name = "fixed_window"

    def __init__(self, window: int = 90, overlap: int = 24, min_words: int = 12) -> None:
        if overlap >= window:
            raise ValueError("overlap must be smaller than window")
        self.window = window
        self.overlap = overlap
        self.stride = window - overlap
        self.min_words = min_words

    @property
    def overlap_ratio(self) -> float:
        return self.overlap / self.window

    def split(self, doc: Document) -> list[Chunk]:
        text = normalize(doc.text)
        spans = _word_spans(text)
        if not spans:
            return []
        chunks: list[Chunk] = []
        ordinal = 0
        for lo in range(0, max(len(spans), 1), self.stride):
            hi = min(lo + self.window, len(spans))
            if hi - lo < self.min_words and chunks:
                break  # tail shorter than the floor is already covered by overlap
            start, end = spans[lo][0], spans[hi - 1][1]
            body = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(self.name, doc.doc_id, ordinal, body),
                    doc_id=doc.doc_id,
                    text=body,
                    strategy=self.name,
                    start=start,
                    end=end,
                    lang=doc.lang,
                    metadata={"window": self.window, "overlap": self.overlap},
                )
            )
            ordinal += 1
            if hi >= len(spans):
                break
        return chunks


class SentenceWindowChunker:
    """Windows of whole sentences — retrieval is precise, context stays readable.

    The centre sentence is recorded in metadata so the answer can quote exactly
    the sentence that matched while the generator still sees its neighbours.
    """

    name = "sentence_window"

    def __init__(self, window: int = 3, stride: int = 1, max_words: int = 140) -> None:
        self.window = window
        self.stride = stride
        self.max_words = max_words

    def split(self, doc: Document) -> list[Chunk]:
        text = normalize(doc.text)
        spans = _sentence_spans(text)
        if not spans:
            return []
        chunks: list[Chunk] = []
        ordinal = 0
        for lo in range(0, len(spans), self.stride):
            hi = min(lo + self.window, len(spans))
            start, end = spans[lo][0], spans[hi - 1][1]
            body = text[start:end]
            if len(body.split()) > self.max_words:
                body = " ".join(body.split()[: self.max_words])
                end = start + len(body)
            centre = lo + (hi - lo) // 2
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(self.name, doc.doc_id, ordinal, body),
                    doc_id=doc.doc_id,
                    text=body,
                    strategy=self.name,
                    start=start,
                    end=end,
                    lang=doc.lang,
                    metadata={
                        "sentences": hi - lo,
                        "centre_sentence": spans[centre][2],
                        "sentence_index": centre,
                    },
                )
            )
            ordinal += 1
            if hi >= len(spans):
                break
        return chunks


class RecursiveCharacterChunker:
    """Separator cascade with a hard character cap.

    Walks paragraph -> line -> sentence -> clause -> word, using the coarsest
    separator that keeps every piece under `chunk_size`. Because the cascade ends
    at the empty separator, the cap holds even for scripts this code has never
    seen — useful insurance on a corpus spanning eleven languages.
    """

    name = "recursive_char"
    SEPARATORS: tuple[str, ...] = (
        "\n\n", "\n", "। ", "॥ ", ". ", "? ", "! ", "; ", ", ", " ", "",
    )

    def __init__(self, chunk_size: int = 420, overlap: int = 80, min_chars: int = 48) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chars = min_chars

    def _atoms(self, text: str, separators: tuple[str, ...]) -> list[str]:
        separator, rest = separators[0], separators[1:]
        if separator == "":
            pieces = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        else:
            raw = text.split(separator)
            pieces = [p + separator for p in raw[:-1]] + raw[-1:]
        out: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) <= self.chunk_size or not rest:
                out.append(piece)
            else:
                out.extend(self._atoms(piece, rest))
        return out

    def split(self, doc: Document) -> list[Chunk]:
        text = normalize(doc.text)
        if not text:
            return []
        atoms = self._atoms(text, self.SEPARATORS)
        bodies: list[str] = []
        buffer = ""
        for atom in atoms:
            if buffer and len(buffer) + len(atom) > self.chunk_size:
                bodies.append(buffer.strip())
                buffer = (buffer[-self.overlap :] if self.overlap else "") + atom
            else:
                buffer += atom
        if buffer.strip():
            bodies.append(buffer.strip())

        chunks: list[Chunk] = []
        cursor = 0
        for ordinal, body in enumerate(bodies):
            if len(body) < self.min_chars and len(bodies) > 1:
                continue
            probe = body[: min(len(body), 40)]
            start = text.find(probe, max(cursor - self.overlap, 0))
            start = start if start >= 0 else cursor
            end = min(start + len(body), len(text))
            cursor = end
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(self.name, doc.doc_id, ordinal, body),
                    doc_id=doc.doc_id,
                    text=body,
                    strategy=self.name,
                    start=start,
                    end=end,
                    lang=doc.lang,
                    metadata={"chunk_size": self.chunk_size, "overlap": self.overlap},
                )
            )
        return chunks


class SemanticChunker:
    """Split where the *meaning* changes, not where the word count runs out.

    Sentences are embedded, the cosine distance between neighbours is measured,
    and a boundary is cut wherever that distance exceeds a percentile threshold.
    Static embeddings make this affordable at index time — encoding a passage's
    sentences costs microseconds, so semantic splitting is no longer a luxury
    reserved for offline batch jobs.
    """

    name = "semantic"

    def __init__(
        self,
        encode: EncodeFn,
        percentile: float = 78.0,
        min_words: int = 20,
        max_words: int = 130,
        min_distance: float = 0.12,
    ) -> None:
        self.encode = encode
        self.percentile = percentile
        self.min_words = min_words
        self.max_words = max_words
        self.min_distance = min_distance

    def _breakpoints(self, sentences: list[str]) -> set[int]:
        if len(sentences) < 3:
            return set()
        vectors = np.asarray(self.encode(sentences), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-8)
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = max(float(np.percentile(distances, self.percentile)), self.min_distance)
        return {int(i) + 1 for i, d in enumerate(distances) if d >= threshold}

    def split(self, doc: Document) -> list[Chunk]:
        text = normalize(doc.text)
        spans = _sentence_spans(text)
        if not spans:
            return []
        cuts = self._breakpoints([s[2] for s in spans])

        groups: list[list[int]] = [[]]
        for index in range(len(spans)):
            current = groups[-1]
            words = sum(len(spans[i][2].split()) for i in current)
            starts_new = index in cuts and words >= self.min_words
            too_long = words >= self.max_words
            if current and (starts_new or too_long):
                groups.append([index])
            else:
                current.append(index)
        groups = [g for g in groups if g]

        chunks: list[Chunk] = []
        for ordinal, group in enumerate(groups):
            start, end = spans[group[0]][0], spans[group[-1]][1]
            body = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(self.name, doc.doc_id, ordinal, body),
                    doc_id=doc.doc_id,
                    text=body,
                    strategy=self.name,
                    start=start,
                    end=end,
                    lang=doc.lang,
                    metadata={
                        "sentences": len(group),
                        "breakpoints": sorted(cuts),
                        "percentile": self.percentile,
                    },
                )
            )
        return chunks


