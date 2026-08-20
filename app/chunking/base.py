"""Core chunking types.

A `Document` is one MSMARCO-XI passage. Every chunker turns a document into
`Chunk`s that keep three things distinct:

* `text`      — what a human reads and what the answer quotes.
* `embed_text`— what actually gets embedded/indexed (may carry a context header).
* `parent_id` — the wider window handed to the generator (small-to-big retrieval).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    lang: str = "eng_Latn"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    strategy: str
    start: int = 0
    end: int = 0
    parent_id: str = ""
    embed_text: str = ""
    lang: str = "eng_Latn"
    strategies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parent_id:
            self.parent_id = self.doc_id
        if not self.embed_text:
            self.embed_text = self.text
        if not self.strategies:
            self.strategies = [self.strategy]

    @property
    def token_count(self) -> int:
        return len(self.text.split())

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "embed_text": self.embed_text,
            "strategy": self.strategy,
            "strategies": self.strategies,
            "lang": self.lang,
            "start": self.start,
            "end": self.end,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=record["chunk_id"],
            doc_id=record["doc_id"],
            parent_id=record.get("parent_id", ""),
            text=record["text"],
            embed_text=record.get("embed_text", ""),
            strategy=record.get("strategy", "unknown"),
            strategies=list(record.get("strategies") or []),
            lang=record.get("lang", "eng_Latn"),
            start=int(record.get("start", 0)),
            end=int(record.get("end", 0)),
            metadata=dict(record.get("metadata") or {}),
        )


@runtime_checkable
class Chunker(Protocol):
    """Anything that turns one document into chunks."""

    name: str

    def split(self, doc: Document) -> list[Chunk]: ...


def make_chunk_id(strategy: str, doc_id: str, ordinal: int, text: str) -> str:
    """Stable, content-addressed id: same input always yields the same id."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=5).hexdigest()
    return f"{strategy}:{doc_id}:{ordinal}:{digest}"
