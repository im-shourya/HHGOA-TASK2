"""BM25+ lexical retrieval over an inverted index.

Dense static embeddings are fast but lossy on rare tokens — exactly the tokens
MS MARCO questions turn on (model numbers, drug names, "1994"). BM25 covers that
blind spot, and because scoring only touches the postings of query terms it costs
well under a millisecond at this corpus size.

Implemented directly on numpy arrays rather than pulled from a library so the
postings layout (flat arrays + offsets) can be memory-mapped from disk and the
term weighting stays inspectable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


class BM25Index:
    """BM25+ (Lv & Zhai) — the `delta` floor stops long documents scoring zero."""

    def __init__(self, k1: float = 1.4, b: float = 0.72, delta: float = 0.5) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.vocab: dict[str, int] = {}
        self.postings_docs: np.ndarray = np.zeros(0, dtype=np.int32)
        self.postings_tf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.float32)
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    # ------------------------------------------------------------------ build
    def build(self, tokenized_docs: Sequence[Sequence[str]]) -> "BM25Index":
        self.n_docs = len(tokenized_docs)
        self.doc_len = np.array([len(d) for d in tokenized_docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        term_docs: dict[str, list[int]] = {}
        term_tfs: dict[str, list[float]] = {}
        for doc_id, tokens in enumerate(tokenized_docs):
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, tf in counts.items():
                term_docs.setdefault(token, []).append(doc_id)
                term_tfs.setdefault(token, []).append(float(tf))

        self.vocab = {term: i for i, term in enumerate(sorted(term_docs))}
        docs_flat: list[np.ndarray] = []
        tf_flat: list[np.ndarray] = []
        offsets = [0]
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for term, term_id in self.vocab.items():
            docs = np.asarray(term_docs[term], dtype=np.int32)
            docs_flat.append(docs)
            tf_flat.append(np.asarray(term_tfs[term], dtype=np.float32))
            offsets.append(offsets[-1] + docs.size)
            df = docs.size
            idf[term_id] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

        self.postings_docs = (
            np.concatenate(docs_flat) if docs_flat else np.zeros(0, dtype=np.int32)
        )
        self.postings_tf = (
            np.concatenate(tf_flat) if tf_flat else np.zeros(0, dtype=np.float32)
        )
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.idf = idf
        return self

    # ----------------------------------------------------------------- search
    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        """Full score vector. Only postings of matched query terms are touched."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avgdl <= 0:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * (self.doc_len / self.avgdl))
        seen: set[int] = set()
        for token in query_tokens:
            term_id = self.vocab.get(token)
            if term_id is None or term_id in seen:
                continue
            seen.add(term_id)
            lo, hi = int(self.offsets[term_id]), int(self.offsets[term_id + 1])
            docs = self.postings_docs[lo:hi]
            tf = self.postings_tf[lo:hi]
            saturated = (tf * (self.k1 + 1.0)) / (tf + norm[docs]) + self.delta
            scores[docs] += self.idf[term_id] * saturated
        return scores

    def search(self, query_tokens: Sequence[str], top_k: int = 40) -> tuple[np.ndarray, np.ndarray]:
        """Top-k `(indices, scores)`, highest first, zero-score docs removed."""
        scores = self.scores(query_tokens)
        if not scores.size:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        top_k = min(top_k, scores.size)
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
        candidates = candidates[scores[candidates] > 0]
        return candidates, scores[candidates]

    # ---------------------------------------------------------------- persist
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "bm25.npz",
            postings_docs=self.postings_docs,
            postings_tf=self.postings_tf,
            offsets=self.offsets,
            idf=self.idf,
            doc_len=self.doc_len,
        )
        (directory / "bm25_meta.json").write_text(
            json.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "delta": self.delta,
                    "avgdl": self.avgdl,
                    "n_docs": self.n_docs,
                    "vocab": self.vocab,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        meta = json.loads((directory / "bm25_meta.json").read_text(encoding="utf-8"))
        index = cls(k1=meta["k1"], b=meta["b"], delta=meta["delta"])
        index.avgdl = float(meta["avgdl"])
        index.n_docs = int(meta["n_docs"])
        index.vocab = {term: int(i) for term, i in meta["vocab"].items()}
        with np.load(directory / "bm25.npz") as payload:
            index.postings_docs = payload["postings_docs"]
            index.postings_tf = payload["postings_tf"]
            index.offsets = payload["offsets"]
            index.idf = payload["idf"]
            index.doc_len = payload["doc_len"]
        return index

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
