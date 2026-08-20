#!/usr/bin/env python
"""Build the searchable index from an ingested corpus.

    passages.jsonl -> 5 chunking strategies -> reconcile -> embed -> BM25 -> disk

Everything under `data/index/` is derived: delete it and re-run this script to get
a byte-identical rebuild (chunk ids are content-addressed, so they are stable).

Example:
    python scripts/build_index.py                 # default corpus + model
    python scripts/build_index.py --limit 500     # quick smoke build
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.chunking import ChunkingPipeline, Document  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.retrieval.embedder import get_embedder  # noqa: E402
from app.retrieval.index_store import RagIndex  # noqa: E402


def load_documents(corpus_dir: Path, limit: int | None = None) -> list[Document]:
    path = corpus_dir / "passages.jsonl"
    if not path.exists():
        raise SystemExit(f"no corpus at {path} — run scripts/ingest.py first")
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            documents.append(
                Document(
                    doc_id=record["passage_id"],
                    text=record["text"],
                    lang=record.get("lang", "eng_Latn"),
                    metadata={
                        "query_id": record.get("query_id"),
                        "query_type": record.get("query_type"),
                        "is_selected": record.get("is_selected", False),
                    },
                )
            )
            if limit and len(documents) >= limit:
                break
    return documents


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=settings.corpus_dir)
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--limit", type=int, default=None, help="cap passages (smoke builds)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--no-semantic", action="store_true", help="skip embedding-breakpoint chunking"
    )
    args = parser.parse_args()

    print(f"[index] loading corpus from {args.corpus}")
    documents = load_documents(args.corpus, args.limit)
    print(f"[index] {len(documents)} passages")

    embedder = get_embedder(args.model)
    print(f"[index] embedder: {embedder.name} (dim={embedder.dim})")

    started = time.perf_counter()
    pipeline = ChunkingPipeline(encode=None if args.no_semantic else embedder.encode)
    chunks = pipeline.run(documents)
    chunk_seconds = time.perf_counter() - started
    print(f"[index] chunking: {len(chunks)} chunks in {chunk_seconds:.1f}s")
    print(json.dumps(pipeline.stats.as_dict(), indent=2, ensure_ascii=False))

    started = time.perf_counter()
    vectors = np.zeros((len(chunks), embedder.dim), dtype=np.float32)
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        vectors[start : start + len(batch)] = embedder.encode([c.embed_text for c in batch])
        print(f"\r[index] embedding {min(start + len(batch), len(chunks))}/{len(chunks)}", end="")
    embed_seconds = time.perf_counter() - started
    print(f"\r[index] embedding: {len(chunks)} chunks in {embed_seconds:.1f}s")

    started = time.perf_counter()
    index = RagIndex.build(chunks, vectors)
    bm25_seconds = time.perf_counter() - started
    print(f"[index] bm25: {index.bm25.vocab_size} terms in {bm25_seconds:.1f}s")

    corpus_manifest_path = args.corpus / "manifest.json"
    index.save(
        args.out,
        extra_manifest={
            "embedding_model": embedder.name,
            "chunking": pipeline.stats.as_dict(),
            "corpus": json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
            if corpus_manifest_path.exists()
            else {},
            "build_seconds": {
                "chunking": round(chunk_seconds, 2),
                "embedding": round(embed_seconds, 2),
                "bm25": round(bm25_seconds, 2),
            },
        },
    )
    print(f"[index] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
