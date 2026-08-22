"""Builds a throwaway, in-memory FAISS index from the sampled MSMARCO-XI
examples' candidate passages. This is testing infrastructure, not a copy
of the target's production index -- it deliberately does NOT borrow the
target's chunking function, HNSW parameters, or embedding-dimension
constant. Those used to be imported from the target project (app.chunking,
app.config's CHUNK_SIZE/CHUNK_OVERLAP/HNSW_*/EMBEDDING_DIM), which meant a
target that didn't happen to use FAISS+HNSW with those exact config names
couldn't be evaluated by this suite at all -- an unnecessary requirement,
since nothing about *this* index needs to match the target's real one.
The only genuinely required dependency on the target is app.embedder's
embed()/embed_one()/get_model() (see eval/target.py's interface contract);
everything else below is this suite's own, overridable via the
EVAL_CHUNK_SIZE / EVAL_CHUNK_OVERLAP / EVAL_HNSW_* environment variables.

Mixed-language on purpose: every candidate passage from every sampled
example goes in, English and Hindi both, tagged by language. This is what
makes the retrieval check's cross-lingual metric possible (see
eval/checks/retrieval.py) -- useful for any target whose embedding model
was trained for cross-lingual retrieval (as this suite's original target
project's was), and harmless (same-language recall is unaffected) for one
that wasn't.
"""
import os
from dataclasses import dataclass

import numpy as np

from eval.dataset import EvalExample

# This suite's own defaults, not the target's -- reasonable values for a
# small throwaway eval index, not tuned production settings. Override via
# env var if you have a specific reason to (e.g. matching a target's chunk
# size to see whether that changes retrieval quality on this dataset).
CHUNK_SIZE = int(os.environ.get("EVAL_CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(os.environ.get("EVAL_CHUNK_OVERLAP", 60))
HNSW_M = int(os.environ.get("EVAL_HNSW_M", 32))
HNSW_EF_CONSTRUCTION = int(os.environ.get("EVAL_HNSW_EF_CONSTRUCTION", 40))
HNSW_EF_SEARCH = int(os.environ.get("EVAL_HNSW_EF_SEARCH", 32))


@dataclass
class ChunkRecord:
    query_id: int
    lang: str          # "en" | "hi"
    is_selected: bool
    text: str


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """A plain fixed-size sliding-window chunker with overlap -- not the
    target's own chunker (that would be an unnecessary required interface
    item; see this module's docstring). Good enough for building a
    representative eval index out of MSMARCO-XI passages, which are
    already short (a few sentences), so most passages end up as one chunk
    regardless of the exact splitting strategy."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def build_index(examples: list[EvalExample]):
    import faiss

    from eval import target

    embedder = target.get_embedder()
    embed, embed_one, get_model = embedder.embed, embedder.embed_one, embedder.get_model

    texts: list[str] = []
    records: list[ChunkRecord] = []

    for ex in examples:
        selected_idx = ex.gt_passage_index  # None for unanswerable examples
        for lang, candidates in (("en", ex.candidates_en), ("hi", ex.candidates_hi)):
            for i, passage in enumerate(candidates):
                if not passage:
                    continue
                for chunk in _chunk_text(passage, CHUNK_SIZE, CHUNK_OVERLAP):
                    texts.append(chunk)
                    records.append(
                        ChunkRecord(query_id=ex.query_id, lang=lang, is_selected=(i == selected_idx), text=chunk)
                    )

    get_model()  # ensure loaded before timing/embedding

    # Infer the embedding dimension empirically instead of requiring an
    # EMBEDDING_DIM constant from the target -- one real call to the
    # target's own embed_one() is authoritative and needs no assumption
    # about how (or whether) the target names that config value.
    embedding_dim = embed_one("dimension probe").shape[-1]

    batch_size = 64
    vector_batches = [embed(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    vectors = np.vstack(vector_batches) if vector_batches else np.zeros((0, embedding_dim), dtype=np.float32)

    index = faiss.IndexHNSWFlat(embedding_dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    if len(vectors):
        index.add(vectors)

    return index, records
