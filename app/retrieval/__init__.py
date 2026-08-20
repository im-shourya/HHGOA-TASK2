"""Retrieval: static embeddings, BM25+, hybrid fusion."""

from app.retrieval.bm25 import BM25Index
from app.retrieval.embedder import Embedder, get_embedder, l2_normalize
from app.retrieval.hybrid import HybridRetriever, RetrievalResult
from app.retrieval.index_store import RagIndex
from app.retrieval.vector_store import build_vector_store

__all__ = [
    "BM25Index",
    "Embedder",
    "HybridRetriever",
    "RagIndex",
    "RetrievalResult",
    "build_vector_store",
    "get_embedder",
    "l2_normalize",
]
