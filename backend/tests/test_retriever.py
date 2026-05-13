"""
Phase 2 unit test: dense-only vs. hybrid retriever precision.

Run from backend/:
    python -m pytest tests/test_retriever.py -v

Each query has a known "expected source" (the paper that should appear in top results).
Precision@K = fraction of top-K results whose `source` metadata matches the expected paper.
"""
from __future__ import annotations
import sys
import asyncio
import pickle
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.index_builder import IndexBuilder
from rag.retrievers.vector_retriever import VectorRetriever

# ---------------------------------------------------------------------------
# 10 sample queries: (query, expected_source_substring)
# ---------------------------------------------------------------------------
QUERIES = [
    ("How does the attention mechanism work in transformers?",      "Attention"),
    ("What is multi-head attention and why is it used?",           "Attention"),
    ("Explain the HNSW graph structure for approximate search",    "HNSW"),
    ("How does HNSW select neighbors during index construction?",  "HNSW"),
    ("What is LoRA and how does it reduce trainable parameters?",  "LoRA"),
    ("How does low-rank decomposition apply to fine-tuning LLMs?", "LoRA"),
    ("How does the RAG model combine retrieval with generation?",  "RAG"),
    ("What metrics does RAGAS use to evaluate RAG pipelines?",     "RAGAS"),
    ("How does vLLM use PagedAttention to manage KV cache?",       "vLLM"),
    ("What is the role of ef_construction in HNSW indexing?",      "HNSW"),
]

_INDEX_DIR = Path(__file__).parent.parent / "indices"
_FAISS_PATH = _INDEX_DIR / "faiss_index"
_BM25_PATH  = _INDEX_DIR / "bm25_index.pkl"


@pytest.fixture(scope="module")
def retriever():
    builder = IndexBuilder()
    faiss_store, bm25 = builder.load_indices()
    return VectorRetriever(faiss_store, bm25)


@pytest.fixture(scope="module")
def dense_retriever():
    """FAISS-only retriever (no BM25, no reranker) for comparison baseline."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_store = FAISS.load_local(
        str(_FAISS_PATH), embeddings, allow_dangerous_deserialization=True
    )
    return faiss_store.as_retriever(search_kwargs={"k": 5})


def _precision(results: list, expected: str, k: int = 5) -> float:
    hits = sum(1 for r in results[:k] if expected.lower() in r.lower())
    return hits / k


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected", QUERIES)
def test_hybrid_retriever_top5(retriever, query, expected):
    """Hybrid pipeline must surface the expected paper in at least 1 of top-5."""
    results = asyncio.run(retriever.retrieve(query))
    sources = [e.metadata.get("source", "") for e in results]
    assert any(expected.lower() in s.lower() for s in sources), (
        f"Expected '{expected}' in top-5 for query: {query!r}\n"
        f"Got sources: {sources}"
    )


@pytest.mark.parametrize("query,expected", QUERIES)
def test_dense_only_top5(dense_retriever, query, expected):
    """Dense-only baseline: expected paper must appear in top-5."""
    docs = dense_retriever.invoke(query)
    sources = [d.metadata.get("source", "") for d in docs[:5]]
    assert any(expected.lower() in s.lower() for s in sources), (
        f"Dense-only: expected '{expected}' in top-5 for query: {query!r}\n"
        f"Got sources: {sources}"
    )


def test_hybrid_vs_dense_precision(retriever, dense_retriever):
    """
    Aggregate comparison across all 10 queries.
    Hybrid Precision@5 must be >= dense-only Precision@5.
    Prints a per-query breakdown for inspection.
    """
    hybrid_scores = []
    dense_scores  = []

    print("\n{:<60} {:>8} {:>8}".format("Query", "Hybrid", "Dense"))
    print("-" * 80)

    for query, expected in QUERIES:
        hybrid_results = asyncio.run(retriever.retrieve(query))
        hybrid_sources = [e.metadata.get("source", "") for e in hybrid_results]
        h_prec = _precision(hybrid_sources, expected)

        dense_docs    = dense_retriever.invoke(query)
        dense_sources = [d.metadata.get("source", "") for d in dense_docs[:5]]
        d_prec = _precision(dense_sources, expected)

        hybrid_scores.append(h_prec)
        dense_scores.append(d_prec)
        print(f"{query[:58]:<60} {h_prec:>8.2f} {d_prec:>8.2f}")

    avg_hybrid = sum(hybrid_scores) / len(hybrid_scores)
    avg_dense  = sum(dense_scores)  / len(dense_scores)
    print("-" * 80)
    print(f"{'AVERAGE':<60} {avg_hybrid:>8.2f} {avg_dense:>8.2f}")

    assert avg_hybrid >= avg_dense, (
        f"Hybrid Precision@5 ({avg_hybrid:.2f}) is lower than "
        f"dense-only ({avg_dense:.2f}) — check RRF weights or reranker."
    )
