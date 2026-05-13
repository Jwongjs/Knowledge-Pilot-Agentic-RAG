"""
Phase 5 evaluation tests: evaluator logic, frozen-cache, tool-selection cases.

Run from backend/:
    python -m pytest tests/test_eval.py -v -s
"""
from __future__ import annotations
import asyncio
import sys
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.agent_state import Evidence
from evaluation.evaluators import (
    faithfulness_evaluator,
    citation_accuracy_evaluator,
    tool_selection_evaluator,
    re_retrieval_counter,
)
from evaluation.run_eval import _CachedRetriever, _load_cache, _save_cache
import tempfile, pickle
from pathlib import Path as _Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRun:
    def __init__(self, outputs: dict):
        self.outputs = outputs


class _FakeExample:
    def __init__(self, inputs: dict):
        self.inputs = inputs


def _citation(source_type: str, snippet: str, citation_id: str = "1") -> dict:
    return {
        "citation_id": citation_id,
        "source_type": source_type,
        "snippet": snippet,
        "source_document": "test.pdf",
    }


# ---------------------------------------------------------------------------
# faithfulness_evaluator (LLM-as-judge) — only smoke-test shape, not score
# ---------------------------------------------------------------------------

def test_faithfulness_no_citations():
    run = _FakeRun({"answer": "some answer", "citations": []})
    result = faithfulness_evaluator(run, _FakeExample({}))
    assert result["key"] == "faithfulness"
    assert result["score"] == 0.0


def test_faithfulness_no_answer():
    run = _FakeRun({"answer": "", "citations": [_citation("vector_chunk", "stuff")]})
    result = faithfulness_evaluator(run, _FakeExample({}))
    assert result["score"] == 0.0


@pytest.mark.integration
def test_faithfulness_llm_judge():
    """LLM judge returns a float score between 0 and 1."""
    run = _FakeRun({
        "answer": "HNSW builds a hierarchical graph for approximate nearest neighbour search [1].",
        "citations": [_citation("vector_chunk",
                                "HNSW uses hierarchical navigable small world graphs for ANN search.")],
    })
    result = faithfulness_evaluator(run, _FakeExample({}))
    assert result["key"] == "faithfulness"
    assert 0.0 <= result["score"] <= 1.0
    print(f"\n[faithfulness] score={result['score']}, reason={result.get('comment')}")


# ---------------------------------------------------------------------------
# citation_accuracy_evaluator
# ---------------------------------------------------------------------------

def test_citation_accuracy_no_expected():
    run = _FakeRun({"citations": []})
    result = citation_accuracy_evaluator(run, _FakeExample({"expected_sources": []}))
    assert result["score"] == 1.0


def test_citation_accuracy_hit():
    run = _FakeRun({"citations": [_citation("web", "some web content")]})
    result = citation_accuracy_evaluator(run, _FakeExample({"expected_sources": ["web"]}))
    assert result["score"] == 1.0


def test_citation_accuracy_miss():
    run = _FakeRun({"citations": [_citation("vector_chunk", "some chunk")]})
    result = citation_accuracy_evaluator(run, _FakeExample({"expected_sources": ["web"]}))
    assert result["score"] == 0.0


def test_citation_accuracy_partial():
    run = _FakeRun({"citations": [
        _citation("vector_chunk", "chunk", "1"),
        _citation("arxiv", "arxiv content", "2"),
    ]})
    result = citation_accuracy_evaluator(run, _FakeExample({"expected_sources": ["vector_chunk", "web"]}))
    assert result["score"] == 0.5


# ---------------------------------------------------------------------------
# tool_selection_evaluator
# ---------------------------------------------------------------------------

def test_tool_selection_no_expected():
    run = _FakeRun({"citations": []})
    result = tool_selection_evaluator(run, _FakeExample({"expected_tool_calls": []}))
    assert result["score"] == 1.0


def test_tool_selection_web_hit():
    """T18-style: web_search expected → web citation present → score=1.0"""
    run = _FakeRun({"citations": [_citation("web", "openai announcement")]})
    result = tool_selection_evaluator(run, _FakeExample({"expected_tool_calls": ["web_search"]}))
    assert result["score"] == 1.0


def test_tool_selection_multi_tool():
    """Multi-tool query: both vector_retriever and web_search expected."""
    run = _FakeRun({"citations": [
        _citation("vector_chunk", "chunking info", "1"),
        _citation("web", "web content", "2"),
    ]})
    result = tool_selection_evaluator(run, _FakeExample({
        "expected_tool_calls": ["vector_retriever", "web_search"]
    }))
    assert result["score"] == 1.0


def test_tool_selection_miss():
    run = _FakeRun({"citations": [_citation("vector_chunk", "chunk")]})
    result = tool_selection_evaluator(run, _FakeExample({"expected_tool_calls": ["web_search"]}))
    assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# re_retrieval_counter
# ---------------------------------------------------------------------------

def test_re_retrieval_not_fired():
    run = _FakeRun({"iteration_count": 1})
    result = re_retrieval_counter(run, _FakeExample({}))
    assert result["score"] == 0


def test_re_retrieval_fired():
    run = _FakeRun({"iteration_count": 2})
    result = re_retrieval_counter(run, _FakeExample({}))
    assert result["score"] == 1


# ---------------------------------------------------------------------------
# Frozen-cache: _CachedRetriever
# ---------------------------------------------------------------------------

def test_cached_retriever_cache_hit():
    """On a cache hit the inner retriever must not be called."""
    class _NeverCalled:
        async def retrieve(self, query, **kwargs):
            raise AssertionError("inner retriever should not be called on cache hit")

    cached_result = [object()]
    import hashlib
    key = f"web_search:{hashlib.md5(b'test query').hexdigest()}"
    cache = {key: cached_result}
    retriever = _CachedRetriever(_NeverCalled(), "web_search", cache, write=False)
    result = asyncio.run(retriever.retrieve("test query"))
    assert result is cached_result


def test_cached_retriever_cache_miss_writes():
    """On a cache miss the result is fetched from inner and written to cache."""
    sentinel = [object()]

    class _StubInner:
        async def retrieve(self, query, **kwargs):
            return sentinel

    cache: dict = {}
    retriever = _CachedRetriever(_StubInner(), "web_search", cache, write=True)
    result = asyncio.run(retriever.retrieve("new query"))
    assert result is sentinel
    assert len(cache) == 1


def test_cached_retriever_no_write_mode():
    """With write=False, a cache miss returns live result but does NOT populate cache."""
    class _StubInner:
        async def retrieve(self, query, **kwargs):
            return ["live"]

    cache: dict = {}
    retriever = _CachedRetriever(_StubInner(), "web_search", cache, write=False)
    asyncio.run(retriever.retrieve("some query"))
    assert len(cache) == 0


def test_cache_roundtrip(tmp_path):
    """_save_cache / _load_cache round-trips through pickle."""
    import importlib, evaluation.run_eval as m
    original = m._CACHE_PATH
    m._CACHE_PATH = tmp_path / "test_cache.pkl"
    try:
        data = {"key1": ["evidence1"], "key2": ["evidence2"]}
        _save_cache(data)
        loaded = _load_cache()
        assert loaded == data
    finally:
        m._CACHE_PATH = original
