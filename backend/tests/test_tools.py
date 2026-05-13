"""
Tool-dispatch tests: web_search against live API + planner/executor behavior.

Run from backend/:
    python -m pytest tests/test_tools.py -v -s

Requires TAVILY_API_KEY and GROQ_API_KEY in backend/.env.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from models.agent_state import AgentState, Evidence, ToolCall
from rag.retrievers.web_search import WebSearch
from rag.nodes.planner import Planner, _normalise
from rag.nodes.action_executor import ActionExecutor
from rag.llm_factory import create_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def web_search():
    return WebSearch(api_key=os.environ["TAVILY_API_KEY"])


@pytest.fixture(scope="module")
def planner():
    llm = create_llm()
    return Planner(llm, corpus_titles=["Attention is All You Need", "HNSW", "LoRA", "RAG", "RAGAS", "vLLM"])


@pytest.fixture(scope="module")
def executor(web_search):
    from ingest.index_builder import IndexBuilder
    from rag.retrievers.vector_retriever import VectorRetriever
    builder = IndexBuilder()
    faiss_store, bm25 = builder.load_indices()
    vector_retriever = VectorRetriever(faiss_store, bm25)
    return ActionExecutor(vector_retriever, web_search)


# ---------------------------------------------------------------------------
# WebSearch (live API)
# ---------------------------------------------------------------------------

def test_web_search_returns_evidence(web_search):
    results = asyncio.run(web_search.retrieve("LangGraph tutorial 2025"))
    assert len(results) > 0
    for e in results:
        assert e.source_type == "web"
        assert e.content
        assert e.url
        assert e.evidence_id
    print(f"\n[web_search] {len(results)} results, first url: {results[0].url}")


def test_web_search_evidence_shape(web_search):
    results = asyncio.run(web_search.retrieve("FAISS vector store documentation"))
    e = results[0]
    assert isinstance(e, Evidence)
    assert e.source_type == "web"
    assert e.metadata.get("source")


# ---------------------------------------------------------------------------
# Planner normalisation — unknown tool names fall back to vector_retriever
# ---------------------------------------------------------------------------

def test_normalise_unknown_tool_falls_back():
    state: AgentState = {"query": "anything"}
    tc = {"tool_name": "nonexistent_tool", "args": {}, "sub_query": "anything",
          "rationale": "", "expected_evidence_type": ""}
    result = _normalise(tc, state)
    assert result.tool_name == "vector_retriever"


def test_normalise_valid_tools_pass_through():
    state: AgentState = {"query": "anything"}
    for tool in ("vector_retriever", "web_search"):
        tc = {"tool_name": tool, "args": {}, "sub_query": "anything",
              "rationale": "", "expected_evidence_type": ""}
        result = _normalise(tc, state)
        assert result.tool_name == tool


# ---------------------------------------------------------------------------
# ActionExecutor — deduplication
# ---------------------------------------------------------------------------

def test_executor_deduplicates(executor):
    """Running the same plan twice must not duplicate evidence."""
    plan = [
        ToolCall(tool_name="vector_retriever", args={},
                 sub_query="what is HNSW", rationale="test", expected_evidence_type="vector_chunk"),
    ]
    state: AgentState = {"query": "what is HNSW", "action_plan": plan,
                          "tool_results": {}, "iteration_count": 1}

    state1 = asyncio.run(executor.run(state))
    count_after_first = sum(len(v) for v in state1["tool_results"].values())

    state2 = asyncio.run(executor.run({**state1, "action_plan": plan}))
    count_after_second = sum(len(v) for v in state2["tool_results"].values())

    assert count_after_second == count_after_first, (
        f"Deduplication failed: {count_after_first} → {count_after_second}"
    )
    print(f"\n[dedup] {count_after_first} unique evidence items, stable on re-run")
