"""
Phase 3 end-to-end smoke test: verifies all 4 LangGraph routing paths.

Run from backend/:
    python -m pytest tests/test_graph_e2e.py -v -s

Requires a valid .env with GROQ_API_KEY (TAVILY_API_KEY not needed — external tools are stubbed).
Uses vector_retriever only — web_search is stubbed so the
test doesn't consume Tavily quota or depend on network state.
"""
from __future__ import annotations
import asyncio
import sys
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.llm_factory import create_llm
from models.agent_state import AgentState, Evidence
from rag.graph_orchestrator import KnowledgePilotGraphOrchestrator
from rag.nodes.query_analyzer import QueryAnalyzer
from rag.nodes.query_decomposer import QueryDecomposer
from rag.nodes.planner import Planner
from rag.nodes.action_executor import ActionExecutor
from rag.nodes.answer_synthesizer import AnswerSynthesizer
from rag.retrievers.vector_retriever import VectorRetriever
from ingest.index_builder import IndexBuilder


# ---------------------------------------------------------------------------
# Stubs for external tools (no network needed)
# ---------------------------------------------------------------------------

class _StubRetriever:
    """Returns one canned Evidence so relevance grader always finds enough."""
    async def retrieve(self, query: str, **kwargs) -> list[Evidence]:
        return [
            Evidence(
                evidence_id=f"stub-{i}",
                source_type="vector_chunk",
                content=f"Relevant content about: {query}. " * 10,
                metadata={"source": "stub.pdf", "doc_type": "research_paper"},
                dense_score=0.9,
                rerank_score=0.85,
            )
            for i in range(3)
        ]


# ---------------------------------------------------------------------------
# Shared graph fixture (module-scoped: built once, reused across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph():
    llm = create_llm()
    stub = _StubRetriever()

    return KnowledgePilotGraphOrchestrator(
        query_analyzer=QueryAnalyzer(llm),
        query_decomposer=QueryDecomposer(llm),
        planner=Planner(llm),
        action_executor=ActionExecutor(
            vector_retriever=stub,
            web_search=stub,
        ),
        answer_synthesizer=AnswerSynthesizer(llm),
    )


def _run(graph, query: str) -> AgentState:
    initial: AgentState = {
        "query": query,
        "sub_queries": [],
        "tool_results": {},
        "iteration_count": 1,
        "citations": [],
        "hallucination_flag": False,
    }
    return asyncio.run(graph.ainvoke(initial))


# ---------------------------------------------------------------------------
# Path 1: conversation → END (answer set inline by query_analyzer)
# ---------------------------------------------------------------------------

def test_conversation_path(graph):
    state = _run(graph, "Hello!")
    assert state.get("query_complexity") == "conversation"
    assert state.get("answer"), "query_analyzer must set answer inline"
    # Retrieval nodes must NOT have run
    assert not state.get("tool_results"), "retrieval should be skipped for greetings"
    assert not state.get("citations"), "no citations for greeting"
    print(f"\n[conversation] answer: {state['answer'][:80]}")


# ---------------------------------------------------------------------------
# Path 2: simple → planner → executor → grader → synthesizer → END
# ---------------------------------------------------------------------------

def test_simple_path(graph):
    state = _run(graph, "What is attention mechanism in transformers?")
    assert state.get("query_complexity") == "simple"
    assert state.get("answer"), "synthesizer must produce an answer"
    assert state.get("citations"), "simple factual query must produce citations"
    assert state.get("iteration_count", 0) >= 1
    assert "hallucination_flag" in state
    print(f"\n[simple] complexity={state['query_complexity']}, "
          f"citations={len(state['citations'])}, "
          f"hallucination={state['hallucination_flag']}")


# ---------------------------------------------------------------------------
# Path 3: complex → decomposer → planner → ... → END
# ---------------------------------------------------------------------------

def test_complex_path(graph):
    state = _run(graph, "How does LoRA reduce training cost compared to full fine-tuning of large language models?")
    assert state.get("query_complexity") in ("complex", "simple"), (
        "complex or fallback-to-simple is acceptable"
    )
    assert state.get("answer"), "synthesizer must produce an answer"
    print(f"\n[complex] complexity={state['query_complexity']}, "
          f"sub_queries={state.get('sub_queries')}, "
          f"iterations={state.get('iteration_count')}")


# ---------------------------------------------------------------------------
# Path 4: multi_hop → decomposer → planner → ... → END
# ---------------------------------------------------------------------------

def test_multi_hop_path(graph):
    state = _run(
        graph,
        "Compare how HNSW and FAISS handle approximate nearest neighbour search "
        "and explain which is better for dynamic datasets."
    )
    assert state.get("query_complexity") in ("multi_hop", "complex", "simple"), (
        "multi_hop, complex, or simple fallback all acceptable"
    )
    assert state.get("answer"), "synthesizer must produce an answer"
    sub_queries = state.get("sub_queries") or []
    print(f"\n[multi_hop] complexity={state['query_complexity']}, "
          f"sub_queries={sub_queries}, "
          f"iterations={state.get('iteration_count')}")


# ---------------------------------------------------------------------------
# Routing invariants across all paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_complexity", [
    ("hello", "conversation"),
    ("what can you do", "conversation"),
])
def test_conversation_routing(graph, query, expected_complexity):
    state = _run(graph, query)
    assert state["query_complexity"] == expected_complexity
    assert not state.get("tool_results"), "retrieval must be skipped on conversation path"


def test_answer_always_set(graph):
    """Every non-error path must produce a non-empty answer."""
    for query in [
        "hi",
        "What is RAG?",
        "How does vLLM implement PagedAttention?",
    ]:
        state = _run(graph, query)
        assert state.get("answer"), f"No answer produced for: {query!r}"


def test_simple_path_no_hallucination_flag(graph):
    """A well-grounded answer with real citations must not be flagged as hallucination."""
    state = _run(graph, "What is the attention mechanism in transformers?")
    assert state.get("answer")
    assert state.get("hallucination_flag") is False, (
        f"Guard falsely flagged a valid answer: {state.get('answer')[:120]}"
    )


# ---------------------------------------------------------------------------
# Single-pass: no re-plan loop; answer produced on first retrieval pass
# ---------------------------------------------------------------------------

def test_single_pass_no_replan(graph):
    """Graph runs in a single pass — no re-plan loop, iteration_count stays at 1."""
    state = _run(graph, "What is retrieval augmented generation?")

    assert state.get("answer"), "Graph must produce an answer"
    assert state.get("iteration_count", 0) == 1, (
        f"Expected exactly 1 iteration (no re-plan loop), got {state.get('iteration_count')}"
    )
    print(f"\n[single-pass] iterations={state['iteration_count']}")
