"""
Phase 4 citation tests: inline markers, Citation field population,
hallucination guard checks, and multi-hop provenance.

Run from backend/:
    python -m pytest tests/test_citations.py -v -s
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

from models.agent_state import AgentState, Evidence, ToolCall
from rag.nodes.answer_synthesizer import AnswerSynthesizer
from rag.nodes.hallucination_guard import check_hallucination, _CITATION_RE
from rag.nodes.action_executor import ActionExecutor
from rag.llm_factory import create_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(content: str, source: str, sub_query: str = "test query",
                   source_type: str = "vector_chunk", iteration: int = 1) -> Evidence:
    return Evidence(
        evidence_id=str(uuid.uuid4()),
        source_type=source_type,
        content=content,
        metadata={
            "source": source,
            "sub_query_origin": sub_query,
            "tool_rationale": "test",
            "iteration": iteration,
        },
        dense_score=0.9,
        rerank_score=0.85,
    )


def _state_with_evidence(evidence_list: list[Evidence], query: str = "test") -> AgentState:
    return {
        "query": query,
        "sub_queries": [],
        "tool_results": {"vector_retriever": evidence_list},
        "iteration_count": 1,
        "citations": [],
        "hallucination_flag": False,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthesizer():
    llm = create_llm()
    return AnswerSynthesizer(llm)


# ---------------------------------------------------------------------------
# Citation marker regex — handles all styles Gemini may produce
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_ids", [
    ("[1]",         ["1"]),
    ("[¹]",         ["¹"]),
    ("[^1]",        ["1"]),
    ("**[2]**",     ["2"]),
    ("[1][2]",      ["1", "2"]),
    ("no markers",  []),
])
def test_citation_re_patterns(text, expected_ids):
    # findall returns (g1, g2) tuples when the regex has 2 groups; flatten to non-empty strings
    raw = _CITATION_RE.findall(text)
    found = [g1 or g2 for g1, g2 in raw] if raw and isinstance(raw[0], tuple) else raw
    assert found == expected_ids, f"Pattern mismatch for {text!r}: got {found}"


# ---------------------------------------------------------------------------
# Citation field population (sub_query_origin, tool_rationale, iteration)
# ---------------------------------------------------------------------------

def test_action_executor_stamps_provenance():
    """ActionExecutor must stamp sub_query_origin, tool_rationale, iteration onto Evidence."""

    class _StubTool:
        async def retrieve(self, query, **kwargs):
            return [Evidence(
                evidence_id=str(uuid.uuid4()),
                source_type="vector_chunk",
                content="HNSW uses hierarchical navigable small world graphs.",
                metadata={"source": "HNSW.pdf"},
            )]

    executor = ActionExecutor(
        vector_retriever=_StubTool(),
        web_search=_StubTool(),
    )
    plan = [ToolCall(
        tool_name="vector_retriever",
        args={},
        sub_query="how does HNSW work",
        rationale="factual query about indexed paper",
        expected_evidence_type="vector_chunk",
    )]
    state: AgentState = {
        "query": "how does HNSW work",
        "action_plan": plan,
        "tool_results": {},
        "iteration_count": 2,
    }
    result = asyncio.run(executor.run(state))
    evidence = result["tool_results"]["vector_retriever"][0]
    assert evidence.metadata["sub_query_origin"] == "how does HNSW work"
    assert evidence.metadata["tool_rationale"] == "factual query about indexed paper"
    assert evidence.metadata["iteration"] == 2


# ---------------------------------------------------------------------------
# AnswerSynthesizer — inline markers and Citation object fields
# ---------------------------------------------------------------------------

def test_synthesizer_produces_citations(synthesizer):
    evidence = [
        _make_evidence(
            "The attention mechanism computes a weighted sum of values "
            "based on query-key similarity scores.",
            source="Attention is All You Need.pdf",
            sub_query="how does attention work",
        )
    ]
    state = _state_with_evidence(evidence, query="How does attention work?")
    result = asyncio.run(synthesizer.run(state))

    assert result.get("answer"), "synthesizer must produce an answer"
    assert result.get("citations"), "synthesizer must produce citations"
    c = result["citations"][0]
    assert c.citation_id == "1"
    assert c.source_type == "vector_chunk"
    assert c.source_document == "Attention is All You Need.pdf"
    assert c.snippet
    assert c.dense_score == 0.9
    assert c.rerank_score == 0.85
    assert c.sub_query_origin == "how does attention work"
    assert c.tool_rationale == "test"
    assert c.iteration == 1
    print(f"\n[synthesizer] answer snippet: {result['answer'][:100]}")


def test_synthesizer_answer_contains_marker(synthesizer):
    """The LLM answer must contain at least one citation marker."""
    evidence = [
        _make_evidence(
            "LoRA reduces trainable parameters by decomposing weight matrices "
            "into two low-rank matrices, reducing GPU memory by up to 3x.",
            source="LoRA.pdf",
            sub_query="how does LoRA reduce parameters",
        )
    ]
    state = _state_with_evidence(evidence, query="How does LoRA reduce trainable parameters?")
    result = asyncio.run(synthesizer.run(state))
    assert _CITATION_RE.search(result["answer"]), (
        f"No citation marker found in answer:\n{result['answer']}"
    )


# ---------------------------------------------------------------------------
# Multi-hop provenance — different sub-queries → distinct sub_query_origin
# ---------------------------------------------------------------------------

def test_multihop_citation_provenance(synthesizer):
    """Citations from different sub-queries must carry distinct sub_query_origin values."""
    evidence = [
        _make_evidence(
            "HNSW builds a hierarchical graph structure for approximate nearest neighbour search.",
            source="HNSW.pdf",
            sub_query="how does HNSW work",
            iteration=1,
        ),
        _make_evidence(
            "FAISS uses flat index with brute-force search or IVF for partitioned search.",
            source="RAG.pdf",
            sub_query="how does FAISS work",
            iteration=1,
        ),
    ]
    state: AgentState = {
        "query": "Compare HNSW and FAISS for vector search",
        "sub_queries": ["how does HNSW work", "how does FAISS work"],
        "tool_results": {"vector_retriever": evidence},
        "iteration_count": 1,
        "citations": [],
        "hallucination_flag": False,
    }
    result = asyncio.run(synthesizer.run(state))
    origins = {c.sub_query_origin for c in result["citations"]}
    assert len(origins) == 2, (
        f"Expected 2 distinct sub_query_origin values, got: {origins}"
    )
    assert "how does HNSW work" in origins
    assert "how does FAISS work" in origins
    print(f"\n[multi-hop] citation origins: {origins}")


# ---------------------------------------------------------------------------
# Hallucination check — all three validations (now inline in answer_synthesizer)
# ---------------------------------------------------------------------------

def test_guard_passes_valid_answer():
    evidence = [_make_evidence("LoRA uses rank decomposition. It reduces parameters.", "LoRA.pdf")]
    answer = "LoRA uses rank decomposition [1]."
    assert check_hallucination(answer, evidence) is False


def test_guard_flags_invalid_citation_id():
    """Citing [99] when only 1 evidence item exists must flag hallucination."""
    evidence = [_make_evidence("some content", "paper.pdf")]
    answer = "This is a claim [99]."
    assert check_hallucination(answer, evidence) is True


def test_guard_flags_uncited_number():
    """A sentence with a bare number and no citation must flag hallucination."""
    evidence = [_make_evidence("some content", "paper.pdf")]
    answer = "The model achieves 94.5 accuracy on the benchmark."
    assert check_hallucination(answer, evidence) is True


def test_guard_flags_misaligned_citation():
    """A citation whose chunk shares no tokens with the citing sentence must flag."""
    evidence = [_make_evidence("quantum physics wavelength spectrum photon", "physics.pdf")]
    answer = "Transformers use multi-head attention mechanisms [1]."
    assert check_hallucination(answer, evidence) is True


def test_guard_passes_multisource_answer():
    """Answer citing two different sources correctly must not flag hallucination."""
    evidence = [
        _make_evidence("attention computes weighted values queries keys", "Attention.pdf"),
        _make_evidence("HNSW hierarchical graph layers navigable small world", "HNSW.pdf"),
    ]
    answer = "Attention uses queries and keys [1]. HNSW builds hierarchical graph layers [2]."
    assert check_hallucination(answer, evidence) is False
