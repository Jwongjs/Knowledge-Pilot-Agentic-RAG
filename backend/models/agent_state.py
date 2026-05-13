from __future__ import annotations
from typing import TypedDict, Literal, Optional
from dataclasses import dataclass, field


@dataclass
class Evidence:
    evidence_id: str
    source_type: Literal["vector_chunk", "web"]
    content: str
    metadata: dict = field(default_factory=dict)
    dense_score: Optional[float] = None
    rerank_score: Optional[float] = None
    url: Optional[str] = None
    page: Optional[int] = None


@dataclass
class ToolCall:
    tool_name: Literal["vector_retriever", "web_search"]
    args: dict
    sub_query: str
    rationale: str
    expected_evidence_type: str


@dataclass
class Citation:
    citation_id: str
    source_type: Literal["vector_chunk", "web"]
    evidence_id: str
    source_document: str
    url: Optional[str]
    page_number: Optional[int]
    section_title: Optional[str]
    snippet: str
    dense_score: Optional[float]
    rerank_score: Optional[float]
    published_date: Optional[str]
    sub_query_origin: Optional[str]
    tool_rationale: str
    iteration: int


class AgentState(TypedDict, total=False):
    query: str
    conversation_history: str  # last N turns formatted as "role: content\n..."
    query_complexity: Literal["conversation", "simple", "complex", "multi_hop"]
    sub_queries: list[str]
    action_plan: list[ToolCall]
    tool_results: dict[str, list[Evidence]]
    iteration_count: int
    answer: str
    citations: list[Citation]
    hallucination_flag: bool