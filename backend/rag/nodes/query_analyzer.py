from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from models.agent_state import AgentState


class QueryAnalysisResult(BaseModel):
    query_complexity: Literal["conversation", "simple", "complex", "multi_hop"]
    entities: list[str]
    out_of_corpus: bool = False


_GREETING_TOKENS = {"hi", "hello", "hey", "thanks", "bye", "goodbye", "see you", "good morning"}
_META_PHRASES = {"who are you", "what can you do", "how do you work", "what are you", "tell me about yourself"}
_FACTUAL_SIGNALS = {"what is", "how does", "define", "explain", "compare"}

_GREETING_TEMPLATE = (
    "Hi! I can help you with questions about AI/ML papers, LangChain docs, and RAG best practices. "
    'Try asking: "What is cosine similarity?" or "How does hybrid retrieval work?"'
)

_META_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer only about your actual capabilities: answering factual questions "
    "about AI/ML papers, LangChain/LangGraph/FAISS documentation, and RAG best practices. "
    "Do not hallucinate capabilities or promise to do things outside this scope. Be concise (2-3 sentences)."
)


class QueryAnalyzer:
    def __init__(self, llm):
        self._llm = llm
        self._structured_llm = llm.with_structured_output(QueryAnalysisResult)

    async def run(self, state: AgentState) -> AgentState:
        query = state["query"].lower().strip()

        is_greeting = any(tok in query for tok in _GREETING_TOKENS)
        is_meta = any(phrase in query for phrase in _META_PHRASES)
        has_factual = any(sig in query for sig in _FACTUAL_SIGNALS)

        if (is_greeting or is_meta) and not has_factual:
            answer = await self._compose_conversation_reply(state["query"], is_meta)
            return {
                **state,
                "query_complexity": "conversation",
                "answer": answer,
            }

        try:
            result: QueryAnalysisResult = await self._structured_llm.ainvoke(
                _SYSTEM_PROMPT + f"\n\nQuery: {state['query']}"
            )
            return {**state, "query_complexity": result.query_complexity}
        except Exception:
            return {**state, "query_complexity": "simple"}

    async def _compose_conversation_reply(self, raw_query: str, is_meta: bool) -> str:
        if not is_meta:
            return _GREETING_TEMPLATE
        try:
            response = await self._llm.ainvoke(
                [{"role": "system", "content": _META_SYSTEM_PROMPT},
                 {"role": "user", "content": raw_query}]
            )
            return response.content
        except Exception:
            return _GREETING_TEMPLATE


_SYSTEM_PROMPT = """Classify the user query into exactly one category:
- conversation: pure social/meta input with NO domain entities and NO factual question-words
- simple: single-concept factual question answerable from one chunk
- complex: multi-concept but concepts likely co-occur in the same document
- multi_hop: explicit cross-source comparison (compare/differ/versus + entities from different sub-corpora)

When uncertain, fall back to 'simple'. Return JSON matching the schema."""
