from __future__ import annotations
import re
from pydantic import BaseModel
from models.agent_state import AgentState

_STOPWORDS = {"a", "an", "the", "is", "in", "of", "and", "or", "to", "for", "with", "that", "this"}


class DecompositionResult(BaseModel):
    sub_queries: list[str]


def _jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split()) - _STOPWORDS
    tb = set(b.lower().split()) - _STOPWORDS
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class QueryDecomposer:
    def __init__(self, llm):
        self._llm = llm.with_structured_output(DecompositionResult)

    async def run(self, state: AgentState) -> AgentState:
        try:
            result: DecompositionResult = await self._llm.ainvoke(
                _SYSTEM_PROMPT + f"\n\nQuery: {state['query']}"
            )
            sub_queries = result.sub_queries[:3]  # hard cap at 3
        except Exception:
            return {**state, "sub_queries": []}

        # Sanity gate: only 1 sub-query → skip decomposition
        if len(sub_queries) <= 1:
            return {**state, "sub_queries": []}

        # Sanity gate: pairwise Jaccard overlap > 0.7 → skip decomposition
        for i in range(len(sub_queries)):
            for j in range(i + 1, len(sub_queries)):
                if _jaccard(sub_queries[i], sub_queries[j]) > 0.7:
                    return {**state, "sub_queries": []}

        # Sanity gate: ensure named entities from original query are preserved
        original_entities = _extract_entities(state["query"])
        for entity in original_entities:
            if not any(entity.lower() in sq.lower() for sq in sub_queries):
                sub_queries[-1] += f" {entity}"

        return {**state, "sub_queries": sub_queries}


def _extract_entities(query: str) -> list[str]:
    # Heuristic: capitalised multi-word phrases and quoted strings
    matches = re.findall(r'"([^"]+)"|([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)', query)
    return [m[0] or m[1] for m in matches if m[0] or m[1]]


_SYSTEM_PROMPT = """Break the following query into 2-3 atomic, non-overlapping sub-queries.
Each sub-query must be independently retrievable. Hard cap: 3 sub-queries.
Return JSON matching the schema with a 'sub_queries' list."""
