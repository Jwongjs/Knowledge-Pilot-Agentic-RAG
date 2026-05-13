from __future__ import annotations
from pydantic import BaseModel
from models.agent_state import AgentState, ToolCall


class PlannerOutput(BaseModel):
    tool_calls: list[dict]


class Planner:
    def __init__(self, llm, corpus_titles: list[str] | None = None):
        self._llm = llm.with_structured_output(PlannerOutput)
        self._corpus_titles = corpus_titles or []

    async def run(self, state: AgentState) -> AgentState:
        queries = state.get("sub_queries") or [state["query"]]

        try:
            result: PlannerOutput = await self._llm.ainvoke(
                _build_prompt(queries, self._corpus_titles)
            )
            raw_calls = result.tool_calls
            for i, tc in enumerate(raw_calls):
                if not tc.get("sub_query"):
                    tc["sub_query"] = queries[i] if i < len(queries) else state["query"]
        except Exception:
            raw_calls = [
                {"tool_name": "vector_retriever", "args": {}, "sub_query": q,
                 "rationale": "default fallback", "expected_evidence_type": "vector_chunk"}
                for q in queries
            ]

        tool_calls = [_normalise(tc, state) for tc in raw_calls]
        return {**state, "action_plan": tool_calls}


def _normalise(tc: dict, state: AgentState) -> ToolCall:
    query = tc.get("sub_query", state.get("query", ""))
    tool = tc.get("tool_name", "vector_retriever")
    if tool not in ("vector_retriever", "web_search"):
        tool = "vector_retriever"
    return ToolCall(
        tool_name=tool,
        args=tc.get("args", {}),
        sub_query=query,
        rationale=tc.get("rationale", ""),
        expected_evidence_type=tc.get("expected_evidence_type", "vector_chunk"),
    )


def _build_prompt(queries: list[str], corpus_titles: list[str]) -> str:
    corpus_block = (
        f"LOCAL CORPUS (indexed papers, searchable via vector_retriever):\n"
        f"{chr(10).join(f'  - {t}' for t in corpus_titles)}\n\n"
        if corpus_titles
        else ""
    )
    return (
        "You are a Planner for a RAG system with a local corpus of AI/ML research papers.\n\n"
        f"{corpus_block}"
        "TOOL SELECTION RULES:\n"
        "1. vector_retriever — Use for ANY question about a paper or topic that appears in the "
        "LOCAL CORPUS above. This is the default for in-corpus academic topics.\n"
        "2. web_search — Use for anything NOT in the LOCAL CORPUS: papers we haven't indexed, "
        "recent news/releases, library documentation, or any out-of-corpus question.\n\n"
        "For each sub-query, return one tool_call with fields: "
        "tool_name, sub_query, rationale (one sentence), args (empty dict {}), expected_evidence_type.\n\n"
        f"Sub-queries: {queries}\n\nReturn JSON with 'tool_calls' list."
    )
