from __future__ import annotations
import hashlib
from models.agent_state import AgentState, Evidence, ToolCall


class ActionExecutor:
    def __init__(self, vector_retriever, web_search, langsmith_client=None):
        self._tools = {
            "vector_retriever": vector_retriever,
            "web_search": web_search,
        }
        self._langsmith = langsmith_client

    async def run(self, state: AgentState) -> AgentState:
        plan: list[ToolCall] = state.get("action_plan", [])
        existing: dict[str, list[Evidence]] = state.get("tool_results", {})
        seen_hashes: set[str] = _build_seen_hashes(existing)

        for tool_call in plan:
            tool = self._tools.get(tool_call.tool_name)
            if tool is None:
                continue
            try:
                safe_args = {k: v for k, v in tool_call.args.items() if k != "query"}
                results: list[Evidence] = await tool.retrieve(tool_call.sub_query, **safe_args)
            except Exception as exc:
                # Log failure; agent continues with remaining tools
                if self._langsmith:
                    self._langsmith.log_error(str(exc))
                continue

            bucket = existing.setdefault(tool_call.tool_name, [])
            iteration = state.get("iteration_count", 0)
            for evidence in results:
                h = _content_hash(evidence)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    evidence.metadata["sub_query_origin"] = tool_call.sub_query
                    evidence.metadata["tool_rationale"] = tool_call.rationale
                    evidence.metadata["iteration"] = iteration
                    bucket.append(evidence)

        return {**state, "tool_results": existing}


def _content_hash(e: Evidence) -> str:
    key = f"{e.source_type}:{e.content[:200]}"
    return hashlib.md5(key.encode()).hexdigest()


def _build_seen_hashes(tool_results: dict[str, list[Evidence]]) -> set[str]:
    return {_content_hash(e) for bucket in tool_results.values() for e in bucket}
