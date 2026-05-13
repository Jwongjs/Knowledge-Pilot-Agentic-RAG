from __future__ import annotations
from typing import Literal
from langgraph.graph import StateGraph, END
from models.agent_state import AgentState


def _route_after_analyzer(state: AgentState) -> Literal["__end__", "query_decomposer", "planner"]:
    complexity = state.get("query_complexity", "simple")
    if complexity == "conversation":
        return "__end__"
    if complexity in ("complex", "multi_hop"):
        return "query_decomposer"
    return "planner"


class KnowledgePilotGraphOrchestrator:
    def __init__(
        self,
        query_analyzer,
        query_decomposer,
        planner,
        action_executor,
        answer_synthesizer,
    ):
        self._nodes = {
            "query_analyzer": query_analyzer,
            "query_decomposer": query_decomposer,
            "planner": planner,
            "action_executor": action_executor,
            "answer_synthesizer": answer_synthesizer,
        }
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(AgentState)

        for name, node in self._nodes.items():
            g.add_node(name, node.run)

        g.set_entry_point("query_analyzer")

        g.add_conditional_edges(
            "query_analyzer",
            _route_after_analyzer,
            {
                "__end__": END,
                "query_decomposer": "query_decomposer",
                "planner": "planner",
            },
        )

        g.add_edge("query_decomposer", "planner")
        g.add_edge("planner", "action_executor")
        g.add_edge("action_executor", "answer_synthesizer")
        g.add_edge("answer_synthesizer", END)

        return g.compile()

    async def ainvoke(self, state: AgentState, config: dict | None = None) -> AgentState:
        return await self._graph.ainvoke(state, config=config)
