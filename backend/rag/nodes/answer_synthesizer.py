from __future__ import annotations
from models.agent_state import AgentState, Evidence, Citation
from rag.nodes.hallucination_guard import check_hallucination


class AnswerSynthesizer:
    def __init__(self, llm):
        self._llm = llm

    async def run(self, state: AgentState) -> AgentState:
        all_evidence: list[Evidence] = [
            e for bucket in state.get("tool_results", {}).values() for e in bucket
        ]

        context_blocks = "\n\n".join(
            f"[{i+1}] ({e.source_type}) {e.content}"
            for i, e in enumerate(all_evidence)
        )

        prompt = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {state['query']}\n\nContext:\n{context_blocks}"},
        ]

        response = await self._llm.ainvoke(prompt)
        answer_text = response.content

        citations = _build_citations(all_evidence, state)
        hallucination_flag = check_hallucination(answer_text, all_evidence)

        return {
            **state,
            "answer": answer_text,
            "citations": citations,
            "hallucination_flag": hallucination_flag,
        }


_SNIPPET_LIMIT = 240


def _build_citations(evidence_list: list[Evidence], state: AgentState) -> list[Citation]:
    citations = []
    for i, e in enumerate(evidence_list):
        citations.append(Citation(
            citation_id=str(i + 1),
            source_type=e.source_type,
            evidence_id=e.evidence_id,
            source_document=e.metadata.get("source", "unknown"),
            url=e.url,
            page_number=e.page,
            section_title=e.metadata.get("section_title"),
            snippet=_truncate(e.content, _SNIPPET_LIMIT),
            dense_score=e.dense_score,
            rerank_score=e.rerank_score,
            published_date=e.metadata.get("published_date"),
            sub_query_origin=e.metadata.get("sub_query_origin"),
            tool_rationale=e.metadata.get("tool_rationale", ""),
            iteration=e.metadata.get("iteration", 1),
        ))
    return citations


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut == -1:
        cut = limit
    return text[:cut].rstrip(",;:.- ") + "…"


_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the question using ONLY the numbered context blocks provided. "
    "Cite each claim inline using superscript markers [¹], [²], etc. matching the context block numbers. "
    "Every factual claim must have at least one citation. Do not make claims not supported by the context. "
    "If the provided context does not contain sufficient information to answer, say so clearly."
)
