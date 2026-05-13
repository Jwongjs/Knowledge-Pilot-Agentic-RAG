from __future__ import annotations
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.llm_factory import create_llm

_FAITHFULNESS_PROMPT = """\
You are a faithfulness judge. Given an answer and the source snippets it was derived from, \
score whether every factual claim in the answer is supported by the snippets.

Answer:
{answer}

Source snippets:
{snippets}

Respond with ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
score=1.0 means every claim is grounded; score=0.0 means the answer is entirely hallucinated.
"""


def _get_llm():
    return create_llm()


def faithfulness_evaluator(run, example) -> dict:
    """LLM-as-judge: are answer claims grounded in retrieved Evidence?"""
    answer = run.outputs.get("answer", "")
    citations = run.outputs.get("citations", [])
    if not answer:
        return {"key": "faithfulness", "score": 0.0, "comment": "No answer"}
    if not citations:
        return {"key": "faithfulness", "score": 0.0, "comment": "No citations found"}

    snippets = "\n".join(
        f"[{c.get('citation_id', i+1)}] {c.get('snippet', '')}"
        for i, c in enumerate(citations)
    )
    prompt = _FAITHFULNESS_PROMPT.format(answer=answer, snippets=snippets)

    try:
        import json, re
        llm = _get_llm()
        response = llm.invoke(prompt)
        raw = response.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        reason = data.get("reason", "")
        return {"key": "faithfulness", "score": score, "comment": reason}
    except Exception as exc:
        return {"key": "faithfulness", "score": 0.0, "comment": f"judge error: {exc}"}


def citation_accuracy_evaluator(run, example) -> dict:
    """Verifies inline citation markers correspond to cited sources with correct source_type."""
    expected_sources = example.inputs.get("expected_sources", [])
    actual_citations = run.outputs.get("citations", [])
    actual_types = {c.get("source_type") for c in actual_citations}

    if not expected_sources:
        return {"key": "citation_accuracy", "score": 1.0}

    hits = sum(1 for s in expected_sources if s in actual_types)
    score = hits / len(expected_sources)
    return {"key": "citation_accuracy", "score": score}


def tool_selection_evaluator(run, example) -> dict:
    """For tool-selection test cases: asserts Planner's action_plan includes expected tool_name."""
    expected_tools = example.inputs.get("expected_tool_calls", [])
    if not expected_tools:
        return {"key": "tool_selection", "score": 1.0, "comment": "N/A"}

    # action_plan is not directly in outputs; inferred from citation source_types as proxy
    actual_citations = run.outputs.get("citations", [])
    actual_types = {c.get("source_type") for c in actual_citations}

    tool_to_source = {
        "web_search": "web",
        "vector_retriever": "vector_chunk",
    }

    hits = sum(1 for t in expected_tools if tool_to_source.get(t) in actual_types)
    score = hits / len(expected_tools)
    return {"key": "tool_selection", "score": score}


def re_retrieval_counter(run, example) -> dict:
    """Counts whether re-planning fired (iteration_count > 1)."""
    iteration_count = run.outputs.get("iteration_count", 1)
    fired = int(iteration_count > 1)
    return {"key": "re_retrieval_fired", "score": fired}
