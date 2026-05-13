"""CLI runner: loads golden dataset, runs the chain, uploads scores to LangSmith.

Usage:
    python -m evaluation.run_eval           # live mode (calls real APIs)
    python -m evaluation.run_eval --frozen  # frozen-cache mode (replays cached web/arxiv results)
"""
from __future__ import annotations
import asyncio
import hashlib
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langsmith import Client
from langsmith.evaluation import evaluate
from evaluation.golden_dataset import GOLDEN_DATASET
from evaluation.evaluators import (
    faithfulness_evaluator,
    citation_accuracy_evaluator,
    tool_selection_evaluator,
    re_retrieval_counter,
)
from rag.knowledge_pilot_service import KnowledgePilotService
from models.api_models import AskRequest

_DATASET_NAME = "KnowledgePilot-Golden-v1"
_CACHE_PATH = Path(__file__).parent / "eval_cache.pkl"

# ---------------------------------------------------------------------------
# Frozen-cache wrapper — replays saved Evidence lists for web/arxiv tools
# ---------------------------------------------------------------------------

class _CachedRetriever:
    """Wraps a live retriever; on cache hit returns saved Evidence, else calls live and saves."""

    def __init__(self, inner, tool_name: str, cache: dict, write: bool):
        self._inner = inner
        self._key_prefix = tool_name
        self._cache = cache
        self._write = write

    async def retrieve(self, query: str, **kwargs):
        key = f"{self._key_prefix}:{hashlib.md5(query.encode()).hexdigest()}"
        if key in self._cache:
            return self._cache[key]
        results = await self._inner.retrieve(query, **kwargs)
        if self._write:
            self._cache[key] = results
        return results


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        with _CACHE_PATH.open("rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with _CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)


# ---------------------------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------------------------

def _upload_dataset(client: Client) -> str:
    existing = [d.name for d in client.list_datasets()]
    if _DATASET_NAME in existing:
        return client.read_dataset(dataset_name=_DATASET_NAME).id

    dataset = client.create_dataset(_DATASET_NAME, description="25 curated golden test cases")
    client.create_examples(
        inputs=[
            {
                "query": case["query"],
                "expected_sources": case.get("expected_sources", []),
                "expected_tool_calls": case.get("expected_tool_calls", []),
            }
            for case in GOLDEN_DATASET
        ],
        outputs=[{"pass_criteria": case["pass_criteria"]} for case in GOLDEN_DATASET],
        dataset_id=dataset.id,
    )
    return dataset.id


# ---------------------------------------------------------------------------
# Service singleton (avoids 25x re-instantiation overhead)
# ---------------------------------------------------------------------------

_service: KnowledgePilotService | None = None


def _get_service(frozen: bool = False) -> KnowledgePilotService:
    global _service
    if _service is None:
        _service = _build_service(frozen)
    return _service


def _build_service(frozen: bool) -> KnowledgePilotService:
    """Build KnowledgePilotService; in frozen mode wrap web_search with cache."""
    svc = KnowledgePilotService()
    if not frozen:
        return svc

    cache = _load_cache()
    executor = svc._graph._nodes["action_executor"]  # type: ignore[attr-defined]
    web = executor._tools.get("web_search")
    if web:
        executor._tools["web_search"] = _CachedRetriever(web, "web_search", cache, write=True)
    svc._eval_cache = cache
    return svc


async def _run_chain(inputs: dict) -> dict:
    response = await _get_service().ask(AskRequest(query=inputs["query"]))
    return {
        "answer": response.answer,
        "query_complexity": response.query_complexity,
        "citations": response.citations,
        "hallucination_flag": response.hallucination_flag,
        "iteration_count": response.iteration_count,
    }


def run_chain_sync(inputs: dict) -> dict:
    return asyncio.run(_run_chain(inputs))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import sys
    frozen = "--frozen" in sys.argv

    # reads LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2 from env automatically
    client = Client()
    dataset_id = _upload_dataset(client)

    _get_service(frozen=frozen)  # initialise singleton before evaluate() forks

    results = evaluate(
        run_chain_sync,
        data=dataset_id,
        evaluators=[
            faithfulness_evaluator,
            citation_accuracy_evaluator,
            tool_selection_evaluator,
            re_retrieval_counter,
        ],
        experiment_prefix="KnowledgePilot",
        client=client,
    )

    # Persist cache after all runs complete (frozen or live populates it for next run)
    if hasattr(_service, "_eval_cache"):
        _save_cache(_service._eval_cache)  # type: ignore[union-attr]
        print(f"Cache saved to {_CACHE_PATH} ({len(_service._eval_cache)} entries)")  # type: ignore[union-attr]

    print(f"Evaluation complete. Results: {results.experiment_name}")


if __name__ == "__main__":
    main()
