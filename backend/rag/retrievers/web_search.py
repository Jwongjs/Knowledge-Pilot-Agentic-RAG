from __future__ import annotations
import asyncio
import uuid
from tavily import TavilyClient
from models.agent_state import Evidence


class WebSearch:
    def __init__(self, api_key: str):
        self._client = TavilyClient(api_key=api_key)

    async def retrieve(self, query: str, **kwargs) -> list[Evidence]:
        response = await asyncio.to_thread(
            self._client.search,
            query,
            search_depth="advanced",
            max_results=5,
        )
        results = response.get("results", [])
        return [
            Evidence(
                evidence_id=str(uuid.uuid4()),
                source_type="web",
                content=r.get("content", r.get("snippet", "")),
                metadata={
                    "source": r.get("title", ""),
                    "published_date": r.get("published_date"),
                },
                url=r.get("url"),
            )
            for r in results
        ]
