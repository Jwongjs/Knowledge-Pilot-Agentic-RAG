from __future__ import annotations
import uuid
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from models.agent_state import Evidence

_DENSE_TOP_K = 20
_SPARSE_TOP_K = 20
_ENSEMBLE_TOP_K = 15
_RERANK_TOP_K = 5
_DENSE_WEIGHT = 0.6
_SPARSE_WEIGHT = 0.4

# So the same 470 chunks are indexed twice, in two completely different data structures. 
# The EnsembleRetriever runs both in parallel, fuses their ranked lists with RRF, 
# then the CrossEncoder re-scores the merged top-15 to pick the best 5
# User asks a question
#         │
#         ▼
#   VectorRetriever.retrieve(query)
#         │
#         ├── FAISS retriever  → top-20 by cosine similarity
#         ├── BM25 retriever   → top-20 by keyword score
#         │
#         ├── EnsembleRetriever (RRF fusion)  → merged top-15
#         │
#         └── CrossEncoder reranker  → rescores all 15 → returns top-5 Evidence

class VectorRetriever:
    def __init__(self, faiss_store: FAISS, bm25_retriever: BM25Retriever):
        self._faiss = faiss_store
        self._ensemble = EnsembleRetriever(
            retrievers=[
                faiss_store.as_retriever(search_kwargs={"k": _DENSE_TOP_K}),
                bm25_retriever,
            ],
            weights=[_DENSE_WEIGHT, _SPARSE_WEIGHT],
        )
        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    async def retrieve(self, query: str, **kwargs) -> list[Evidence]:
        import asyncio

        # Ensemble (FAISS + BM25 fused) — provides the ranked doc list
        docs = await self._ensemble.ainvoke(query)
        docs = docs[:_ENSEMBLE_TOP_K]

        # Separate scored FAISS search to get real dense scores (ensemble drops them)
        scored_docs: list[tuple] = await asyncio.to_thread(
            self._faiss.similarity_search_with_score, query, k=_DENSE_TOP_K
        )
        dense_scores = {d.page_content: float(s) for d, s in scored_docs}

        pairs = [(query, doc.page_content) for doc in docs]
        scores = await asyncio.to_thread(self._reranker.predict, pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:_RERANK_TOP_K]

        return [
            Evidence(
                evidence_id=str(uuid.uuid4()),
                source_type="vector_chunk",
                content=doc.page_content,
                metadata=doc.metadata,
                dense_score=dense_scores.get(doc.page_content, 0.0),
                rerank_score=float(score),
                page=doc.metadata.get("page"),
            )
            for doc, score in top
        ]
