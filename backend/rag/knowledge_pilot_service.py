from __future__ import annotations
import os
from rag.llm_factory import create_llm
from rag.graph_orchestrator import KnowledgePilotGraphOrchestrator
from rag.nodes.query_analyzer import QueryAnalyzer
from rag.nodes.query_decomposer import QueryDecomposer
from rag.nodes.planner import Planner
from rag.nodes.action_executor import ActionExecutor
from rag.nodes.answer_synthesizer import AnswerSynthesizer
from rag.retrievers.vector_retriever import VectorRetriever
from rag.retrievers.web_search import WebSearch
from ingest.index_builder import IndexBuilder
from rag.conversation_store import ConversationStore
from pathlib import Path
from models.api_models import AskRequest, AskResponse, IngestRequest, IngestResponse, UploadResponse
from models.agent_state import AgentState


class KnowledgePilotService:
    def __init__(self):
        llm = create_llm(
            provider=os.getenv("LLM_PROVIDER", "groq"),
            model=os.getenv("LLM_MODEL") or None,
        )

        index_builder = IndexBuilder()
        faiss_store, bm25_retriever = index_builder.load_indices()

        self._vector_retriever = VectorRetriever(faiss_store, bm25_retriever)
        web_search = WebSearch(api_key=os.environ["TAVILY_API_KEY"])

        action_executor = ActionExecutor(self._vector_retriever, web_search)
        corpus_titles = [_strip_ext(s["name"]) for s in index_builder.list_sources()]

        self._planner = Planner(llm, corpus_titles=corpus_titles)
        self._graph = KnowledgePilotGraphOrchestrator(
            query_analyzer=QueryAnalyzer(llm),
            query_decomposer=QueryDecomposer(llm),
            planner=self._planner,
            action_executor=action_executor,
            answer_synthesizer=AnswerSynthesizer(llm),
        )
        self._index_builder = index_builder
        self._store = ConversationStore()

    async def ask(self, request: AskRequest) -> AskResponse:
        session_id = request.session_id or "default"
        history = self._store.get_history(session_id)
        history_text = "\n".join(f"{t.role}: {t.content}" for t in history[-6:])

        run_id = str(__import__("uuid").uuid4())
        initial_state: AgentState = {
            "query": request.query,
            "conversation_history": history_text,
            "sub_queries": [],
            "tool_results": {},
            "iteration_count": 1,
            "citations": [],
            "hallucination_flag": False,
        }
        final_state: AgentState = await self._graph.ainvoke(
            initial_state,
            config={"run_id": run_id},
        )

        self._store.append_turn(session_id, "user", request.query)
        self._store.append_turn(session_id, "assistant", final_state.get("answer", ""))

        trace_url = _build_trace_url(run_id)
        return AskResponse(
            answer=final_state.get("answer", ""),
            query_complexity=final_state.get("query_complexity", "simple"),
            sub_queries=final_state.get("sub_queries", []),
            citations=[_citation_to_dict(c) for c in final_state.get("citations", [])],
            hallucination_flag=final_state.get("hallucination_flag", False),
            iteration_count=final_state.get("iteration_count", 1),
            langsmith_trace_url=trace_url,
        )

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        count = await self._index_builder.ingest(request.source_path, request.doc_type)
        return IngestResponse(chunks_indexed=count, source_path=request.source_path)

    async def upload(self, filename: str, file_bytes: bytes, doc_type: str = "research_paper") -> UploadResponse:
        suffix = Path(filename).suffix.lower()
        if suffix not in (".pdf", ".docx"):
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .docx")

        target_dir = Path(__file__).parent.parent.parent / "data" / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        target_path.write_bytes(file_bytes)

        count = await self._index_builder.ingest(str(target_path), doc_type)

        sources = self._index_builder.list_sources()
        self._planner._corpus_titles = [_strip_ext(s["name"]) for s in sources]
        faiss_store, bm25_retriever = self._index_builder.load_indices()
        self._vector_retriever.refresh(faiss_store, bm25_retriever)

        return UploadResponse(
            filename=filename,
            chunks_indexed=count,
            total_documents=len(sources),
        )

    async def list_documents(self) -> list[dict]:
        return self._index_builder.list_sources()


def _strip_ext(name: str) -> str:
    for ext in (".pdf", ".docx"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _build_trace_url(run_id: str) -> str | None:
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return None
    project = os.environ.get("LANGCHAIN_PROJECT", "KnowledgePilot")
    return f"https://smith.langchain.com/public/{run_id}/r?project={project}"


def _citation_to_dict(citation) -> dict:
    return {
        "citation_id": citation.citation_id,
        "source_type": citation.source_type,
        "source_document": citation.source_document,
        "url": citation.url,
        "page_number": citation.page_number,
        "section_title": citation.section_title,
        "snippet": citation.snippet,
        "dense_score": citation.dense_score,
        "rerank_score": citation.rerank_score,
        "published_date": citation.published_date,
        "sub_query_origin": citation.sub_query_origin,
        "tool_rationale": citation.tool_rationale,
        "iteration": citation.iteration,
    }
