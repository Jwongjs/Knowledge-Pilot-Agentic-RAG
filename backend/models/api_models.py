from pydantic import BaseModel
from typing import Optional
from models.agent_state import Citation


class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    query_complexity: str
    sub_queries: list[str] = []
    citations: list[dict] = []
    hallucination_flag: bool = False
    iteration_count: int = 1
    langsmith_trace_url: Optional[str] = None


class IngestRequest(BaseModel):
    source_path: str
    doc_type: str  # "pdf" | "html" | "faq"


class IngestResponse(BaseModel):
    chunks_indexed: int
    source_path: str


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    total_documents: int