from dotenv import load_dotenv
load_dotenv()  # must run before any LangChain/OpenAI imports resolve env vars

from fastapi import FastAPI, HTTPException
from models.api_models import AskRequest, AskResponse, IngestRequest, IngestResponse
from rag.knowledge_pilot_service import KnowledgePilotService

app = FastAPI(title="KnowledgePilot API")

_service = KnowledgePilotService()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        return await _service.ask(request)
    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "rate_limit_exceeded" in msg or "429" in msg or "spending cap" in msg:
            raise HTTPException(status_code=429, detail="LLM API quota exhausted. Check your provider's rate limits.")
        raise HTTPException(status_code=500, detail=msg)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    try:
        return await _service.ingest(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    return await _service.list_documents()
