# KnowledgePilot — Agentic RAG Assistant

An agentic retrieval-augmented generation system built with LangGraph. KnowledgePilot routes each query through a 5-node graph, retrieves evidence from a hybrid vector + keyword index, and returns a grounded answer with inline citations and a hallucination guard.

The AI/ML research corpus is illustrative — swap the PDFs and the graph is unchanged. The same architecture applies to legal document review, clinical guidelines, financial filings, or any domain where answers need to be traceable to sources.

---

## Demo

<!-- Add a screen recording or screenshot here -->
> **App demo coming soon**
>
> ![Demo placeholder](https://placehold.co/800x450?text=App+Demo)
>
> *Replace the image above with a GIF or screenshot of the running Streamlit app.*

---

## How it works

```
User query
    │
    ▼
Query Analyzer ──► conversation ──► Answer (direct)
    │
    ├──► simple ──────────────────────────────────┐
    │                                             │
    └──► complex / multi_hop                      │
              │                                   │
              ▼                                   │
       Query Decomposer                           │
              │                                   │
              ▼                                   ▼
           Planner ──► Action Executor ──► Answer Synthesizer
                           │                      │
                     vector_retriever         Hallucination
                     web_search                  Guard
```

- **Query Analyzer** — classifies intent and short-circuits greetings inline
- **Query Decomposer** — breaks complex questions into targeted sub-queries
- **Planner** — decides which tool (vector retriever or web search) to call for each sub-query
- **Action Executor** — runs all tool calls and collects Evidence objects
- **Answer Synthesizer** — produces a cited answer; hallucination guard validates every marker

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph StateGraph |
| LLM | Groq — Llama 3.3 70B (Gemini 2.5 Flash fallback) |
| Dense retrieval | FAISS + all-MiniLM-L6-v2 |
| Sparse retrieval | BM25 (rank-bm25) |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Web search | Tavily |
| PDF ingestion | LlamaParse (PyPDFLoader fallback) |
| Session memory | Firestore (in-memory fallback) |
| Evaluation | LangSmith — 25 golden test cases |
| API | FastAPI |
| Frontend | Streamlit |

---

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/Jwongjs/Knowledge-Pilot-Agentic-RAG.git
cd Knowledge-Pilot-Agentic-RAG/backend
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required:
- `GROQ_API_KEY` — [console.groq.com](https://console.groq.com) (free tier: 14,400 req/day)
- `TAVILY_API_KEY` — [tavily.com](https://tavily.com)

Optional:
- `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2=true` — LangSmith tracing
- `GOOGLE_API_KEY` + `LLM_PROVIDER=google` — Gemini 2.5 Flash fallback
- `LLAMA_CLOUD_API_KEY` — higher-quality PDF parsing via LlamaParse
- `FIRESTORE_PROJECT_ID` + `FIRESTORE_CREDENTIALS_JSON` — persistent sessions

### 3. Ingest the corpus

```bash
python ingest_papers.py
```

Builds FAISS and BM25 indices from the PDFs in `data/papers/`. To use your own documents, drop PDFs into that folder and re-run.

### 4. Start the backend

```bash
uvicorn main:app --reload
```

### 5. Start the frontend

```bash
cd ../frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## Run the tests

```bash
# From backend/
python -m pytest tests/test_graph_e2e.py tests/test_citations.py tests/test_eval.py -v
```

41 tests covering all 4 routing paths, citation marker extraction, hallucination guard logic, and evaluation harness utilities.

---

## Corpus

Six foundational AI/ML papers are included in `data/papers/`:

- Attention Is All You Need
- HNSW (Hierarchical Navigable Small World graphs)
- LoRA: Low-Rank Adaptation of Large Language Models
- RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- RAGAS: Automated Evaluation of Retrieval Augmented Generation
- Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)

---

## Project structure

```
backend/
├── main.py                        # FastAPI entry point
├── ingest_papers.py               # CLI: build indices from data/papers/
├── requirements.txt
├── models/
│   └── agent_state.py             # AgentState, Evidence, Citation, ToolCall
├── rag/
│   ├── graph_orchestrator.py      # LangGraph StateGraph definition
│   ├── knowledge_pilot_service.py # Service layer called by FastAPI
│   ├── llm_factory.py             # Groq / Gemini adapter
│   ├── conversation_store.py      # Firestore + in-memory session store
│   ├── nodes/                     # One file per graph node
│   └── retrievers/                # vector_retriever, web_search
├── ingest/                        # pdf_loader, docx_loader, index_builder
├── evaluation/                    # 25 golden cases + LangSmith evaluators
└── tests/                         # 41 pytest tests
data/
└── papers/                        # Source PDFs
frontend/
└── app.py                         # Streamlit UI
```
