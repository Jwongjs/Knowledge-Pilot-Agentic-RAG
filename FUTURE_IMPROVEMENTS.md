# Future Improvements

Production-readiness and exploration items deferred from the v1 build. Each entry has a rationale, a sketch of the change, and the reason it was deferred.

The intent is for this doc to be a public artifact during interviews / handoffs: it shows what was *considered and rejected* alongside what was *built*, which is usually a stronger signal than the code alone.

---

## Retrieval Quality

### Cohere Rerank API
**Swap target:** [backend/rag/retrievers/vector_retriever.py](backend/rag/retrievers/vector_retriever.py) — replace the local CrossEncoder (`ms-marco-MiniLM-L-6-v2`) with Cohere's hosted `rerank-v3`.

**Why:**
- Hosted rerank-v3 is consistently 5-10pp better on BEIR than `ms-marco-MiniLM-L-6-v2`.
- ~30ms p50 latency vs ~80-200ms local (no model load, runs on Cohere's GPUs).
- Drops one heavyweight ML dependency from the deployment image.

**Tradeoffs:**
- ~$1 per 1000 reranks at production scale.
- Network dependency adds tail-latency variance.
- Vendor lock-in on a hot-path component.

**Sketch:**
```python
import cohere
co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
reranked = co.rerank(
    model="rerank-v3-multilingual",
    query=query,
    documents=[c.page_content for c in candidates],
    top_n=5,
)
return [candidates[r.index] for r in reranked.results]
```

**Status:** deferred — local CrossEncoder is fast enough for the demo and avoids an extra API key requirement.

---

### Exa.ai Semantic Web Search
**Swap target:** [backend/rag/retrievers/web_search.py](backend/rag/retrievers/web_search.py) — replace Tavily with Exa.ai.

**Why:**
- Built for agent retrieval (neural search, not SERP-style keyword matching).
- Returns full page content + autoprompt rewriting in one call.
- Likely to improve T18/T21/T22 in the golden eval (out-of-corpus paper title queries).

**Tradeoffs:**
- Tavily is currently working and has a generous free tier.
- Exa pricing similar order of magnitude after free tier.
- Adds another vendor to the dependency surface.

**Status:** deferred. A commented-out swap block will be added to `web_search.py` so the migration is a 5-line change when the time comes.

---

### Dedicated paper-discovery tool (arxiv / Semantic Scholar / OpenAlex)
**Where it fits:** a third tool alongside `vector_retriever` and `web_search`, called when the user wants to *find* relevant papers (not just answer a question from a known one).

**Why considered:**
- Returns structured paper metadata (authors, abstract, citation counts, PDF URLs) rather than web snippets.
- Stronger signal than Tavily for "what's the best paper on X" style queries.

**Why deferred:**
- v1 already supports the same workflow via *web_search → user uploads the PDF → vector_retriever queries it*. The user-driven curation step is a feature, not a friction point — it keeps the corpus relevant rather than auto-bloating it.
- A third tool means a third planner branch, a third evidence type, and ranking logic to merge structured paper hits with web results.
- The 25-case eval set doesn't include "find me papers" queries, so improvement would be unmeasured.

**When to revisit:** if a user need emerges for autonomous literature surveys (e.g. "give me the top 10 papers on retrieval-augmented generation since 2024").

---

### Contextual Retrieval (Anthropic technique)
**Where it fits:** during ingestion in [backend/ingest/pdf_loader.py](backend/ingest/pdf_loader.py) — prepend each chunk with a 50-100 token LLM-generated summary of its position in the document before embedding.

**Why considered:**
- Anthropic reports ~35% reduction in retrieval failures vs naive chunking.
- One-time cost at ingestion (~500-1500 LLM calls for the current 6-paper corpus).

**Why deferred:**
- The current 25-case eval set probably doesn't have enough long-tail queries to *show* the improvement vs CrossEncoder reranking.
- Adds an ingestion-time dependency on the LLM provider, complicating cold-start.
- Hybrid retrieval (FAISS+BM25) + CrossEncoder already captures most of the gains for a small, curated corpus.

**When to revisit:** if eval pass rate plateaus and analysis shows retrieval (not generation) is the bottleneck — likely once the corpus grows past ~30 documents.

---

## Observability

### Langfuse (self-hosted alternative to LangSmith)
**Why considered:**
- Open-source, self-hostable — avoids vendor lock-in on traces and datasets.
- Similar feature surface: traces, datasets, evaluators, prompt management.

**Why deferred:**
- LangSmith already integrated and feature-equivalent for this use case.
- The differentiator is hosting model, not capability — running both is observability noise without insight.
- Only justified if production deployment requires data residency or on-prem hosting.

---

## PDF Ingestion

### LlamaParse — *implemented*
**Status:** Default PDF parser when `LLAMA_CLOUD_API_KEY` is set. Falls back to `PyPDFLoader` otherwise.

**Why:**
- Preserves table structure and figure captions that `pypdf` flattens or drops.
- Important for HNSW (algorithm pseudocode), RAGAS (metric tables), vLLM (throughput charts).

**Where:** [backend/ingest/pdf_loader.py](backend/ingest/pdf_loader.py)

---

## Multi-User Backend (accounts, sessions, uploads)

Today the system is single-tenant: one global FAISS+BM25 index, an in-memory or single-project Firestore conversation store, and uploads written to a shared `data/uploads/` directory. Going multi-user means four things need a persistence layer:

1. **Accounts & auth** — sign-up, login, password reset, OAuth, session tokens.
2. **Conversation history** — already partially solved via [backend/rag/conversation_store.py](backend/rag/conversation_store.py); needs scoping by `user_id` instead of `session_id`.
3. **Uploaded documents** — PDF bytes plus parsed chunks, ideally scoped per user so one user's papers don't leak into another's retrieval.
4. **Per-user vector indices** — today FAISS is one in-process index. Multi-tenant needs either per-user indices or a hosted vector DB with metadata filtering.

### Recommendation: Supabase

**Why Supabase over Firestore:**
- **Single console for everything.** Auth, Postgres for relational data, Storage for PDF blobs, and Row-Level Security policies for per-user isolation — all in one product. Firestore needs Firebase Auth + Cloud Storage stitched together, with security rules written in a separate DSL.
- **Postgres beats Firestore for this shape.** A user owning many documents, each having many chunks and citations, is fundamentally relational. Firestore forces denormalisation and makes "list all my documents with their chunk counts" awkward.
- **Storage with signed URLs out of the box.** Uploaded PDFs become `supabase.storage.from('papers').upload(user_id/filename.pdf)` with RLS — no separate IAM config.
- **pgvector ships in Supabase.** The same Postgres instance can hold embeddings, so the FAISS in-process index could move to `pgvector` with `user_id` filtering — solving multi-tenancy and persistence in one move.
- **Generous free tier** (500MB DB, 1GB storage, 50k MAU) covers demo/portfolio scale without a credit card.

**Why not Firestore (despite already being wired in):**
- Existing conversation store is ~80 LOC; the migration cost is small.
- Firestore document size limit (1MB) means chunk arrays for large papers need sub-collection gymnastics.
- Adding Cloud Storage + Firebase Auth + Firestore Security Rules is three products and two configuration languages.

**Why not custom Postgres / Django / etc:**
- For a demo/portfolio, hand-rolling auth and storage is unjustified work. Supabase is "managed Postgres with the boilerplate done."

### Schema sketch (Supabase / Postgres)

```sql
-- auth.users provided by Supabase Auth

create table documents (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users not null,
  filename    text not null,
  doc_type    text not null,
  storage_path text not null,       -- supabase storage key
  chunks      int not null,
  uploaded_at timestamptz default now()
);

create table chunks (
  id          uuid primary key default gen_random_uuid(),
  document_id uuid references documents on delete cascade,
  content     text not null,
  embedding   vector(384),          -- pgvector, MiniLM-L6-v2 dims
  metadata    jsonb
);

create table conversations (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid references auth.users not null,
  title      text,
  created_at timestamptz default now()
);

create table messages (
  id              uuid primary key default gen_random_uuid(),
  conversation_id uuid references conversations on delete cascade,
  role            text not null check (role in ('user', 'assistant')),
  content         text not null,
  citations       jsonb,
  created_at      timestamptz default now()
);

-- Row-Level Security: a user only sees their own rows
alter table documents enable row level security;
create policy "users see own documents" on documents
  for all using (auth.uid() = user_id);
-- repeat for chunks (via documents.user_id), conversations, messages
```

### Migration order

1. Wire Supabase Auth into FastAPI (`Depends(get_current_user)` middleware).
2. Move conversation_store to a Supabase-backed implementation; keep `ConversationStore` interface intact.
3. Add `user_id` to uploaded-document metadata; move blob storage to Supabase Storage.
4. Replace FAISS with `pgvector` queries filtered by `user_id` (this also removes the in-process index limitation flagged under Production Hardening).

**Status:** deferred — v1 is a single-user demo with a curated corpus. The current `ConversationStore` already falls back gracefully between Firestore and in-memory, so the abstraction needed for this change exists.

---

## Production Hardening

- **Authentication:** API key / session-token middleware on FastAPI `/ask`.
- **Rate limiting:** per-session and per-IP throttling (e.g. `slowapi`).
- **Cost tracking:** log per-query token usage to Firestore for billing/quota.
- **Structured logging:** swap `print()` for `structlog`, ship to a log aggregator.
- **Error boundaries:** retry-with-backoff on LLM 429/503; circuit breaker on `web_search` timeouts.
- **Response caching:** Redis cache keyed on `(normalised_query, corpus_version)` for repeat queries.
- **Concurrency:** the FAISS retriever holds a single in-process index — sharding/replication needed for multi-worker uvicorn deployment.

---

## Corpus & Scope

- **General SWE corpus expansion:** add chapters from *Designing Data-Intensive Applications*, the Google SRE Book, key distributed systems papers — broadens the "research assistant" framing beyond AI/ML.
- **Document versioning:** track `ingested_at` per source and surface it in citations so users can spot stale content.
- **Selective re-ingestion:** today BM25 is rebuilt from scratch on every ingest call. Fine at 6 docs, painful at 600.
