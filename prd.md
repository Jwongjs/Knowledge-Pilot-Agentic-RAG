TO-DO
- New feature allow upload research paper at streamlit for user
- Question if finding relevant research paper for the user is needed (whether web search is sufficient to find research paper)

**PRODUCT REQUIREMENTS DOCUMENT**

**Agentic RAG Knowledge Assistant**

"The domain I chose is AI/ML because it makes the failure modes of traditional RAG immediately obvious to an AI engineer. But the architecture is domain-agnostic — the same graph works as a student literature review tool, legal research assistant, or medical guideline synthesizer. You swap the corpus; the agentic loop is unchanged."

May 2026

| **Category**             | **Details**                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| **Document Type**        | Technical PRD - Interview Assessment Response                          |
| **System Name**          | KnowledgePilot - Agentic RAG Assistant                                 |
| **Architecture Pattern** | Tool-Using Agentic RAG: Plan → Act → Evaluate → Iterate (LangGraph)    |
| **Domain**               | Multi-domain Technical Knowledge Base (AI/ML Research + Software Docs) |
| **Core Stack**           | Python · LangGraph · LangChain · FastAPI · Streamlit · LangSmith       |
| **Retrieval Strategy**   | Hybrid Dense+Sparse + Cross-Encoder Reranking                          |
| **Evaluation**           | LangSmith Tracing · Golden Dataset · LLM-as-Judge                      |

# **1\. Executive Summary**

This document specifies the design, architecture, and evaluation strategy for KnowledgePilot - an Agentic RAG (Retrieval-Augmented Generation) system built as a technical response to the AI Engineering interview assignment. The system demonstrates a step-change advancement beyond traditional RAG by enabling the model to reason about its own retrieval, self-correct when retrieved context is insufficient, and execute multi-hop queries through an explicit LangGraph-orchestrated state machine.

KnowledgePilot targets a multi-domain technical knowledge base (AI/ML research papers, software documentation, technical guides), a corpus chosen specifically to expose the failure modes of single-pass retrieval and justify every agentic design decision. The system is delivered as a working Streamlit prototype with full LangSmith observability and a structured evaluation framework.

## **1.1 What This System Demonstrates**

| **Capability**                | **How It Is Demonstrated**                                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agentic Retrieval Loop**    | LangGraph graph routes conversation queries to a direct response, simple queries to the Planner, and multi-hop queries through the Decomposer — each path is a distinct, intentional routing decision |
| **Query Decomposition**       | Complex multi-hop questions are broken into sub-queries by a dedicated Decomposer node; each sub-query retrieves independently before final synthesis         |
| **Hybrid Retrieval**          | Dense vector search (FAISS/Chroma) combined with BM25 sparse retrieval via LangChain EnsembleRetriever; weighted fusion returns best-of-both results          |
| **Cross-Encoder Reranking**   | Top-K hybrid results re-scored by a sentence-transformers cross-encoder, elevating semantically precise chunks over vector-distance approximations            |
| **Citations with Provenance** | Every synthesized answer carries inline citations (\[¹\], \[²\]) linked to specific source chunks with page number, score, and which sub-query generated them |
| **LangSmith Observability**   | Full graph execution traced to LangSmith: retrieval scores, node transitions, LLM decisions, latency per step, and token counts all captured                  |
| **Automated Evaluation**      | LLM-as-Judge harness using RAGAS metrics (Faithfulness, Context Precision, Answer Relevancy) run against a golden dataset of 25 curated test cases           |
| **Tool-Using Agent**          | Planner node selects from a two-tool action space (vector retrieval, web search) per query/sub-query; tool choice is justified, traceable, and citation-attributed by source_type |

# **2\. Domain & Use Cases**

The interview brief did not mandate a domain, so KnowledgePilot is built as a **general-purpose, domain-agnostic research assistant** that exercises every component of the modern agentic-RAG stack:

- **Hybrid retrieval** — FAISS dense + BM25 sparse, fused via LangChain `EnsembleRetriever` (RRF, 0.6/0.4)
- **Cross-encoder reranking** — top-15 candidates → top-5 via `ms-marco-MiniLM-L-6-v2`
- **Tool-using planner** — agent chooses `vector_retriever` (uploaded corpus) vs `web_search` (Tavily) per sub-query
- **LangGraph orchestration** — 5-node state machine: analyzer → decomposer → planner → executor → synthesizer
- **Adaptive routing** — conversation / simple / complex / multi-hop classified at intake
- **Query decomposition** — multi-hop questions split into independent sub-queries before retrieval
- **Inline citations with provenance** — `[¹]` markers mapped back to chunk + source + originating sub-query
- **Hallucination guard** — three-check validator (citation validity, factual coverage, token alignment)
- **LangSmith tracing** — every node decision, retrieval score, and LLM call captured
- **Golden-dataset evaluation** — 25 curated cases × 4 evaluators (faithfulness, citation accuracy, tool selection, re-retrieval rate)

## **2.1 Advanced Research Use Cases**

The AI/ML papers corpus shipped with the demo is **illustrative, not architectural** — swap the PDFs and the same graph supports advanced research in any domain:

- **AI/ML engineering** *(current corpus):* *"Compare the chunking strategy in the RAGAS paper with LangChain's RecursiveCharacterTextSplitter, and tell me what the latest LangGraph release notes say about checkpointing."*
- **Legal research:** *"Which precedents cite Roe v. Wade and were later overturned by the same circuit?"* — upload case briefs + statutes
- **Clinical decision support:** *"Compare first-line hypertension treatments across the 2023 ESC and 2024 ACC guidelines — flag any conflicts."* — upload guideline PDFs
- **Equity research:** *"How has Company X's gross margin trajectory changed across 2020-2024 earnings calls, and what does the latest analyst coverage say?"* — upload 10-Ks + transcripts
- **Academic literature review:** *"Identify methodological gaps in the cited works of Paper X's literature review, and find recent papers that address them."*
- **PRD / product research** *(the personal use case this was originally built for):* *"What does current research recommend for chunking RAG corpora, and what does the most recent LangChain documentation default to?"* — papers in the corpus, framework specifics via web search

In every case the architecture is unchanged: the agent decides *if*, *when*, and *where* to search; ensembles dense + sparse retrieval over the corpus; reranks; cites; self-validates.

## **2.2 Why This Corpus for the Demo**

The AI/ML research papers were chosen as the demo corpus because:

- **Public and reproducible** — all six papers are open-access arxiv PDFs
- **Genuine cross-paper relationships** — papers cite each other, making multi-hop queries authentic rather than contrived
- **Failure-mode visibility** — an AI/ML-fluent audience can verify retrieval and synthesis on the spot, since they already know the source material
- **Mixed-source coverage** — the locally-indexed papers pair naturally with live-web LangChain / LangGraph / vLLM docs, exercising both `vector_retriever` and `web_search` in the same query (T18, T21, T22 in the eval set)

Corpus included: RAGAS (arxiv 2309.09217), Attention is All You Need (1706.03762), RAG — Lewis et al. (2005.11401), HNSW (1603.09320), LoRA (2106.09685), vLLM (2309.06180).

# **3\. Traditional RAG vs. Agentic RAG**

Understanding the architectural delta between traditional and agentic RAG is the conceptual foundation of this system. This section maps the comparison concretely and positions each agentic upgrade against a specific limitation of the pipeline built in the prior Residex/DocuMind project.

## **3.1 Side-by-Side Architecture Comparison**

| **Dimension**          | **Traditional RAG (DocuMind)**                                   | **Agentic RAG (KnowledgePilot)**                                                                                        |
| ---------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Retrieval passes**   | Single pass per query - fixed                                    | Single pass; Planner selects tools per (sub-)query; multi-hop queries decompose before retrieval                        |
| **Query handling**     | Query embedded as-is                                             | Complexity classifier → optional decomposition into sub-queries                                                         |
| **Retrieval strategy** | Dense vector search (Firestore Vector Search) only               | Hybrid: Dense (FAISS) + Sparse (BM25) → EnsembleRetriever → Cross-Encoder Reranker                                      |
| **Action space**       | Single action: query the vector DB                                | Two tools: vector retrieval (FAISS+BM25+reranker), web search (Tavily). Planner selects tool(s) per (sub-)query; agent decides *if*, *when*, and *where* to search |
| **Self-correction**    | None - first retrieved chunks used regardless of quality         | Hallucination Guard validates every answer against retrieved Evidence; query decomposition handles multi-hop gaps        |
| **Orchestration**      | LangGraph with deterministic intent gate and category checkpoint | LangGraph with adaptive routing: simple path vs. decomposition path vs. iterative retrieval loop                        |
| **Citations**          | Citation model per chunk (doc_id, filename, page, snippet)       | Multi-hop provenance: inline citations (\[¹\]) mapped to specific sub-query + chunk, with confidence score per citation |
| **Observability**      | Print statements / Firestore session logs                        | LangSmith full-graph tracing: every node, LLM call, retrieval score, and latency captured                               |
| **Evaluation**         | Manual 12-case golden dataset, human grading                     | Automated: RAGAS metrics (Faithfulness, Context Precision, Answer Relevancy) with LLM-as-Judge; 25 golden cases including tool-selection, adversarial, and citation accuracy tests |

## **3.2 Why Agentic RAG - The Honest Trade-offs**

**Agentic RAG is not universally better - it is better when queries are complex and variable. The system routes simple queries through the fast path to avoid unnecessary overhead.**

| **Agentic RAG Advantages**                                                 | **Agentic RAG Trade-offs**                                                  |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Handles multi-hop queries that single-pass fails                           | Higher latency for complex queries (2-4x single-pass)                       |
| Self-corrects on poor retrieval - higher answer quality for hard questions | More complex graph to debug and maintain                                    |
| Adaptive: routes simple queries through a fast path (near-zero overhead)   | Token usage scales with iterations - higher LLM cost per complex query      |
| Hybrid retrieval captures both semantic and keyword relevance              | Reranker adds ~100-300ms latency (mitigated by async execution)             |
| Full observability via LangSmith - every decision is inspectable           | LangSmith requires API key and adds tracing overhead (~10ms per trace call) |

# **4\. System Architecture**

## **4.1 High-Level Architecture**

KnowledgePilot is implemented as a LangGraph StateGraph that realises the agentic loop **Query → Analyze → (Decompose) → Plan → Act → Generate**. It has five functional nodes and an adaptive router with four classification outcomes. Conversation queries are handled directly by the Query Analyzer; simple queries go to the Planner; complex/multi-hop queries pass through the Decomposer first.

| **KnowledgePilot - LangGraph Execution Flow (Query → Plan → Act → Evaluate → Iterate → Generate)** |
| --- |
| **USER QUERY** *(Query)*<br><br>↓<br><br>**\[Node 1\] Query Analyzer** \| Classify: conversation / simple / complex / multi-hop \| Extract entities \| Detect temporal/out-of-corpus signals \| Conservative fallback to *simple* when uncertain<br><br>↓ conversation ↓ simple ↓ complex / multi-hop<br><br>**\[Node 1.5\] Conversation Responder** **\[Skip Decomposer\]** **\[Node 2\] Query Decomposer**<br><br>Hybrid template+LLM reply Direct to Planner Break into 2-3 atomic sub-queries (LLM) + Sanity gate<br><br>↓ END ↓ ↓ (per sub-query)<br><br>**\[Node 2.5\] Planner** *(Plan)* \| For each (sub-)query, select tool(s) from action space: `vector_retriever` \| `web_search` \| `arxiv_search` \| Output ordered ToolCall plan with rationale<br><br>↓<br><br>**\[Node 3\] Action Executor** *(Act)* \| Dispatch each ToolCall \| `vector_retriever`: Dense FAISS + BM25 → EnsembleRetriever → Cross-Encoder Reranker \| `web_search`: Tavily API → top-5 results \| `arxiv_search`: arxiv API → abstract + metadata \| Normalise all results into Evidence objects<br><br>↓<br><br>**\[Node 4\] Relevance Grader** *(Evaluate)* \| LLM grades each Evidence item: RELEVANT / PARTIAL / IRRELEVANT \| Counts relevant evidence per sub-query<br><br>↓ sufficient context ↓ insufficient *(Iterate, iter < max_iter)*<br><br>\[Node 5\] Answer Synthesizer Re-Plan: Planner revises tool choice (e.g. vector → web_search) and/or refines query, loops to Node 3<br><br>*(Generate)* Answer with inline citations (¹²³) attributed by source_type<br><br>↓<br><br>**\[Node 6\] Hallucination Guard** \| Deterministic citation-grounding validator (coverage / validity / alignment) \| Optional LLM verifier on risk signals \| **Bypassed on conversation path**<br><br>↓<br><br>**FINAL RESPONSE + CITATIONS (multi-source) + LANGSMITH TRACE** |

## **4.2 LangGraph State Schema**

The AgentState TypedDict persists all information across node transitions:

| **Field**              | **Type**                              | **Purpose**                                                  |
| ---------------------- | ------------------------------------- | ------------------------------------------------------------ |
| **query**              | str                                   | Original user query, immutable through the graph             |
| **query_complexity**   | Literal\[conversation, simple, complex, multi_hop\] | Set by Node 1; routes to Conversation Responder (conversation), Planner directly (simple), or Decomposer (complex/multi-hop) |
| **sub_queries**        | List\[str\]                           | Populated by Query Decomposer; empty list for simple queries |
| **action_plan**        | List\[ToolCall\]                      | Set by Planner; each ToolCall = (tool_name, args, sub_query, rationale). Re-written on Iterate |
| **tool_results**       | Dict\[str, List\[Evidence\]\]         | Evidence accumulated by source_type (vector_chunk / web); deduped by content hash          |
| **iteration_count**    | int                                   | Guards against infinite plan-act-evaluate loops (max_iterations=3) |
| **answer**             | str                                   | Synthesized answer with inline citation markers              |
| **citations**          | List\[Citation\]                      | Rich citation objects linked to specific Evidence items, with source_type |
| **hallucination_flag** | bool                                  | Set by Hallucination Guard if answer claims outside Evidence |

**Evidence object** (unified across tools): `{evidence_id, source_type, content, metadata, dense_score?, rerank_score?, url?, page?}`. The Action Executor wraps every tool's raw output in this shape so downstream nodes (Grader, Synthesizer, Guard) are tool-agnostic.

## **4.3 Node Specifications**

### **Node 1 - Query Analyzer**

Uses an LLM with a structured JSON output schema to classify every incoming query into one of four labels. Outputs query_complexity and a list of key entities. Classification is deliberately conservative - **conversation** requires explicit social/meta signals; otherwise the Analyzer falls back to **simple** rather than risk refusing a real factual question. Only **multi-hop** triggers decomposition; **complex** uses a single retrieval with a higher top-K. This prevents over-decomposition of within-document multi-concept queries.

- Conversation: Pure social input or meta-questions about the assistant itself - no retrieval needed. Triggers: greeting/farewell lexicon ('hi', 'hello', 'hey', 'thanks', 'bye', 'good morning'); meta-questions ('who are you', 'what can you do', 'how do you work'); social text ('how's it going', 'nice to meet you'). **Hard requirement**: zero domain entities AND zero factual question-words ('what is', 'how does', 'define', 'explain', 'compare'). If unsure, fall back to *simple* - a wasted retrieval is cheaper than refusing to answer a real question. Routes to Node 1.5 Conversation Responder; bypasses retrieval and Hallucination Guard entirely.
- Simple: Single-concept factual question answerable from one chunk; zero comparison conjunctions ('and', 'vs', 'compare', 'differ', 'relate'). Example: 'What is cosine similarity?' Routes directly to the Planner (skips Decomposer).
- Complex: Multi-concept question whose concepts plausibly co-occur in the same document or sub-domain. Example: 'What is HNSW and how does FAISS implement it?' - both concepts live in the same doc neighbourhood, so a single retrieval with top-K=8 suffices.
- Multi-hop: Explicit cross-source signal required - comparison verbs ('compare', 'differ', 'versus') AND named entities from different sub-corpora, or phrasing like 'according to X and Y'. Example: 'How does the chunking in the RAGAS paper differ from the LangChain default?'

### **Node 1.5 - Conversation Responder (conversation only)**

> **Implementation note:** In the shipped v1, this node's logic is merged into Node 1 (Query Analyzer). The Analyzer detects conversation signals, sets `answer` directly in state, and the graph routes to END — bypassing retrieval entirely. The design below describes the intended behaviour, which is fully preserved.

Short-circuits the retrieval and synthesis pipeline for pure social interaction, greeting, and meta-questions about the assistant. Implemented as a hybrid strategy to balance cost and naturalness:

**Hybrid template + LLM approach:**

- **Greetings/Farewells** (lexicon match: 'hi', 'hello', 'hey', 'thanks', 'bye', 'good morning', 'goodbye', 'see you'): Respond with a lightweight canned template + one example query
  - Example: 'Hi! I can help you with questions about AI/ML papers, LangChain docs, and RAG best practices. Try asking: "What is cosine similarity?" or "How does hybrid retrieval work?"'
  - Cost: ~0ms, zero hallucination risk

- **Meta-questions** (lexicon match: 'who are you', 'what can you do', 'how do you work', 'what are you', 'tell me about yourself'): Respond with a single LLM call with a tight, scope-bounded system prompt
  - System instruction: 'You are a helpful assistant. Answer only about your actual capabilities: answering factual questions about AI/ML papers, LangChain/LangGraph/FAISS documentation, and RAG best practices. Do not hallucinate capabilities or promise to do things outside this scope. Be concise (2-3 sentences).'
  - This prevents the common failure mode where the assistant over-promises capabilities
  - Cost: 1 LLM call (~0.5s), grounds response in actual system capabilities

**Output:** Plain text response with no citations, no retrieval, no Hallucination Guard check.

**Design rationale:** Conversation detection fires on explicit signals only; falling back to *simple* is preferred when uncertain. This node exists to avoid wasting retrieval + ranking on pure social input, which is a real-world failure mode ("here are some documents about greetings"). The hybrid approach avoids a full LLM call on high-volume greeting input while preserving naturalness on meta-questions where the LLM's response reflects actual capabilities.

### **Node 2 - Query Decomposer (complex / multi-hop only)**

**Fires only when query_complexity is 'complex' or 'multi-hop'.** Skipped entirely for 'conversation' (handled by Node 1.5) and 'simple' (handled by Planner directly).

Prompts an LLM to break the original query into 2-3 atomic sub-queries (hard cap at 3). Each sub-query is independently retrievable and collectively sufficient to answer the original. Design constraint: sub-queries must be non-overlapping to avoid duplicate chunk retrieval.

**Decomposition Sanity Gate.** After generation, sub-queries pass through a deterministic validation step before retrieval fires:

- If only 1 sub-query is produced → skip decomposition, fall back to single-retrieval path
- If pairwise token overlap between any two sub-queries exceeds 70% (Jaccard on lowercased tokens minus stopwords) → reject decomposition, fall back to single retrieval
- If sub-queries collectively drop a named entity present in the original query → log warning and append the entity to the most relevant sub-query

This gate prevents the highest-cost failure mode: paying 3x retrieval latency for sub-queries that retrieve overlapping chunks.

- Example: 'How does the chunking strategy in the RAG paper compare to LangChain's default?' decomposes into:
  - 'What chunking strategy does the RAGAS paper recommend?'
  - 'What is LangChain RecursiveCharacterTextSplitter default configuration?'
  - 'How do fixed vs. recursive chunking strategies affect retrieval recall?'

### **Node 2.5 - Planner (the *Plan* stage of the agent loop)**

The Planner is what makes this system *agentic* in the broad sense - it decides **if**, **when**, and **where** to search. For each (sub-)query it selects an ordered set of tool calls from the action space, with a written rationale per call. The Planner is an LLM with a structured JSON output schema and access to tool descriptions but not tool implementations.

**Action space (initial, extensible):**

| **Tool** | **When to select** | **Returns** |
| -------- | ------------------ | ----------- |
| **vector_retriever** | Default for in-corpus knowledge: AI/ML papers, LangChain/LangGraph/FAISS docs, curated FAQs. Always tried first unless a hard out-of-corpus signal is present. | Top-K reranked chunks from FAISS+BM25+CrossEncoder pipeline |
| **web_search** (Tavily) | Temporal markers ('latest', 'recent', '2026', 'newest', 'current'); query references entities not in the local corpus; or paper title not present in indexed corpus | Top-5 web results: title, URL, snippet, published_date |

**Selection policy (deterministic guards on top of LLM choice):**

- Default action for any (sub-)query is `vector_retriever` - LLM must justify deviating
- `web_search` is gated behind: temporal regex match OR Analyzer flagged `out_of_corpus=true` OR paper title not found in the corpus manifest shown to the Planner
- Hard cap: ≤2 tool calls per sub-query to bound cost

**Planner output schema (per ToolCall):** `{tool_name, args, sub_query, rationale, expected_evidence_type}`. The rationale is logged to LangSmith so tool selection is fully inspectable - if a wrong tool fires, the trace shows *why*.

**First-pass tool selection:** the Planner receives a manifest of all locally-indexed paper titles so it can route to `web_search` on the first pass when the query is clearly out-of-corpus, rather than requiring a grader feedback loop. This is what makes the single-pass graph accurate on T18/T21/T22 without needing a re-plan iteration.

### **Node 3 - Action Executor (the *Act* stage; was Hybrid Retriever)**

The Action Executor dispatches each ToolCall in the plan, normalises raw tool output into a unified Evidence object, and accumulates results into `tool_results` keyed by source_type. Tool implementations:

**vector_retriever** (the primary retrieval optimization layer, four stages):

- Stage 1 - Dense Retrieval: Query embedding via sentence-transformers/all-MiniLM-L6-v2 (384-dim), FAISS IndexFlatIP for cosine similarity, top-20 candidates
- Stage 2 - Sparse Retrieval: BM25Retriever (LangChain) over the same corpus, top-20 candidates. BM25 excels at exact keyword matches (function names, parameter names, paper titles)
- Stage 3 - Ensemble Fusion: EnsembleRetriever with weights \[0.6 dense, 0.4 sparse\], Reciprocal Rank Fusion (RRF) combining ranked lists from both retrievers, returning top-15
- Stage 4 - Cross-Encoder Reranking: sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2 scores all 15 candidates against the query; top-5 selected by cross-encoder score (not vector distance)

**Why cross-encoder reranking? Bi-encoder vector similarity is an approximation - it compares query and document independently. A cross-encoder reads both jointly, producing a true relevance score at the cost of latency. Applied only on the top-15 pre-filtered candidates, it adds ~200ms while lifting Precision@1 by 8-15% in practice.**

**web_search**: Tavily Search API (`search_depth=advanced`, `max_results=5`). Each result is wrapped as Evidence with `source_type='web'`, `url`, `published_date`, and `content=snippet`. No bi-encoder/rerank scores apply.

**Cross-tool deduplication**: Evidence items are hashed on `(source_type, normalised_content[:200])` to prevent duplicate evidence when web and vector return overlapping content (e.g. arxiv abstract vs. local paper chunk).

**Failure handling**: if a tool errors (network timeout, API quota), the Executor logs the failure to LangSmith and the Planner is re-invoked to choose an alternative - the agent does not crash on a single tool failure.

### **Node 4 - Relevance Grader (the *Evaluate* stage)**

> **Design decision — removed in v1:** After profiling, this node generated 3-5 LLM calls per query and was the main driver of multi-hop query latency (~25-40s). The grader's verdict also had low marginal impact on final answer quality when CrossEncoder reranking was already selecting high-relevance chunks. Removed and replaced with: (1) CrossEncoder rerank score as an implicit relevance signal inside the retriever, (2) the Hallucination Guard's citation alignment check as the quality gate, and (3) a Planner prompt that includes the corpus manifest so tool selection is right on the first pass.

Original design for reference:

For each Evidence item (regardless of source_type), an LLM is prompted with: 'Given the query \[Q\], rate this evidence \[E\] as: RELEVANT, PARTIAL, or IRRELEVANT.' Evidence count is aggregated across all tools. If fewer than 3 RELEVANT items are found and iteration_count < max_iterations:

- Verdict is set to `insufficient` and control returns to the **Planner** (not Node 3 directly)
- The Planner receives the prior plan, the prior verdict, and which tools have been tried, and is prompted to *change strategy* - typically switch tools (e.g. add `web_search` after vector_retriever produced thin evidence) or refine the query
- New Evidence is merged with existing relevant Evidence (deduplication by content hash)
- iteration_count is incremented; the loop terminates at max_iterations=3 even if still insufficient (Synthesizer then produces a low-confidence answer)

### **Node 5 - Answer Synthesizer**

Constructs the final prompt with: (1) system instructions for citation-grounded answering, (2) numbered context blocks for each relevant chunk, (3) citation instruction: reference chunks inline as \[¹\], \[²\], etc. Output is plain text with inline citation markers.

### **Node 6 - Hallucination Guard**

**Skipped on conversation path** - no retrieval occurred, so citation grounding checks are inapplicable. Guard fires only after synthesis on complex/simple/multi-hop queries.

Post-generation safety check, implemented as a **deterministic citation-grounding validator** rather than a separate LLM call. Runtime LLM verifiers double answer latency and largely re-check what a well-prompted Synthesizer already enforces; a deterministic check catches the highest-frequency failure mode (uncited or mis-cited claims) at near-zero cost. Deeper semantic faithfulness is measured offline via RAGAS Faithfulness on the golden dataset (see §7).

The guard validates three properties:

- **Citation coverage**: every sentence containing a factual claim (heuristic: contains a number, named entity, or technical term) must include at least one citation marker
- **Citation validity**: every inline marker `[ⁿ]` in the answer must reference a chunk_id that was actually present in retrieved_chunks
- **Citation alignment**: each cited chunk's snippet must share ≥1 content token (excluding stopwords) with the sentence it supports

If any check fails, hallucination_flag is set true and the system returns a degraded response noting low confidence rather than a potentially wrong answer. The LLM-based verifier is retained as an optional fallback that fires only when relevance_verdict was 'insufficient' at the final iteration - i.e. defense in depth on risk signals, not on every query.

## **4.4 Reference Architecture - Patterns from DocuMind**

The DocuMind system (Residex's property document Q&A feature) established a working LangGraph orchestration pattern that KnowledgePilot directly builds upon. The design decisions below are proven in production and carried forward.

### **Component Relationship Map**

```
DocuMindService  (facade / API boundary)
│
│  creates & injects
├─► ConversationRouter        (LLM node: conversation vs. RAG intent)
├─► CategoryPredictor         (LLM node: which document category to search)
│
│  injects both nodes into
└─► DocuMindGraphOrchestrator (LangGraph StateGraph)
         │
         │  compiled graph nodes
         ├─ route_conversation    → calls ConversationRouter.route()
         ├─ respond_conversation  → emits action=conversation
         ├─ predict_categories    → calls CategoryPredictor.predict()
         ├─ decide_action         → deterministic logic (explicit / confirm / cancel / predict)
         ├─ prepare_confirmation  → emits action=ask_confirmation
         ├─ prepare_cancel        → emits action=cancel
         └─ prepare_retrieve      → emits action=retrieve (passes through)

ConversationStore  (Firestore-backed session memory, used by Service only, not injected into graph)
    ├─ get_or_create_session()
    ├─ append_turn()
    ├─ set_pending_confirmation() / get_pending_confirmation() / clear_pending_confirmation()
    └─ in-memory cache layer over Firestore reads
```

### **Key Design Patterns (Carry Forward into KnowledgePilot)**

| **Pattern** | **DocuMind Implementation** | **How It Applies to KnowledgePilot** |
| --- | --- | --- |
| **Dependency injection into graph** | `DocuMindGraphOrchestrator(conversation_router=..., category_predictor=...)` | Each LangGraph node is a class injected into the orchestrator — nodes are testable in isolation without wiring the full graph |
| **TypedDict state schema** | `DocuMindState(TypedDict, total=False)` carries all fields across node transitions | `AgentState` TypedDict (§4.2) is the same pattern; every node reads from and writes to the shared typed state dict |
| **Graph owns routing, Service owns retrieval** | Graph emits `action=retrieve` or `action=conversation`; Service executes the actual Firestore vector query and LLM synthesis after graph returns | Graph (nodes 1-6) decides *what to do* and produces Evidence; Service owns indexing, persistence, and the final API response shape |
| **Checkpoint / confirmation pattern** | Graph emits `action=ask_confirmation`; Service persists pending state in `ConversationStore.set_pending_confirmation()`; next request resolves it via `user_action=confirm/cancel/override` | Relevance Grader → re-plan loop is the agentic equivalent: graph state holds iteration context across the plan-act-evaluate cycle without external state |
| **Two-layer routing** | Layer 1: `ConversationRouter` (conversation vs. document intent) → Layer 2: `CategoryPredictor` (which categories to search) | Layer 1: `QueryAnalyzer` (conversation / simple / complex / multi-hop) → Layer 2: `Planner` (which tools to call, which sub-queries to run) |
| **LLM node + keyword fallback** | Both `ConversationRouter` and `CategoryPredictor` have `except Exception` blocks with keyword-based fallback logic | Every LLM node should define deterministic fallback behaviour so a single LLM failure does not crash the entire graph |
| **Async graph execution** | `await self._graph.ainvoke(state)` — graph compiled and stored at init; reused per request | Same pattern: graph compiled once at service init, `ainvoke` per request |
| **Session memory decoupled from graph** | `ConversationStore` is instantiated in `DocuMindService.__init__`, not inside the graph; graph is stateless between requests | `ConversationStore` (or equivalent LangSmith/LangGraph Checkpointer) lives in the service layer, not inside node implementations |

### **Structural Lessons from DocuMind**

**What worked well:**
- Compiling the graph once at service init and reusing it avoids re-building the StateGraph on every request — critical for latency
- Keeping node classes (ConversationRouter, CategoryPredictor) independent of the StateGraph means they can be unit-tested with a mock LLM without spinning up the full graph
- The `DocuMindState` TypedDict with `total=False` allows nodes to incrementally add fields without breaking earlier nodes that don't set them
- Structured string parsing (`intent=...;rag_needed=...;confidence=...`) as LLM output format is simpler and more robust than JSON for small structured outputs in lightweight nodes

**What KnowledgePilot improves:**
- DocuMind nodes communicate via free-text LLM responses parsed with string splitting; KnowledgePilot nodes (Analyzer, Planner, Grader) use structured JSON output schema via `with_structured_output()` for stronger contracts
- DocuMind has no iteration loop — the graph terminates after one routing decision; KnowledgePilot adds the plan-act-evaluate-iterate cycle with `iteration_count` guard in AgentState
- DocuMind's ConversationStore uses Firestore directly; KnowledgePilot uses LangSmith as the persistence + observability layer

---

### **Proposed File Structure**

LangGraph node implementations are moved into a dedicated `nodes/` subfolder under `rag/`, mirroring the pattern where each node is a standalone, injectable class. The service and orchestrator remain at the `rag/` level.

```
backend/
├── rag/
│   ├── nodes/                        ← LangGraph node implementations (one class per file)
│   │   ├── __init__.py
│   │   ├── query_analyzer.py         ← Node 1: classifies query; handles conversation replies inline
│   │   ├── query_decomposer.py       ← Node 2: breaks multi-hop into atomic sub-queries
│   │   ├── planner.py                ← Node 3: selects tool(s) per sub-query with rationale
│   │   ├── action_executor.py        ← Node 4: dispatches ToolCalls, normalises to Evidence
│   │   ├── answer_synthesizer.py     ← Node 5: generates answer with inline citations + hallucination check
│   │   └── hallucination_guard.py    ← citation-grounding validator (called inline by synthesizer)
│   │
│   ├── retrievers/                   ← Tool implementations called by action_executor
│   │   ├── __init__.py
│   │   ├── vector_retriever.py       ← FAISS + BM25 EnsembleRetriever + CrossEncoder reranker
│   │   └── web_search.py             ← Tavily Search API wrapper → Evidence normaliser
│   │
│   ├── graph_orchestrator.py         ← DocuMindGraphOrchestrator equivalent: wires nodes into StateGraph
│   ├── knowledge_pilot_service.py    ← Top-level service: ingestion, indexing, ask() API handler
│   └── conversation_store.py         ← Session memory (carried over from DocuMind unchanged)
│
├── models/
│   ├── agent_state.py                ← AgentState TypedDict + Evidence + Citation + ToolCall dataclasses
│   └── api_models.py                 ← FastAPI request/response schemas (AskRequest, AskResponse)
│
├── ingest/
│   ├── pdf_loader.py                 ← PyPDFLoader + per-type chunking strategy
│   ├── docx_loader.py                ← python-docx loader → same chunk pipeline as PDF
│   └── index_builder.py             ← Builds + persists FAISS index and BM25 index; routes .pdf and .docx
│   (html_loader.py removed — LangChain/LangGraph/FAISS docs retrieved live via web_search tool)
│
├── evaluation/
│   ├── golden_dataset.py             ← 25 test cases as LangSmith Dataset
│   ├── evaluators.py                 ← faithfulness_evaluator, citation_accuracy_evaluator, tool_selection_evaluator
│   └── run_eval.py                   ← CLI runner: loads dataset, runs chain, uploads scores to LangSmith
│
├── main.py                           ← FastAPI app: /ask, /ingest, /documents endpoints
└── requirements.txt
```

**Node class contract** (same pattern as DocuMind's ConversationRouter and CategoryPredictor):

```python
# Each node in rag/nodes/ follows this interface:
class QueryAnalyzer:
    def __init__(self, llm):          # LLM injected at service init
        self._llm = llm

    async def run(self, state: AgentState) -> AgentState:
        # reads from state, returns updated state dict
        ...

# Injected into orchestrator at construction time:
class KnowledgePilotGraphOrchestrator:
    def __init__(self, query_analyzer, conversation_responder, planner, ...):
        self._query_analyzer = query_analyzer
        ...
        self._graph = self._build_graph()   # compiled once at init, reused per request
```

# **5\. Citation Handling Design**

Citations are a first-class feature and a bonus requirement. KnowledgePilot's citation design builds on the DocuMind citation model and adds multi-hop provenance tracking: each citation is traceable back to the specific sub-query that generated its retrieval, the retrieval scores from both bi-encoder and cross-encoder stages, and the chunk's exact position within the source document.

## **5.1 Citation Data Model**

| **Field**            | **Type**    | **Description**                                                      |
| -------------------- | ----------- | -------------------------------------------------------------------- |
| **citation_id**      | str         | Inline marker index (1, 2, 3 … matched to \[¹\] in answer text)      |
| **source_type**      | Literal\[vector_chunk, web\] | Which tool produced the underlying Evidence; drives citation rendering and trust badges |
| **evidence_id**      | str         | Unique identifier for the specific Evidence item retrieved (chunk_id, URL hash, or arxiv_id) |
| **source_document**  | str         | Filename, web page title, or paper title (e.g. 'ragas_paper.pdf', 'OpenAI Blog: o5 Reasoning') |
| **url**              | str \| None | Web URL or arxiv link; None for local vector_chunk citations         |
| **page_number**      | int \| None | Page within source document; None for web/arxiv sources              |
| **section_title**    | str \| None | Nearest section heading above the chunk (if extractable)             |
| **snippet**          | str         | 80-120 character excerpt - the most relevant sentence                |
| **dense_score**      | float \| None | Bi-encoder cosine similarity from FAISS (0-1); None for web/arxiv  |
| **rerank_score**     | float \| None | Cross-encoder relevance score; None for web/arxiv (no rerank applied) |
| **published_date**   | str \| None | For web/arxiv sources only - critical for temporal queries          |
| **sub_query_origin** | str \| None | Which sub-query caused this Evidence to be retrieved (for multi-hop) |
| **tool_rationale**   | str         | Why the Planner selected this tool for this sub-query (for inspectability) |
| **iteration**        | int         | Which plan-act-evaluate iteration produced this Evidence (1, 2, or 3) |

## **5.2 Inline Citation Format**

Answer text uses numeric superscript markers generated by the Synthesizer prompt instruction. Each marker links to a citation panel rendered below the answer:

**Example answer output with citations:**

The RAGAS framework evaluates RAG pipelines using four metrics: Faithfulness, Answer Relevancy, Context Precision, and Context Recall \[¹\]. Faithfulness is defined as the proportion of claims in the generated answer that are supported by the retrieved context \[¹\], while Answer Relevancy measures whether the response addresses the original question \[²\].

── Citations ────────────────────────────────────────────

\[¹\] ragas_paper.pdf · p.3 · Section: 'Evaluation Metrics' · Score: 0.92

Sub-query: 'What metrics does RAGAS define?' · Iteration 1

Snippet: '...faithfulness is computed as the number of supported claims...'

\[²\] ragas_paper.pdf · p.4 · Section: 'Answer Relevancy' · Score: 0.88

Sub-query: 'How is Answer Relevancy computed in RAGAS?' · Iteration 1

## **5.3 Citation Quality vs. DocuMind**

DocuMind citations (Citation model: doc_id, filename, category, page, snippet, score) were a strong foundation. KnowledgePilot advances them in three ways:

- Multi-hop provenance: sub_query_origin field traces each citation to the specific sub-query that surfaced it - critical for multi-hop answers where different parts of the answer come from different retrieval passes
- Dual scoring: both dense_score and rerank_score are exposed, making it inspectable whether a chunk ranked highly because of semantic similarity or cross-encoder relevance - useful for debugging and trust calibration
- Inline integration: citation markers are generated inside the answer text by the LLM (not appended post-hoc), enabling per-claim attribution rather than per-response attribution

# **6\. Retrieval Optimisation**

## **6.1 Hybrid Search Architecture**

The core retrieval upgrade over traditional single-mode dense search is the EnsembleRetriever combining dense and sparse signals. The motivation for each component:

| **Component**                                        | **Strength**                                                                        | **Weakness (compensated by other)**                                                                                                        |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Dense FAISS (semantic vector search)**             | Captures meaning and paraphrase; finds relevant chunks even when exact terms differ | Misses exact technical terms (class names, error codes, parameter names) - 'chunk_overlap' might rank lower than semantically similar text |
| **Sparse BM25 (keyword matching)**                   | Exact term matching; first-pass recall for technical jargon and precise queries     | Ignores semantic meaning; 'batch size' and 'mini-batch size' are unrelated terms to BM25                                                   |
| **Cross-Encoder Reranker (joint relevance scoring)** | Reads query and chunk jointly - true relevance, not distance approximation          | Slow: cannot run over full corpus; only applied to top-15 from fusion                                                                      |

## **6.2 Reciprocal Rank Fusion (RRF)**

EnsembleRetriever combines ranked lists from dense and sparse retrievers using RRF. For each document d across both ranked lists:

**RRF_score(d) = Σ weight_i / (k + rank_i(d))**

where k=60 (smoothing constant), rank_i(d) is d's rank in retriever i, weight_i ∈ {0.6 dense, 0.4 sparse}

Documents appearing in both ranked lists accumulate score from both - a natural document-level AND gate

## **6.3 Chunking Strategy**

Chunking strategy is a first-order retrieval quality variable. KnowledgePilot uses RecursiveCharacterTextSplitter with parameters tuned per document type:

| **Document Type**                  | **chunk_size** | **chunk_overlap** | **Rationale**                                                                |
| ---------------------------------- | -------------- | ----------------- | ---------------------------------------------------------------------------- |
| **Research papers (PDF)**          | 1000 tokens    | 150 tokens        | Dense academic text; larger chunks preserve methodological context           |
| **Technical documentation (HTML)** | 600 tokens     | 100 tokens        | Sections tend to be modular; smaller chunks improve precision for API lookup |
| **FAQ / How-To guides**            | 400 tokens     | 80 tokens         | Q&A pairs are naturally atomic; tight chunks = precise retrieval             |

## **6.4 Performance Targets**

| **Metric**                   | **Baseline (dense only)** | **Target (hybrid + rerank)** | **Measurement Method**                  |
| ---------------------------- | ------------------------- | ---------------------------- | --------------------------------------- |
| **Precision@1**              | ~70%                      | **\>85%**                    | Golden dataset ground truth comparison  |
| **Precision@3**              | ~88%                      | **\>95%**                    | Golden dataset, correct doc in top-3    |
| **RAGAS Faithfulness**       | N/A (new metric)          | **\>0.85**                   | LLM-as-judge on 23 test cases           |
| **RAGAS Context Precision**  | N/A                       | **\>0.80**                   | LLM-as-judge on 23 test cases           |
| **Answer latency (simple)**  | <2s target                | **<2s**                      | Streamlit app wall-clock time           |
| **Answer latency (complex)** | <8s target                | **<8s**                      | Multi-hop with reranking, Streamlit app |

# **7\. Evaluation Framework & Test Cases**

The evaluation strategy addresses the recruiter's explicit requirement ('explain test cases built to assure quality') and the checklist gap identified from DocuMind: no automated LLM-as-judge, no RAGAS metrics, no regression gates. KnowledgePilot closes all three gaps.

## **7.1 Evaluation Architecture**

- **LangSmith as Tracing + Evaluation Platform: Every graph execution creates a LangSmith run with all node inputs/outputs logged. Golden dataset test cases are stored as LangSmith datasets and evaluated with built-in and custom evaluators.**
- **RAGAS Metrics (automated, LLM-as-judge):**
  - Faithfulness: Is the answer supported by the retrieved context? (cross-checks each claim)
  - Context Precision: Of the retrieved chunks, what fraction were actually useful for the answer?
  - Answer Relevancy: Does the answer address the user's original question?
  - Context Recall: Were all relevant chunks retrieved? (requires ground truth)
- **Manual golden dataset: 23 curated test cases (expanded from DocuMind's 12) with expected answers, expected source documents, expected citation documents, and - for tool-selection cases - expected tool_name in the Planner trace.**

## **7.2 Golden Test Dataset (23 Cases)**

Test cases are designed to cover all query complexity levels, all agentic capabilities (including tool selection and re-planning), and all known failure modes of traditional RAG. Cases T18, T21, T22, T23 specifically validate the tool-using agent loop - they fail by construction on a vector-only system:

| **#**   | **Complexity** | **Test Query**                                                                                                                                     | **Tests For**                                             | **Pass Criteria**                                                                                                     |
| ------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **T01** | Simple         | What is the definition of cosine similarity in the context of vector search?                                                                       | Basic retrieval, accurate factual answer                  | Answer contains formula + normalization explanation with citation                                                     |
| **T02** | Simple         | What are the default parameters of RecursiveCharacterTextSplitter in LangChain?                                                                    | Exact keyword retrieval (BM25 strength)                   | Correct chunk_size=1000, chunk_overlap=200 from docs with citation                                                    |
| **T03** | Simple         | What does the 'k' parameter control in HNSW indexing?                                                                                              | Technical term precision                                  | Correct explanation of k as number of bi-directional links                                                            |
| **T04** | Simple         | What is the hallucination rate reported in the RAGAS paper evaluation?                                                                             | Numeric fact extraction from paper                        | Specific numeric value cited correctly with page citation                                                             |
| **T05** | Complex        | Explain how BM25 and dense retrieval complement each other in hybrid search systems                                                                | Multi-concept synthesis from one domain                   | Both BM25 mechanics and dense search mechanics present, coherent comparison                                           |
| **T06** | Complex        | What evaluation metrics does RAGAS define, and how is Faithfulness computed?                                                                       | Multi-concept from one document                           | All 4 metrics named; Faithfulness formula/method described with citation                                              |
| **T07** | Complex        | Compare the chunking approaches recommended in the RAG survey paper versus the LangChain documentation                                             | Cross-document synthesis                                  | Both sources cited independently; comparison is accurate per each source                                              |
| **T08** | Multi-hop      | How does the attention mechanism described in 'Attention is All You Need' relate to the KV-cache optimisation discussed in the vLLM documentation? | Multi-hop: paper → implementation doc                     | Both transformer attention and KV-cache correctly explained; relationship stated; 2+ citations from different sources |
| **T09** | Multi-hop      | What chunking strategy does the RAGAS paper recommend, and does the LangChain RecursiveCharacterTextSplitter support it by default?                | Cross-domain multi-hop (paper + code docs)                | RAGAS recommendation + LangChain default stated; gap or alignment identified; 2 citations from different sources      |
| **T10** | Multi-hop      | How does the bi-encoder retrieval model used in standard RAG differ from cross-encoder reranking, and which papers recommend using both?           | Multi-hop: concept comparison + paper attribution         | Both architectures explained; papers citing hybrid approach referenced; 3+ citations                                  |
| **T11** | Adversarial    | What is the capital of Malaysia?                                                                                                                   | Out-of-domain query handling                              | System responds it cannot find this in its knowledge base; no hallucination                                           |
| **T12** | Adversarial    | According to the documents, what will the stock price of Anthropic be next year?                                                                   | Prediction question - should refuse                       | Graceful refusal: information not in corpus; no fabricated answer                                                     |
| **T13** | Retrieval      | What is the difference between IVF and HNSW vector indexing? (query designed to favour BM25)                                                       | BM25 precision: exact acronyms                            | Correct retrieval of chunk containing 'IVF' and 'HNSW' as exact terms                                                 |
| **T14** | Retrieval      | Describe the feeling of finding the right document when you need it urgently                                                                       | Irrelevant to corpus - semantic search may hallucinate    | System identifies low relevance; no fabricated answer; graceful response                                              |
| **T15** | Re-retrieval   | What are all the hyperparameters involved in training a LoRA adapter?                                                                              | Broad query requiring multiple chunks                     | Relevance Grader triggers at least 1 re-query iteration; all key hyperparameters covered                              |
| **T16** | Citations      | Summarise the key findings of the RAGAS evaluation paper                                                                                           | Citation accuracy for multi-claim summary                 | Every claim has a citation; no claim without supporting citation; citation pages verifiable                           |
| **T17** | Citations      | What does the LangGraph documentation say about adding memory to agents?                                                                           | Exact citation to documentation section                   | Citation includes section title 'Persistence' or equivalent; snippet verifiable                                       |
| **T18** | Tool-Select    | What did OpenAI announce about the o5 reasoning model in 2026?                                                                                     | Temporal trigger → Planner must select web_search         | Planner trace shows web_search selected (not vector); answer cites at least one web source with published_date in 2026; no hallucination from stale corpus |
| **T19** | Faithfulness   | What percentage of RAG systems in production use hybrid retrieval according to the documents?                                                      | Hallucination guard test - statistic may not be in corpus | If not in corpus: 'not found'; if present: cited correctly. No fabricated statistic                                   |
| **T20** | Regression     | Re-run T06 after a prompt template change                                                                                                          | Regression test gate                                      | Score on T06 does not degrade by >5% from baseline run; LangSmith comparison                                          |
| **T21** | Tool-Select    | Summarise the abstract of ImageNet Classification with Deep Convolutional Neural Networks                                                           | Out-of-corpus paper title → Planner must select web_search | Planner trace shows web_search selected; answer cites web source with paper details; vector_retriever returns zero-relevant |
| **T22** | Tool-Iterate   | What does the latest LangGraph release notes say about checkpointing?                                                                              | Iterate after thin vector evidence → Planner re-plans to web_search | Iteration 1 uses vector_retriever → insufficient; Planner re-plan trace shows tool switch to web_search; iteration 2 cites web source. iteration_count=2 |
| **T23** | Tool-Cite      | Compare RecursiveCharacterTextSplitter behaviour (LangChain docs) with the chunking recommendations from the RAG survey paper                       | Multi-tool, multi-source citation                         | Citations include both `source_type=vector_chunk` (RAG survey) and `source_type=web` (LangChain docs); each citation correctly attributes its source_type in UI |
| **T24** | Conversation   | Hi! How's it going?                                                                                                                             | Greeting interaction - no retrieval                        | Analyzer classifies as conversation; Node 1.5 responds with template greeting + example query; zero retrieval fired; response is friendly and on-topic |
| **T25** | Conversation   | What can you do? Who are you?                                                                                                                  | Meta-question about assistant - bounded LLM response       | Analyzer classifies as conversation; Node 1.5 uses meta-question LLM path; response accurately describes actual capabilities (AI/ML papers, LangChain docs, RAG); no hallucinated features; Hallucination Guard is skipped |

## **7.3 LangSmith Integration Plan**

- Dataset creation: Upload all 25 test cases as a LangSmith Dataset with input, expected_output, expected_sources, and expected_tool_calls fields
- Run evaluation: use client.evaluate() to run the Agentic RAG chain over the dataset with RAGAS evaluators as LangSmith custom evaluators
- Custom evaluators registered:
  - faithfulness_evaluator: LLM-as-judge checking if answer claims are grounded in retrieved Evidence
  - citation_accuracy_evaluator: verifies inline citation markers correspond to cited sources, and that source_type is correct
  - tool_selection_evaluator: for tool-selection test cases, asserts the Planner's `action_plan` includes the expected tool_name (e.g. T18 must include `web_search`)
  - re_retrieval_counter: counts how many test cases triggered re-planning and computes average iterations + tool-switch rate
- Tracing: every Streamlit demo run produces a LangSmith trace URL, shown in the UI sidebar for inspector access during the demo

# **8\. Technology Stack**

| **Layer**              | **Technology**                                               | **Justification**                                                                                                                                                                                                               |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Orchestration**      | LangGraph (LangChain)                                        | Stateful graph with conditional edges - enables adaptive routing (conversation / simple / multi-hop paths) with explicit, inspectable state transitions. Chosen over ReAct agent: structured routing avoids dynamic tool-selection ambiguity |
| **LLM**                | Llama 3.3 70B via Groq (primary) / Gemini 2.5 Flash (fallback, via adapter) | Groq's free tier provides 14,400 req/day vs Gemini's 20/day — critical for running the agentic loop without quota exhaustion during demos. A `llm_factory.py` adapter pattern allows provider hot-swap via environment variable |
| **Embeddings**         | sentence-transformers/all-MiniLM-L6-v2                       | 384-dim, fast, open-source, runs locally - no API dependency for embedding. Consistent embedding space across indexing and query time                                                                                           |
| **Dense Retrieval**    | FAISS (IndexFlatIP)                                          | Open-source, runs locally, cosine similarity with normalized vectors. No managed vector DB required for demo scale (<10K chunks)                                                                                                |
| **Sparse Retrieval**   | BM25Retriever (LangChain)                                    | Open-source, no external dependency. Complements FAISS in EnsembleRetriever                                                                                                                                                     |
| **Ensemble Fusion**    | LangChain EnsembleRetriever                                  | Built-in RRF fusion with configurable weights                                                                                                                                                                                   |
| **Reranking**          | cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers) | Open-source, runs locally. Fast cross-encoder (L-6 variant ~200ms for 15 candidates)                                                                                                                                            |
| **Demo UI**            | Streamlit                                                    | Fastest path to a working, shareable demo. Shows retrieved chunks, citations panel, LangSmith trace URL, and query complexity classification in the sidebar                                                                     |
| **Observability**      | LangSmith                                                    | Official LangChain observability platform. Free tier covers demo scale. Full graph tracing with node-level latency and LLM call logging                                                                                         |
| **Evaluation**         | RAGAS + LangSmith Evaluator API                              | RAGAS provides LLM-as-judge metrics (Faithfulness, Context Precision, Answer Relevancy). LangSmith dataset + evaluate() provides regression gate infrastructure                                                                 |
| **Document Ingestion** | LlamaParse (primary) / PyPDFLoader (fallback)                | LlamaParse preserves table structure and figure captions that pypdf flattens — important for HNSW pseudocode and RAGAS metric tables. Falls back to PyPDFLoader if `LLAMA_CLOUD_API_KEY` is not set                            |
| **Web Search Tool**    | Tavily Search API                                            | LLM-optimised search results (titles + snippets + URLs) with `published_date`. Free tier covers demo scale. Used by Planner when temporal/out-of-corpus signals fire                                                            |
| **Tool Orchestration** | LangChain Tool abstractions + LangGraph conditional edges    | Tools wrapped as LangChain `Tool` objects so the Planner produces structured ToolCall outputs; LangGraph routes Action Executor to the correct tool implementation                                                              |
| **Backend API**        | FastAPI (optional, behind Streamlit)                         | Exposes /ask and /ingest endpoints if time permits for production-style delivery signal                                                                                                                                         |

# **9\. Demo Design (Streamlit Prototype)**

The demo is designed to make every agentic capability visible without requiring the interviewer to read code. Each panel in the UI corresponds to an architectural decision explained in the Q&A.

## **9.1 UI Layout**

| **Panel**                             | **Content & Purpose**                                                                                                                                                                                                                                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Left Sidebar**                      | Query Complexity Badge (Simple / Complex / Multi-hop) - shows the Analyzer's verdict<br><br>Sub-queries list - shows decomposition for multi-hop queries<br><br>Retrieval Iterations counter - shows if re-retrieval fired<br><br>LangSmith Trace URL - live link to the run trace for inspector access |
| **Main Chat Area**                    | User query input + response with inline citation markers (\[¹\] \[²\] \[³\])<br><br>Answer text rendered with bold/highlighted citation references                                                                                                                                                      |
| **Retrieved Chunks Panel (expander)** | Top-5 chunks after reranking, showing:<br><br>• Source document + page<br><br>• Dense score (bi-encoder)<br><br>• Rerank score (cross-encoder)<br><br>• Relevance grade (Relevant / Partial / Irrelevant)<br><br>• Snippet text                                                                         |
| **Citations Panel**                   | Numbered citation list matching inline markers:<br><br>• Source document, page, section title<br><br>• Score (reranker), snippet, sub-query origin, iteration number                                                                                                                                    |
| **Evaluation Tab**                    | Run the full 20-case golden dataset evaluation with one button<br><br>Displays RAGAS scores per test case in a table<br><br>Highlights any regressions vs. baseline run                                                                                                                                 |

## **9.2 Demo Script (3-Minute Walkthrough)**

- Open the demo. Point to the left sidebar - explain the query complexity classifier.
- Type a simple query: 'What is cosine similarity?' - show: fast path, 1 retrieval, no decomposition, 1 citation.
- Type a multi-hop query: 'How does the chunking in the RAGAS paper differ from LangChain's default, and does the reranker account for this?' - show: complexity = multi-hop, 3 sub-queries in sidebar, 2 retrieval iterations if triggered, citations from 2+ sources.
- Expand the Retrieved Chunks panel - explain dual scores (dense vs. rerank), show the Relevance Grade column.
- Click the LangSmith Trace URL - show the interviewer the full graph trace in LangSmith (live demo of observability).
- Run the Evaluation tab - show RAGAS scores for the 25 test cases; highlight Faithfulness > 0.85 and tool-selection accuracy on T18, T21, T22, T23.

# **10\. Implementation Roadmap**

| **Phase**                  | **Deliverable**                       | **Key Tasks**                                                                                                                                                                              | **Effort Est.** |
| -------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| **Phase 1 Foundation**     | Corpus + Ingestion Pipeline           | • Collect 10-15 AI/ML papers + LangChain docs<br><br>• Chunking (per-type strategy) + metadata tagging<br><br>• BM25 index + FAISS index built and persisted                               | 4-6 hrs         |
| **Phase 2 Core Retrieval** | Hybrid Retriever + Reranker working   | • EnsembleRetriever (FAISS + BM25)<br><br>• Cross-encoder reranker integration<br><br>• Unit test: hybrid vs. dense-only precision comparison on 10 queries                                | 6-8 hrs         |
| **Phase 3 Agentic Graph**  | LangGraph 5-node graph functional     | • Query Analyzer (with inline conversation handling), Decomposer, Planner, Action Executor, Answer Synthesizer<br><br>• Hallucination Guard integrated into Synthesizer<br><br>• LangSmith tracing connected | 8-10 hrs        |
| **Phase 3.5 Tool Integration** | Multi-tool action space + Planner | • Planner node with structured JSON ToolCall output schema<br><br>• Action Executor dispatcher; Evidence object normaliser<br><br>• Tool implementation: Tavily web_search<br><br>• Corpus manifest in Planner prompt for first-pass tool selection<br><br>• Cross-tool Evidence dedup by content hash | 6-8 hrs         |
| **Phase 4 Citations**      | Inline citation system end-to-end     | • Citation model with source_type + URL + published_date<br><br>• Synthesizer prompt for inline markers across all source types<br><br>• Citation panel rendering with per-source-type badges in Streamlit<br><br>• Multi-hop provenance (sub_query_origin + tool_rationale fields)                             | 3-4 hrs         |
| **Phase 5 Evaluation**     | Golden dataset + RAGAS automated eval | • 25 test cases authored + stored as LangSmith dataset (incl. T18/T21/T22/T23 tool cases)<br><br>• RAGAS + tool_selection_evaluator + citation_accuracy_evaluator registered in LangSmith<br><br>• Frozen-cache mode for web_search to make eval reproducible<br><br>• Evaluation tab in Streamlit                                              | 4-5 hrs         |
| **Phase 6 Demo Polish**    | Submission-ready Streamlit demo       | • Sidebar with complexity badge, sub-queries, action plan view, iteration counter, tool-switch indicator<br><br>• Query examples pre-loaded (incl. tool-selection examples)<br><br>• README with architecture diagram and evaluation results                        | 2-3 hrs         |

**Total estimated effort: 33-44 hours for a complete, submission-ready demo. Phase 1-3.5 are the MVP for a tool-using agentic prototype; Phase 4 adds end-to-end citations across source_types. Phases 5-6 deliver the evaluation framework and polish that differentiate the submission. Phase 3.5 (tool integration) is what elevates this from "smart RAG pipeline" to "tool-using agent" and is non-negotiable for the agentic-RAG framing.**

# **11\. Submission Differentiators**

This section maps each recruiter requirement to how this system addresses it, and identifies where it goes beyond the minimum requirement.

| **Requirement**                                           | **Minimum Pass**                         | **KnowledgePilot - How It Exceeds**                                                                                                                              |
| --------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Build Agentic RAG that retrieves correctly**            | Single agent loop with basic retrieval   | 5-node LangGraph (Analyze→Decompose→Plan→Act→Generate); two-tool action space (vector_retriever, web_search) with a Planner that decides *if/when/where* to search; cross-encoder reranking inside vector_retriever; each design decision individually justified |
| **Demo on working prototype**                             | Streamlit app that answers questions     | Streamlit with 5-panel layout: chat, retrieved chunks with dual scores, citations panel, sidebar complexity/iteration tracking, live LangSmith trace URL         |
| **Discussion of thought process and implementation flow** | Code walkthrough                         | Structured PRD + architecture diagram + design rationale for every node decision (why LangGraph over ReAct, why cross-encoder over pure bi-encoder)              |
| **Investigation of Agentic RAG system as a whole**        | Overview explanation                     | Full LangGraph state schema, node-by-node spec, failure mode analysis, trade-offs vs. traditional RAG mapped in comparative table                                |
| **Investigate Traditional vs. Agentic RAG differences**   | Bullet list comparison                   | 7-dimension comparative table with specific DocuMind vs. KnowledgePilot mapping; honest trade-off analysis including when traditional RAG is appropriate         |
| **Explain test cases for quality assurance**              | Ad-hoc test scenarios                    | 25-case golden dataset with complexity labelling, adversarial cases, regression gate; RAGAS automated metrics; LangSmith evaluation harness; LLM-as-judge rubric |
| **Bonus: Citations**                                      | Source document names appended to answer | Inline citation markers (\[¹\]) in answer text; multi-hop provenance (sub_query_origin); dual scoring (dense + rerank score per citation)                        |
| **Bonus: Optimised retrieval (accuracy + performance)**   | Top-K vector search                      | Hybrid dense+sparse (EnsembleRetriever RRF) + cross-encoder reranker; per-document-type chunking strategy; Precision@1 target >85%                               |