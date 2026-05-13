from __future__ import annotations

# 25 golden test cases matching the PRD (§7.2)
GOLDEN_DATASET: list[dict] = [
    # --- Simple ---
    {"id": "T01", "complexity": "simple",
     "query": "What is the definition of cosine similarity in the context of vector search?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Contains formula + normalization explanation with citation"},

    {"id": "T02", "complexity": "simple",
     "query": "What are the default parameters of RecursiveCharacterTextSplitter in LangChain?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Correct chunk_size=1000, chunk_overlap=200 from docs with citation"},

    {"id": "T03", "complexity": "simple",
     "query": "What does the 'k' parameter control in HNSW indexing?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Correct explanation of k as number of bi-directional links"},

    {"id": "T04", "complexity": "simple",
     "query": "What is the hallucination rate reported in the RAGAS paper evaluation?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Specific numeric value cited correctly with page citation"},

    # --- Complex ---
    {"id": "T05", "complexity": "complex",
     "query": "Explain how BM25 and dense retrieval complement each other in hybrid search systems",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Both BM25 mechanics and dense search mechanics present, coherent comparison"},

    {"id": "T06", "complexity": "complex",
     "query": "What evaluation metrics does RAGAS define, and how is Faithfulness computed?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "All 4 metrics named; Faithfulness formula/method described with citation"},

    {"id": "T07", "complexity": "complex",
     "query": "Compare the chunking approaches recommended in the RAG survey paper versus the LangChain documentation",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Both sources cited independently; comparison accurate per each source"},

    # --- Multi-hop ---
    {"id": "T08", "complexity": "multi_hop",
     "query": "How does the attention mechanism described in 'Attention is All You Need' relate to the KV-cache optimisation discussed in the vLLM documentation?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Both transformer attention and KV-cache correctly explained; 2+ citations from different sources"},

    {"id": "T09", "complexity": "multi_hop",
     "query": "What chunking strategy does the RAGAS paper recommend, and does the LangChain RecursiveCharacterTextSplitter support it by default?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "RAGAS recommendation + LangChain default stated; 2 citations from different sources"},

    {"id": "T10", "complexity": "multi_hop",
     "query": "How does the bi-encoder retrieval model used in standard RAG differ from cross-encoder reranking, and which papers recommend using both?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Both architectures explained; papers citing hybrid approach referenced; 3+ citations"},

    # --- Adversarial ---
    {"id": "T11", "complexity": "adversarial",
     "query": "What is the capital of Malaysia?",
     "expected_sources": [],
     "pass_criteria": "System responds it cannot find this in its knowledge base; no hallucination"},

    {"id": "T12", "complexity": "adversarial",
     "query": "According to the documents, what will the stock price of Anthropic be next year?",
     "expected_sources": [],
     "pass_criteria": "Graceful refusal: information not in corpus; no fabricated answer"},

    # --- Retrieval ---
    {"id": "T13", "complexity": "retrieval",
     "query": "What is the difference between IVF and HNSW vector indexing?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Correct retrieval of chunk containing 'IVF' and 'HNSW' as exact terms"},

    {"id": "T14", "complexity": "retrieval",
     "query": "Describe the feeling of finding the right document when you need it urgently",
     "expected_sources": [],
     "pass_criteria": "System identifies low relevance; no fabricated answer; graceful response"},

    # --- Re-retrieval ---
    {"id": "T15", "complexity": "re_retrieval",
     "query": "What are all the hyperparameters involved in training a LoRA adapter?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Relevance Grader triggers at least 1 re-query iteration; all key hyperparameters covered"},

    # --- Citations ---
    {"id": "T16", "complexity": "citations",
     "query": "Summarise the key findings of the RAGAS evaluation paper",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Every claim has a citation; no claim without supporting citation"},

    {"id": "T17", "complexity": "citations",
     "query": "What does the LangGraph documentation say about adding memory to agents?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Citation includes section title 'Persistence' or equivalent; snippet verifiable"},

    # --- Tool-Select ---
    {"id": "T18", "complexity": "tool_select",
     "query": "What did OpenAI announce about the o5 reasoning model in 2026?",
     "expected_sources": ["web"],
     "expected_tool_calls": ["web_search"],
     "pass_criteria": "Planner trace shows web_search selected; answer cites web source with published_date in 2026"},

    # --- Faithfulness ---
    {"id": "T19", "complexity": "faithfulness",
     "query": "What percentage of RAG systems in production use hybrid retrieval according to the documents?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "If not in corpus: 'not found'; if present: cited correctly. No fabricated statistic"},

    # --- Regression ---
    {"id": "T20", "complexity": "regression",
     "query": "What evaluation metrics does RAGAS define, and how is Faithfulness computed?",
     "expected_sources": ["vector_chunk"],
     "pass_criteria": "Score on T06 does not degrade by >5% from baseline run"},

    # --- Tool-Select (out-of-corpus paper by title) ---
    {"id": "T21", "complexity": "tool_select",
     "query": "Summarise the abstract of ImageNet Classification with Deep Convolutional Neural Networks",
     "expected_sources": ["web"],
     "expected_tool_calls": ["web_search"],
     "pass_criteria": "Planner picks web_search since paper is not in local corpus; answer cites web source"},

    # --- Tool-Iterate ---
    {"id": "T22", "complexity": "tool_iterate",
     "query": "What does the latest LangGraph release notes say about checkpointing?",
     "expected_sources": ["web"],
     "expected_tool_calls": ["vector_retriever", "web_search"],
     "pass_criteria": "Iteration 1 vector insufficient; Planner re-plans to web_search; iteration_count=2"},

    # --- Tool-Cite (mixed corpus + web) ---
    {"id": "T23", "complexity": "tool_cite",
     "query": "Compare RecursiveCharacterTextSplitter behaviour (LangChain docs) with the chunking recommendations from the RAG survey paper",
     "expected_sources": ["vector_chunk", "web"],
     "expected_tool_calls": ["vector_retriever", "web_search"],
     "pass_criteria": "Citations include both source_type=vector_chunk and source_type=web"},

    # --- Conversation ---
    {"id": "T24", "complexity": "conversation",
     "query": "Hi! How's it going?",
     "expected_sources": [],
     "pass_criteria": "Analyzer classifies as conversation; Node 1.5 responds with template greeting; zero retrieval fired"},

    {"id": "T25", "complexity": "conversation",
     "query": "What can you do? Who are you?",
     "expected_sources": [],
     "pass_criteria": "Analyzer classifies as conversation; Node 1.5 uses meta-question LLM path; response describes actual capabilities"},
]
