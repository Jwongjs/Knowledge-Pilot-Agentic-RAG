# Two separate things LangSmith does:

Same service, two different jobs

LangSmith (smith.langchain.com)
│
├── Tracing      ← triggered by LANGCHAIN_TRACING_V2=true
│                   watches every live /ask request automatically
│                   no code needed, just the env var
│
└── Evaluation   ← triggered by python evaluation/run_eval.py
                    you run it manually when you want a score
                    uses langsmith.Client() + evaluate()

Both use the same LANGCHAIN_API_KEY to authenticate with LangSmith. Same key, same dashboard, different features.

So when main.py is running
Every /ask request → graph runs → LangSmith automatically receives a trace via the callback mechanism. You didn't call LangSmith anywhere in the code — the env var wired it up silently.

When you run run_eval.py
It explicitly calls langsmith.Client() to upload the dataset and post evaluation scores. This is the only place LANGCHAIN_API_KEY is used in actual code.

Bottom line
Who triggers it	Code needed
Tracing	LANGCHAIN_TRACING_V2=true env var	None — automatic
Evaluation	You run run_eval.py	langsmith.Client() explicitly

# How LangSmith auto-detects graph runs
When LangChain imports, it checks LANGCHAIN_TRACING_V2 at module load time. If true, it registers a global LangSmithCallbackHandler into LangChain's callback manager.

Every LangChain/LangGraph object — ChatGoogleGenerativeAI, StateGraph, EnsembleRetriever — inherits from Runnable. When .ainvoke() is called on any Runnable, it automatically passes the active callbacks down the call chain. No explicit wiring needed.

graph.ainvoke(state)
  └── LangGraph StateGraph.ainvoke()        ← emits on_chain_start
        ├── query_analyzer.run()
        │     └── llm.ainvoke(prompt)        ← emits on_llm_start / on_llm_end
        ├── planner.run()
        │     └── llm.ainvoke(prompt)        ← emits on_llm_start / on_llm_end
        ├── action_executor.run()
        │     └── vector_retriever.retrieve()
        │           └── ensemble.ainvoke()   ← emits on_retriever_start / end
        └── ...                              ← emits on_chain_end
Each on_* event fires the registered LangSmithCallbackHandler, which ships a span to LangSmith. The spans nest by call depth, so LangSmith reconstructs the full tree.

What you actually see in LangSmith
One trace per /ask request, structured as:


KnowledgePilotGraph                     [root span, full latency]
├── query_analyzer                       [node span]
│     └── ChatGoogleGenerativeAI         [LLM span: prompt, output, token count]
├── planner                              [node span]
│     └── ChatGoogleGenerativeAI         [LLM span]
├── action_executor                      [node span]
│     └── EnsembleRetriever              [retriever span: query, docs returned]
├── relevance_grader                     [node span — repeats if re-plan loop fires]
│     └── ChatGoogleGenerativeAI × N    [one LLM call per Evidence chunk graded]
├── answer_synthesizer                   [node span]
│     └── ChatGoogleGenerativeAI         [LLM span]
└── hallucination_guard                  [node span — no LLM, just CPU]
Every span shows: latency, token usage, exact prompt sent, exact output received, and any errors.

# The evaluation loop
For each of the 25 golden cases, run_eval.py does this:


golden_dataset[T01]
  │
  ├── inputs:  { query, expected_sources, expected_tool_calls }
  └── outputs: { answer, citations, iteration_count, ... }  ← from running the graph
        │
        ├── faithfulness_evaluator(run, example)
        ├── citation_accuracy_evaluator(run, example)
        ├── tool_selection_evaluator(run, example)
        └── re_retrieval_counter(run, example)

How each evaluator scores:
1. faithfulness_evaluator — is the answer grounded?

answer sentences: ["HNSW uses a layered graph.", "Each node connects to k neighbours."]
cited snippets:   "layered graph structure where each node..."

sentence 1 → shares "layered", "graph" with snippet → supported ✓
sentence 2 → shares "node", "connects" with snippet  → supported ✓

score = 2/2 = 1.0
-----------------------------------------------------------------------------------
It's a token overlap heuristic — counts what fraction of answer sentences share at least one >4-character word with the cited snippets. This is honest about being a heuristic, not a true LLM-as-judge. Phase 5 is meant to upgrade this to a real LLM judge.

2. citation_accuracy_evaluator — did the right tool get used?

T08 expected_sources: ["vector_chunk"]
actual citations source_types: {"vector_chunk", "web"}

hits = 1 ("vector_chunk" found)
score = 1/1 = 1.0  ✓

T18 expected_sources: ["web"]
actual citations source_types: {"vector_chunk"}   ← planner used wrong tool

hits = 0
score = 0/1 = 0.0  ✗
---------------------------------------------------------------------------------
Checks whether the source types in the final citations match what the golden case expected. Proxy for "did retrieval come from the right place."

3. tool_selection_evaluator — did the planner pick the right tool?
Only fires on cases that have expected_tool_calls defined (T18, T21, T22, T23). Maps tool names to their source types since action_plan isn't in the response — uses citations as proxy:
--------------------------------------------------------------------------------
T22 expected_tool_calls: ["vector_retriever", "web_search"]
    maps to source_types: ["vector_chunk",    "web"]

actual citation types: {"vector_chunk", "web"}

hits = 2/2 = 1.0  ✓

4. re_retrieval_counter — did the re-plan loop fire?

T15 (LoRA hyperparameters — designed to need re-retrieval)
iteration_count = 2  →  fired = 1  (score: 1)

T01 (simple cosine similarity)
iteration_count = 1  →  fired = 0  (score: 0)
----------------------------------------------------------------------------------
This isn't a pass/fail — it's a rate metric. You want it firing on hard cases (T15, T22) and not firing on easy ones. LangSmith shows you the aggregate rate across all 25 cases.

What "justification" looks like in LangSmith:
Each score is stored as a span with a comment field. For faithfulness you'd see:
- Test: T04
- Score: 0.6
- Comment: "3/5 sentences supported by cited snippets"

And because tracing is also running, you can click into the exact graph trace for that test case and see which node produced what — so if T18 scores 0.0 on tool selection, you can open the planner span and read the exact prompt + output that caused it to pick vector_retriever instead of web_search.