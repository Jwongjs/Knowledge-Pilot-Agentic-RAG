"""
KnowledgePilot — Streamlit demo.

Run from repo root:
    streamlit run frontend/app.py

Requires backend running at localhost:8000:
    cd backend && uvicorn main:app --reload
"""
from __future__ import annotations
import uuid
import httpx
import streamlit as st

_API = "http://localhost:8000"
_TIMEOUT = 120.0

_EXAMPLE_QUERIES = [
    ("Simple", "What does the 'k' parameter control in HNSW indexing?"),
    ("Simple", "What evaluation metrics does RAGAS define, and how is Faithfulness computed?"),
    ("Multi-hop", "How does the attention mechanism in 'Attention is All You Need' relate to KV-cache optimisation in vLLM?"),
    ("Multi-hop", "Compare the chunking approaches in the RAG survey paper versus LangChain's RecursiveCharacterTextSplitter."),
    ("Tool-select", "What did the latest LangGraph release say about checkpointing?"),
    ("Tool-select", "Summarise the abstract of ImageNet Classification with Deep CNNs"),
]

_COMPLEXITY_COLOURS = {
    "conversation": "🟣",
    "simple":       "🟢",
    "complex":      "🟡",
    "multi_hop":    "🔴",
}


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []          # list of (query, response_dict)
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def _post_ask(query: str) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.post(f"{_API}/ask", json={
            "query": query,
            "session_id": st.session_state.session_id,
        })
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(last_response: dict | None):
    with st.sidebar:
        st.title("KnowledgePilot")
        st.caption("Agentic RAG · Llama 3.3 70B (Groq)")
        st.divider()

        if last_response:
            complexity = last_response.get("query_complexity", "simple")
            colour = _COMPLEXITY_COLOURS.get(complexity, "⚪")
            st.markdown(f"**Query type** {colour} `{complexity}`")

            iters = last_response.get("iteration_count", 1)
            st.markdown(f"**Iterations** `{iters}`")

            sub_qs = last_response.get("sub_queries", [])
            if sub_qs:
                st.markdown("**Sub-queries**")
                for sq in sub_qs:
                    st.caption(f"• {sq}")

            flag = last_response.get("hallucination_flag", False)
            if flag:
                st.warning("⚠️ Hallucination guard flagged this answer")
            else:
                st.success("✅ Hallucination guard passed")

            trace_url = last_response.get("langsmith_trace_url")
            if trace_url:
                st.markdown(f"[🔗 LangSmith trace]({trace_url})")

        st.divider()
        st.markdown("**Example queries**")
        for label, q in _EXAMPLE_QUERIES:
            if st.button(f"{label}: {q[:45]}…" if len(q) > 45 else f"{label}: {q}", key=q):
                st.session_state.pending_query = q

        st.divider()
        if st.button("🗑️ Clear conversation"):
            st.session_state.history = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def _render_chat_tab():
    st.header("Chat")

    # Render history
    for query, resp in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            st.markdown(resp.get("answer", ""))

    # Input — picks up example-query button clicks via pending_query
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask about AI/ML papers, LangChain, or RAG…") or pending

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp = _post_ask(user_input)
                    st.markdown(resp.get("answer", ""))
                    st.session_state.history.append((user_input, resp))
                    st.rerun()
                except httpx.ConnectError:
                    st.error("Cannot reach backend at localhost:8000. Is `uvicorn main:app --reload` running?")
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429:
                        st.error("⚠️ Gemini API quota exhausted. Visit https://ai.studio/spend to raise your spending cap, then retry.")
                    else:
                        detail = exc.response.json().get("detail", str(exc))
                        st.error(f"Backend error {exc.response.status_code}: {detail}")
                except Exception as exc:
                    st.error(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Chunks tab
# ---------------------------------------------------------------------------

def _render_chunks_tab(last_response: dict | None):
    st.header("Retrieved Chunks")
    if not last_response:
        st.info("Ask a question to see retrieved chunks here.")
        return

    citations = last_response.get("citations", [])
    if not citations:
        st.info("No chunks retrieved for the last query.")
        return

    for c in citations:
        source_type = c.get("source_type", "")
        icon = {"vector_chunk": "📄", "web": "🌐", "arxiv": "📚"}.get(source_type, "📄")
        label = c.get("source_document") or c.get("url") or "unknown"

        with st.expander(f"{icon} [{c.get('citation_id')}] {label}", expanded=False):
            col1, col2 = st.columns(2)
            dense = c.get("dense_score")
            rerank = c.get("rerank_score")
            col1.metric("Dense score", f"{dense:.3f}" if dense is not None else "—")
            col2.metric("Rerank score", f"{rerank:.3f}" if rerank is not None else "—")

            st.markdown(f"**Snippet:** {c.get('snippet', '')}")

            if c.get("sub_query_origin"):
                st.caption(f"Sub-query: {c['sub_query_origin']}")
            if c.get("iteration"):
                st.caption(f"Retrieved in iteration {c['iteration']}")
            if c.get("url"):
                st.caption(f"URL: {c['url']}")
            if c.get("published_date"):
                st.caption(f"Published: {c['published_date']}")


# ---------------------------------------------------------------------------
# Citations tab
# ---------------------------------------------------------------------------

def _render_citations_tab(last_response: dict | None):
    st.header("Citations")
    if not last_response:
        st.info("Ask a question to see citations here.")
        return

    citations = last_response.get("citations", [])
    if not citations:
        st.info("No citations for the last query.")
        return

    for c in citations:
        cid = c.get("citation_id", "?")
        source = c.get("source_document") or c.get("url") or "unknown"
        source_type = c.get("source_type", "")
        icon = {"vector_chunk": "📄", "web": "🌐", "arxiv": "📚"}.get(source_type, "📄")

        st.markdown(f"**[{cid}]** {icon} `{source_type}` — {source}")

        cols = st.columns(3)
        cols[0].caption(f"Sub-query: {c.get('sub_query_origin') or '—'}")
        cols[1].caption(f"Rationale: {c.get('tool_rationale') or '—'}")
        cols[2].caption(f"Iteration: {c.get('iteration') or '—'}")

        st.markdown(f"> {c.get('snippet', '')}")

        if c.get("url"):
            st.markdown(f"[Open source]({c['url']})")

        st.divider()


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------

def _render_eval_tab():
    st.header("Evaluation")
    st.markdown(
        "Run the 25-case golden dataset against the live backend and display scores. "
        "This calls every case sequentially — expect ~5 minutes."
    )

    if st.button("▶ Run evaluation", type="primary"):
        import sys
        from pathlib import Path
        # Add backend to path so we can import evaluation modules
        backend_path = Path(__file__).parent.parent / "backend"
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        from dotenv import load_dotenv
        load_dotenv(backend_path / ".env")

        from backend.evaluation.golden_dataset import GOLDEN_DATASET

        results = []
        progress = st.progress(0, text="Running evaluation…")
        status = st.empty()

        for i, case in enumerate(GOLDEN_DATASET):
            status.caption(f"[{i+1}/{len(GOLDEN_DATASET)}] {case['id']}: {case['query'][:60]}…")
            try:
                resp = _post_ask(case["query"])
                citations = resp.get("citations", [])
                actual_types = {c.get("source_type") for c in citations}
                expected = set(case.get("expected_sources", []))
                hit = expected.issubset(actual_types) if expected else True
                results.append({
                    "ID": case["id"],
                    "Complexity": case["complexity"],
                    "Query": case["query"][:60] + ("…" if len(case["query"]) > 60 else ""),
                    "Expected sources": ", ".join(case.get("expected_sources", [])) or "—",
                    "Actual sources": ", ".join(sorted(actual_types)) or "—",
                    "Iterations": resp.get("iteration_count", 1),
                    "Hallucination": "⚠️" if resp.get("hallucination_flag") else "✅",
                    "Pass": "✅" if hit else "❌",
                })
            except Exception as exc:
                results.append({
                    "ID": case["id"],
                    "Complexity": case["complexity"],
                    "Query": case["query"][:60],
                    "Expected sources": "",
                    "Actual sources": f"ERROR: {exc}",
                    "Iterations": 0,
                    "Hallucination": "—",
                    "Pass": "❌",
                })
            progress.progress((i + 1) / len(GOLDEN_DATASET))

        status.empty()
        progress.empty()

        import pandas as pd
        df = pd.DataFrame(results)
        pass_rate = (df["Pass"] == "✅").mean()
        hal_rate = (df["Hallucination"] == "⚠️").mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Pass rate", f"{pass_rate:.0%}")
        c2.metric("Hallucination rate", f"{hal_rate:.0%}")
        c3.metric("Cases run", len(results))

        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="KnowledgePilot",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()

    last_response = st.session_state.history[-1][1] if st.session_state.history else None
    _render_sidebar(last_response)

    chat_tab, chunks_tab, citations_tab, eval_tab = st.tabs([
        "💬 Chat", "📄 Chunks", "🔖 Citations", "📊 Evaluation"
    ])

    with chat_tab:
        _render_chat_tab()
    with chunks_tab:
        _render_chunks_tab(last_response)
    with citations_tab:
        _render_citations_tab(last_response)
    with eval_tab:
        _render_eval_tab()


if __name__ == "__main__":
    main()
