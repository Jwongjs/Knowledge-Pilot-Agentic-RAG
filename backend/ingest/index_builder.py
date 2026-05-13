from __future__ import annotations
import pickle
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ingest.pdf_loader import load_pdf
from ingest.docx_loader import load_docx

_INDEX_DIR = Path("indices")
_FAISS_PATH = _INDEX_DIR / "faiss_index"
_BM25_PATH = _INDEX_DIR / "bm25_index.pkl"
_SOURCES_PATH = _INDEX_DIR / "sources.pkl"
_DOCS_PATH = _INDEX_DIR / "docs.pkl"

_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class IndexBuilder:
    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDINGS_MODEL)
        self._docs: list[Document] = []
        self._sources: list[dict] = []

    def load_indices(self) -> tuple[FAISS, BM25Retriever]:
        if _FAISS_PATH.exists() and _BM25_PATH.exists():
            faiss_store = FAISS.load_local(
                str(_FAISS_PATH), self._embeddings, allow_dangerous_deserialization=True
            )
            with open(_BM25_PATH, "rb") as f:
                bm25 = pickle.load(f)
            if _SOURCES_PATH.exists():
                with open(_SOURCES_PATH, "rb") as f:
                    self._sources = pickle.load(f)
            if _DOCS_PATH.exists():
                with open(_DOCS_PATH, "rb") as f:
                    self._docs = pickle.load(f)
            return faiss_store, bm25

        # No indices built yet — return empty placeholders
        faiss_store = FAISS.from_texts(["placeholder"], self._embeddings)
        bm25 = BM25Retriever.from_texts(["placeholder"])
        return faiss_store, bm25

    def build_from_directory(self, directory: str | Path, doc_type: str = "research_paper") -> int:
        """Ingest all PDFs and DOCX files in a directory. Returns total chunks indexed."""
        directory = Path(directory)
        files = list(directory.glob("*.pdf")) + list(directory.glob("*.docx"))
        if not files:
            raise ValueError(f"No .pdf or .docx files found in {directory}")

        total = 0
        for file in sorted(files):
            chunks = self._load_file(file, doc_type)
            self._docs.extend(chunks)
            self._sources.append({
                "path": str(file),
                "name": file.name,
                "doc_type": doc_type,
                "chunks": len(chunks),
            })
            total += len(chunks)
            print(f"  Loaded {file.name}: {len(chunks)} chunks")

        self._persist_all()
        print(f"\nTotal chunks indexed: {total}")
        return total

    async def ingest(self, source_path: str, doc_type: str) -> int:
        """Ingest a single file (used by FastAPI /ingest endpoint)."""
        path = Path(source_path)
        chunks = self._load_file(path, doc_type)

        self._docs.extend(chunks)
        self._sources.append({
            "path": source_path,
            "name": path.name,
            "doc_type": doc_type,
            "chunks": len(chunks),
        })
        self._persist_all()
        return len(chunks)

    def list_sources(self) -> list[dict]:
        return self._sources

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path, doc_type: str) -> list[Document]:
        if path.suffix.lower() == ".pdf":
            return load_pdf(path, doc_type)
        if path.suffix.lower() == ".docx":
            return load_docx(path, doc_type)
        raise ValueError(f"Unsupported file type: {path.suffix}. Supported: .pdf, .docx")

    def _persist_all(self) -> None:
        """Rebuild both indices from the full self._docs corpus and persist everything."""
        _INDEX_DIR.mkdir(exist_ok=True)

        # FAISS — incremental add if index already exists, else create fresh
        if _FAISS_PATH.exists():
            faiss_store = FAISS.load_local(
                str(_FAISS_PATH), self._embeddings, allow_dangerous_deserialization=True
            )
            # Add only the new docs (last N = total - previously persisted)
            previously_persisted = len(self._docs) - self._new_chunk_count()
            new_docs = self._docs[previously_persisted:]
            if new_docs:
                faiss_store.add_documents(new_docs)
        else:
            faiss_store = FAISS.from_documents(self._docs, self._embeddings)
        faiss_store.save_local(str(_FAISS_PATH))

        # BM25 always rebuilt over full corpus so ranking scores are consistent
        bm25 = BM25Retriever.from_documents(self._docs)
        with open(_BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)

        # Persist raw docs and source manifest for future BM25 rebuilds
        with open(_DOCS_PATH, "wb") as f:
            pickle.dump(self._docs, f)
        with open(_SOURCES_PATH, "wb") as f:
            pickle.dump(self._sources, f)

    def _new_chunk_count(self) -> int:
        """How many docs were in the index before the current ingest call."""
        if _DOCS_PATH.exists():
            with open(_DOCS_PATH, "rb") as f:
                old_docs = pickle.load(f)
            return len(old_docs)
        return 0