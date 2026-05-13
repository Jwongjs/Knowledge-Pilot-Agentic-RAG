from __future__ import annotations
import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_CHUNK_CONFIG = {
    "research_paper": {"chunk_size": 1000, "chunk_overlap": 150},
    "documentation":  {"chunk_size": 600,  "chunk_overlap": 100},
    "faq":            {"chunk_size": 400,  "chunk_overlap": 80},
}

# Heuristic: short all-caps or title-case lines likely to be section headings
_HEADING_RE = re.compile(r"^(?:[A-Z][A-Z\s]{2,50}|[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,6})$")


def load_pdf(path: str | Path, doc_type: str = "research_paper") -> list[Document]:
    config = _CHUNK_CONFIG.get(doc_type, _CHUNK_CONFIG["research_paper"])
    pages = _parse_pdf(Path(path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    chunks = splitter.split_documents(pages)

    source_name = Path(path).stem  # e.g. "RAGAS" not "RAGAS.pdf"
    for chunk in chunks:
        chunk.metadata.setdefault("source", Path(path).name)
        chunk.metadata["doc_type"] = doc_type
        chunk.metadata["section_title"] = _extract_section_title(chunk.page_content)

    return chunks


def _parse_pdf(path: Path) -> list[Document]:
    """Use LlamaParse when LLAMA_CLOUD_API_KEY is set, else fall back to PyPDFLoader."""
    if os.environ.get("LLAMA_CLOUD_API_KEY"):
        from llama_parse import LlamaParse
        parser = LlamaParse(result_type="markdown", verbose=False)
        llama_docs = parser.load_data(str(path))
        return [Document(page_content=d.text, metadata={"source": path.name}) for d in llama_docs]
    return PyPDFLoader(str(path)).load()


def _extract_section_title(text: str) -> str | None:
    """Return the first heading-like line in the chunk, or None."""
    for line in text.splitlines():
        line = line.strip()
        if line and _HEADING_RE.match(line):
            return line
    return None