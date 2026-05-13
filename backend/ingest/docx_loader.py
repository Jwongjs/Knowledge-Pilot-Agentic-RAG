from __future__ import annotations
from pathlib import Path
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 150


def load_docx(path: str | Path, doc_type: str = "research_paper") -> list[Document]:
    docx = DocxDocument(str(path))
    text = "\n".join(p.text for p in docx.paragraphs if p.text.strip())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": Path(path).name, "doc_type": doc_type}],
    )
    return chunks
