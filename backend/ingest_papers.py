"""
One-shot script: ingest all PDFs in data/papers/ into FAISS + BM25 indices.
Run from the backend/ directory:
    python ingest_papers.py
"""
import sys
from pathlib import Path

# Ensure backend/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from ingest.index_builder import IndexBuilder

PAPERS_DIR = Path(__file__).parent.parent / "data" / "papers"


def main():
    print(f"Ingesting papers from: {PAPERS_DIR}\n")
    builder = IndexBuilder()
    total = builder.build_from_directory(PAPERS_DIR, doc_type="research_paper")

    print("\nSources indexed:")
    for s in builder.list_sources():
        print(f"  {s['name']:45s} {s['chunks']:>4d} chunks")

    print(f"\nDone. {total} total chunks across {len(builder.list_sources())} documents.")
    print("Indices written to: backend/indices/")


if __name__ == "__main__":
    main()
