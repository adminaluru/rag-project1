"""Ingest PDFs from `docs/` into a ChromaDB collection.

Requirements (install if missing):
  pip install PyPDF2 sentence-transformers chromadb

Usage:
  python .ingest.py

This script:
  - Reads all .pdf files in the `docs/` folder
  - Extracts text from each PDF
  - Splits text into overlapping chunks
  - Converts chunks to embeddings using SentenceTransformers
  - Stores documents + embeddings + metadata in ChromaDB (persisted to `chroma_db/`)
"""

import os
import glob
import uuid
from typing import List

try:
    from PyPDF2 import PdfReader
except Exception as e:
    raise ImportError("PyPDF2 is required. Run: pip install PyPDF2") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("sentence-transformers is required. Run: pip install sentence-transformers") from e

try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    raise ImportError("chromadb is required. Run: pip install chromadb") from e


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def ingest_docs(
    docs_dir: str = "docs",
    persist_dir: str = "chroma_db",
    collection_name: str = "pdfs",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    # Find PDF files
    pdf_paths = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))
    if not pdf_paths:
        print(f"No PDF files found in {docs_dir}")
        return

    # Load embedding model
    print(f"Loading embedding model '{embedding_model_name}'...")
    embedder = SentenceTransformer(embedding_model_name)

    # Setup chroma client
    client = chromadb.Client(Settings(chroma_db_impl="chromadb.db.impl.sqlite3", persist_directory=persist_dir))
    collection = client.get_or_create_collection(name=collection_name)

    total_chunks = 0

    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            print(f"  -> No text extracted from {pdf_path}")
            continue

        # Generate ids, metadatas
        ids = []
        metadatas = []
        documents = chunks
        for i, _ in enumerate(chunks):
            chunk_id = f"{os.path.basename(pdf_path)}-{i}-{uuid.uuid4().hex}"
            ids.append(chunk_id)
            metadatas.append({"source": os.path.basename(pdf_path), "chunk_index": i})

        # Compute embeddings (returns numpy array)
        print(f"  -> Creating embeddings for {len(chunks)} chunks...")
        embeddings = embedder.encode(documents, show_progress_bar=True)

        # Ensure lists
        embeddings = [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

        # Add to collection
        print(f"  -> Adding {len(chunks)} vectors to ChromaDB collection '{collection_name}'")
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        total_chunks += len(chunks)

    # Persist
    try:
        client.persist()
    except Exception:
        # Some chroma setups persist automatically; ignore if not supported
        pass

    print(f"Ingest complete. Added {total_chunks} chunks from {len(pdf_paths)} PDF(s) to '{persist_dir}'")


if __name__ == "__main__":
    ingest_docs()
