# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ── Step 1: Load your API key from .env file ──────────────────────
load_dotenv()

# ── Step 2: Load all PDFs from the docs/ folder ───────────────────
print(" Loading documents...")
loader = PyPDFDirectoryLoader("docs/")
documents = loader.load()
print(f"   Loaded {len(documents)} page(s) from your PDF")

# ── Step 3: Split documents into smaller chunks ───────────────────
print("  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"   Created {len(chunks)} chunks")

# ── Step 4: Create embeddings and store in ChromaDB ───────────────
print(" Creating embeddings and storing in ChromaDB...")
embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

print(" Done! Your documents are now stored in ChromaDB.")
print(f"   Database saved in the 'db/' folder")

