# streamlit_app.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────
MODEL_NAME: str = "gpt-3.5-turbo"
DB_PATH: str = "db"
TOP_K: int = 3
TEMPERATURE: float = 0.0

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title="Chat With Your Docs", page_icon="📄", layout="wide")
st.title("📄 Chat With Your Documents")

# ── API Key check ─────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()

# ── Helper: Get active documents from ChromaDB ───────────────────
def get_active_documents() -> list:
    """Read metadata from ChromaDB and return list of unique filenames."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("langchain")
        results = collection.get(include=["metadatas"])
        sources = set(
            m.get("source", "Unknown")
            for m in results["metadatas"]
        )
        return sorted(sources)
    except Exception:
        return []

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Your PDFs")
    uploaded_files = st.file_uploader(
        "Drag and drop PDFs here",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) ready to process")
        for f in uploaded_files:
            st.caption(f"📄 {f.name}")

    process_btn = st.button("⚙️ Process Documents", disabled=not uploaded_files)

    # ── Show currently active documents ──────────────────────────
    st.divider()
    st.subheader("🗂️ Active Documents")

    active_docs = get_active_documents()
    if active_docs:
        st.caption(f"{len(active_docs)} document(s) loaded:")
        for doc in active_docs:
            st.caption(f"✅ {doc}")
    else:
        st.caption("No documents loaded yet.")
        st.caption("Upload PDFs above and click Process.")

# ── Process uploaded PDFs ─────────────────────────────────────────
def process_pdfs(files) -> bool:
    """
    Takes uploaded files, splits into chunks,
    wipes old ChromaDB collection, stores new chunks.
    Whatever is uploaded together = the active set.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        import chromadb

        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        progress = st.progress(0, text="Reading PDFs...")

        for i, uploaded_file in enumerate(files):
            # Save to temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Tag each chunk with original filename for source tracking
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            os.unlink(tmp_path)  # clean up temp file

            progress.progress(
                int((i + 1) / len(files) * 70),
                text=f"Processed {uploaded_file.name}"
            )

        # Wipe old collection and store new batch
        progress.progress(80, text="Storing in ChromaDB...")
        embeddings = OpenAIEmbeddings()
        client = chromadb.PersistentClient(path=DB_PATH)

        try:
            client.delete_collection("langchain")  # wipe old data
        except Exception:
            pass

        Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )

        progress.progress(100, text="Done!")
        return True

    except Exception as e:
        st.error(f"❌ Error processing PDFs: {str(e)}")
        return False

# ── Trigger processing ────────────────────────────────────────────
if process_btn:
    with st.spinner(""):
        success = process_pdfs(uploaded_files)
    if success:
        total = len(uploaded_files)
        st.sidebar.success(f"✅ {total} document(s) ready! Start chatting.")
        st.cache_resource.clear()      # clear cached chain
        st.session_state.messages = [] # clear old chat history
        st.rerun()                     # restart with new data

# ── Load RAG chain ────────────────────────────────────────────────
@st.cache_resource
def load_chain():
    try:
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_chroma import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": TOP_K})
        llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, streaming=True)

        prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know the answer, say "I don't know."

Context:
{context}

Chat History:
{chat_history}

Question: {question}
""")

        def format_docs(docs: list) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "chat_history": lambda x: x["chat_history"],
                "question": lambda x: x["question"]
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain, retriever

    except Exception as e:
        st.error(f"❌ Failed to load chain: {str(e)}")
        st.stop()

# ── Guard — no documents yet ──────────────────────────────────────
if not os.path.exists(DB_PATH) or not get_active_documents():
    st.info("👈 Upload your PDFs in the sidebar and click **⚙️ Process Documents** to get started.")
    st.stop()

# ── Load chain ────────────────────────────────────────────────────
with st.spinner("🔌 Loading..."):
    chain, retriever = load_chain()

# ── Chat history ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Clear chat button ─────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Display previous messages ─────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle new input ──────────────────────────────────────────────
if question := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    chat_history = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.messages[:-1]
    )

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                chain.stream({
                    "question": question,
                    "chat_history": chat_history
                })
            )

            # Show source documents
            source_docs = retriever.invoke(question)
            if source_docs:
                with st.expander("📄 Sources"):
                    seen = set()
                    for doc in source_docs:
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", 0) + 1
                        key = f"{source}-{page}"
                        if key not in seen:
                            st.caption(f"📄 **{source}** — Page {page}")
                            seen.add(key)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {str(e)}")