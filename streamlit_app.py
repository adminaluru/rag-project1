# streamlit_app.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────
MODEL_NAME: str = "gpt-3.5-turbo"
DB_PATH: str = "db"
TOP_K: int = 5
TEMPERATURE: float = 0.0

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Hide default Streamlit header decoration */
#MainMenu { visibility: hidden; }

/* Footer */
footer {
    visibility: visible;
    background-color: #b7cece;
    color: #666;
}

/* App background */
.stApp {
    background-color: #96a6aa;
}

/* Header banner — soft warm gradient */
.header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(102,126,234,0.3);
}
.header-banner h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
}
.header-banner p {
    color: rgba(255,255,255,0.82);
    font-size: 0.95rem;
    margin: 0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #a790a5;
    border-right: 1px solid #8f7490;
}
[data-testid="stSidebar"] .stButton button {
    width: 100%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: opacity 0.2s;
    box-shadow: 0 2px 8px rgba(102,126,234,0.3);
}
[data-testid="stSidebar"] .stButton button:hover {
    opacity: 0.88;
}

/* Document card in sidebar */
.doc-card {
    background: #faf7ff;
    border: 1px solid #e0d7f5;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.doc-card-icon { font-size: 1.2rem; }
.doc-card-name {
    color: #4a4060;
    font-size: 0.82rem;
    word-break: break-all;
}

/* Stat pills */
.stat-row {
    display: flex;
    gap: 10px;
    margin: 12px 0;
}
.stat-pill {
    background: #ede9fe;
    border: 1px solid #c4b5fd;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #6d28d9;
    font-weight: 600;
}

/* Chat message overrides */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 4px 8px;
    margin-bottom: 6px;
}

/* User message */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #ede9fe;
}

/* Source expander */
[data-testid="stExpander"] {
    background: #faf7ff;
    border: 1px solid #e0d7f5;
    border-radius: 10px;
}

/* Source chip */
.source-chip {
    display: inline-block;
    background: #ede9fe;
    border: 1px solid #c4b5fd;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #6d28d9;
    margin: 3px 4px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 80px 40px;
    color: #9ca3af;
}
.empty-state h2 { color: #6b7280; font-size: 1.4rem; }
.empty-state p  { font-size: 0.95rem; margin-top: 8px; }

/* Clear chat button */
div[data-testid="stHorizontalBlock"] .stButton button {
    background: transparent;
    border: 1px solid #d1c4e9;
    color: #9e86c8;
    font-size: 0.8rem;
    padding: 4px 14px;
    border-radius: 20px;
}
div[data-testid="stHorizontalBlock"] .stButton button:hover {
    border-color: #ef4444;
    color: #ef4444;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🧠 DocChat AI</h1>
  <p>Upload your PDFs and have an intelligent conversation with your documents.</p>
</div>
""", unsafe_allow_html=True)

# ── API Key check ─────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()

# ── Helper: Get active documents from ChromaDB ───────────────────
def get_active_documents() -> list:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("langchain")
        results = collection.get(include=["metadatas"])
        sources = set(m.get("source", "Unknown") for m in results["metadatas"])
        return sorted(sources)
    except Exception:
        return []

def get_chunk_count() -> int:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("langchain")
        return collection.count()
    except Exception:
        return 0

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### <span style='color:#f9a8d4'>📁 Upload Documents</span>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag & drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for f in uploaded_files:
            st.markdown(f"""
            <div class="doc-card">
                <span class="doc-card-icon">📄</span>
                <span class="doc-card-name">{f.name}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    process_btn = st.button(
        "⚙️ Process Documents",
        disabled=not uploaded_files,
        use_container_width=True
    )

    st.divider()

    # Active documents section
    active_docs = get_active_documents()
    chunk_count = get_chunk_count()

    st.markdown("### <span style='color:#f9a8d4'>🗂️ Knowledge Base</span>", unsafe_allow_html=True)

    if active_docs:
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">📄 {len(active_docs)} docs</div>
            <div class="stat-pill">🧩 {chunk_count} chunks</div>
        </div>""", unsafe_allow_html=True)

        for doc in active_docs:
            name = os.path.basename(doc)
            st.markdown(f"""
            <div class="doc-card">
                <span class="doc-card-icon">✅</span>
                <span class="doc-card-name">{name}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No documents loaded yet.")
        st.caption("Upload PDFs above and click **Process**.")

# ── Process uploaded PDFs ─────────────────────────────────────────
def process_pdfs(files) -> bool:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        import chromadb

        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        progress = st.progress(0, text="Reading PDFs...")

        for i, uploaded_file in enumerate(files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            os.unlink(tmp_path)

            progress.progress(
                int((i + 1) / len(files) * 70),
                text=f"Processed: {uploaded_file.name}"
            )

        progress.progress(80, text="Building embeddings...")
        embeddings = OpenAIEmbeddings()
        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            client.delete_collection("langchain")
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
        st.sidebar.success(f"✅ {len(uploaded_files)} document(s) ready!")
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

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
    st.markdown("""
    <div class="empty-state">
        <div style="font-size: 3.5rem;">📂</div>
        <h2>No documents loaded</h2>
        <p>Upload your PDFs in the sidebar and click <strong>Process Documents</strong> to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load chain ────────────────────────────────────────────────────
with st.spinner("Loading knowledge base..."):
    chain, retriever = load_chain()

# ── Chat history ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Toolbar row ───────────────────────────────────────────────────
if st.session_state.messages:
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.messages = []
            st.rerun()

# ── Welcome message when chat is empty ───────────────────────────
if not st.session_state.messages:
    active = get_active_documents()
    names = ", ".join(os.path.basename(d) for d in active[:3])
    if len(active) > 3:
        names += f" and {len(active) - 3} more"
    st.markdown(f"""
    <div style="text-align:center; padding: 40px 20px;">
        <div style="font-size: 2.5rem;">💬</div>
        <p style="color: #7c6b9e; margin-top: 10px;">
            Ready to answer questions about <strong style="color:#764ba2">{names}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Display previous messages ─────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle new input ──────────────────────────────────────────────
if question := st.chat_input("Ask anything about your documents..."):

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

            source_docs = retriever.invoke(question)
            if source_docs:
                seen = set()
                chips = []
                for doc in source_docs:
                    source = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", 0) + 1
                    key = f"{source}-{page}"
                    if key not in seen:
                        chips.append(f'<span class="source-chip">📄 {source} · p{page}</span>')
                        seen.add(key)

                with st.expander("📎 Sources", expanded=False):
                    st.markdown(" ".join(chips), unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {str(e)}")
