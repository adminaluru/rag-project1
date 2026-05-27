# streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────
MODEL_NAME: str = "gpt-3.5-turbo"
DB_PATH: str = "db"
TOP_K: int = 3
TEMPERATURE: float = 0.0

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title="Chat With Your Docs", page_icon="📄")
st.title("📄 Chat With Your Document")
st.caption("Ask anything about your uploaded PDF")

# ── API Key check ─────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()

# ── Load chain safely ─────────────────────────────────────────────
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
        return chain

    except Exception as e:
        st.error(f"❌ Failed to load chain: {str(e)}")
        st.stop()

# ── Load chain with visible status ───────────────────────────────
with st.spinner("🔌 Connecting to ChromaDB..."):
    chain = load_chain()

st.success("✅ Ready! Ask a question below.")

# ── Chat history ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display previous messages ─────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle new input ──────────────────────────────────────────────
if question := st.chat_input("Ask a question about your document..."):

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
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {str(e)}")