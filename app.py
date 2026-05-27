# app.py - Terminal chat interface
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("🔌 Connecting to ChromaDB...")
embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know the answer from the context, say "I don't know."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Ready! Ask questions about your document.")
print("   Type 'exit' to quit\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        print("Goodbye!")
        break
    if not question:
        continue
    answer = chain.invoke(question)
    print(f"\n🤖 Answer: {answer}")
    print("\n" + "─"*50 + "\n")