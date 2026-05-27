# RAG Document Chat — AI-Powered PDF Assistant

Chat with your PDF documents using AI. Upload any PDF and ask questions 
in natural language — powered by LangChain, ChromaDB, and OpenAI GPT.

## Features
- Upload multiple PDFs directly from the browser
- AI answers questions based only on your document content
- Shows exactly which PDF and page the answer came from
- Full conversation memory across questions
- Real-time streaming responses

## Tech Stack
- Python
- LangChain
- ChromaDB (vector database)
- OpenAI GPT-3.5
- Streamlit

## How to Run
1. Clone the repo
2. pip install -r requirements.txt
3. Add your OpenAI API key to .env
4. python ingest.py
5. streamlit run streamlit_app.py
