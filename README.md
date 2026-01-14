# DocumentQueryRAG

A "Chat with your PDF" application built with Python, OpenAI, and Pinecone. This tool uses Retrieval-Augmented Generation (RAG) to allow users to upload documents and ask questions about them in real-time.

#Features
- **Raw RAG Implementation:** Built without heavy frameworks (LangChain) to demonstrate deep understanding of vector search logic.
- **Dynamic Ingestion:** Users can upload PDF files which are instantly chunked, embedded, and indexed.
- **Session Isolation:** Uses Pinecone Namespaces to ensure user data remains private to their current session.
- **Tech Stack:** Python, Streamlit, OpenAI API, Pinecone Vector DB.
