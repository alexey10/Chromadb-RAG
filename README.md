# Chromadb-RAG
RAG with Chroma DB
pip install chromadb openai

# RAG Embedding Flow
sequenceDiagram
    participant User
    participant Frontend UI
    participant API Server (FastAPI)
    participant Embedding Model
    participant Vector DB (e.g., Chroma)
    participant LLM

    User ->> Frontend UI: Enters query
    Frontend UI ->> API Server (FastAPI): Send query text
    API Server (FastAPI) ->> Embedding Model: Generate embedding for query
    Embedding Model ->> API Server (FastAPI): Return embedding vector
    API Server (FastAPI) ->> Vector DB: Vector similarity search using embedding
    Vector DB ->> API Server (FastAPI): Return top matching documents
    API Server (FastAPI) ->> LLM: Provide matches + original query
    LLM ->> API Server (FastAPI): RAG answer generated
    API Server (FastAPI) ->> Frontend UI: Return final answer
    Frontend UI ->> User: Display result
