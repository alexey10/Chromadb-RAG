Developer/User
    |
    | Query text
    v
Frontend / API Gateway
    |
    | send raw query
    v
Query Service (FastAPI)
    |
    | embed(query)
    v
Embedding Model (OpenAI / HuggingFace / Instructor-xl)
    |
    | returns query vector
    v
Vector DB (Chroma / Pinecone / Weaviate)
    |
    | nearest neighbor search
    v
RAG Orchestrator
    |
    | fetch doc chunks
    v
LLM (OpenAI / Llama)
    |
    | synthesized answer
    v
API Response
