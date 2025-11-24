# rag_api.py
import os
from typing import List
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import uvicorn

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "replace-me-with-secure-token")

if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY env var")

# Initialize OpenAI client (library: openai-python v4+ naming)
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Chroma (DuckDB+parquet persistence)
chroma = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"
    )
)

# Ensure collection exists
COLLECTION_NAME = "wineries"
try:
    collection = chroma.get_collection(COLLECTION_NAME)
except Exception:
    collection = chroma.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"})

app = FastAPI(title="Chroma RAG API for Napa Wineries")

# Serve static files -- use ./static folder in container; developer provided path used below in frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# Simple API key dependency
def require_auth(authorization: str = Header(None)):
    """
    Expect header: Authorization: Bearer <token>
    """
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != API_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return True


def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query_post(payload: QueryRequest, authorized: bool = Depends(require_auth)):
    # 1) embed query
    q_emb = embed_text(payload.question)

    # 2) query Chroma
    results = collection.query(query_embeddings=[q_emb], n_results=payload.top_k)

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0] if "metadatas" in results else []

    context = "\n\n---\n\n".join(docs)

    prompt = f"""You are a Napa Valley winery assistant. Use only the context below to answer the question.

Context:
{context}

Question:
{payload.question}

Answer concisely and cite which document you used (by winery name if available).
"""
    chat_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt}],
        max_tokens=300
    )

    answer = chat_resp.choices[0].message.content

    return {"answer": answer, "docs_used": docs, "metadatas": metadatas}


@app.get("/query")
def query_get(q: str, top_k: int = 3, authorized: bool = Depends(require_auth)):
    return query_post(QueryRequest(question=q, top_k=top_k), authorized=True)


# Simple frontend landing page (optional)
@app.get("/", response_class=HTMLResponse)
def index():
    html = open("static/index.html", "r", encoding="utf-8").read()
    return HTMLResponse(content=html, status_code=200)


if __name__ == "__main__":
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
