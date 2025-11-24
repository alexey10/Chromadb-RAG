import chromadb
from chromadb.config import Settings
from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_KEY")

chroma = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                  persist_directory="./chroma_db"))

collection = chroma.get_or_create_collection(
    name="wineries",
    metadata={"hnsw:space": "cosine"}
)

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# Example winery documents
docs = [
    {
        "id": "winery_001",
        "text": (
            "Stag’s Leap Wine Cellars in Oakville specializes in Cabernet "
            "Sauvignon with tastings including the Estate Cabernet Experience "
            "and Cave Tour & Pairing. Reservations required. Price $80–$250."
        )
    },
    {
        "id": "winery_002",
        "text": (
            "Beringer Vineyards in St. Helena offers Chardonnay and Cabernet "
            "tastings like the Rhine House Tasting and Reserve Pairings. No reservation required. "
            "Price $25–$120."
        )
    }
]

collection.add(
    ids=[d["id"] for d in docs],
    documents=[d["text"] for d in docs],
    embeddings=[embed(d["text"]) for d in docs]
)
