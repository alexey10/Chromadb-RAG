# ingest.py
"""
Usage:
  - To ingest CSV: python ingest.py --csv path/to/wineries.csv
  - To ingest a website page: python ingest.py --url https://example.com/wineries
CSV should contain columns: id,name,description,location,price_range,reservations (optional)
"""
import argparse
import os
import pandas as pd
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("set OPENAI_API_KEY env var")

client = OpenAI(api_key=OPENAI_API_KEY)

chroma = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = chroma.get_or_create_collection(name="wineries", metadata={"hnsw:space":"cosine"})

def embed(text):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

def ingest_csv(path):
    df = pd.read_csv(path)
    docs, ids, metadatas, embeddings = [], [], [], []
    for _, row in df.iterrows():
        doc_text = row.get("description") or ""
        # build a canonical document
        doc = f"{row.get('name','')} in {row.get('location','')}. {doc_text} Price: {row.get('price_range','')}. Reservations: {row.get('reservations','')}"
        ids.append(str(row.get("id") or row.get("name")))
        docs.append(doc)
        metadatas.append({"name": row.get("name"), "location": row.get("location")})
        embeddings.append(embed(doc))
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    print(f"Ingested {len(ids)} records from CSV.")

def ingest_url(url):
    r = requests.get(url, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    # crude example: take all paragraphs
    paras = soup.find_all("p")
    text = "\n\n".join(p.get_text().strip() for p in paras if p.get_text().strip())
    doc = f"Content from {url}\n\n{text}"
    collection.add(ids=[url], documents=[doc], metadatas=[{"source": url}], embeddings=[embed(doc)])
    print("Ingested page:", url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to CSV")
    parser.add_argument("--url", help="URL to scrape")
    args = parser.parse_args()
    if args.csv:
        ingest_csv(args.csv)
    if args.url:
        ingest_url(args.url)
