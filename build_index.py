# build_index.py
import json
from sentence_transformers import SentenceTransformer
import chromadb

with open("chunks.json") as f:
    chunks = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")   # persists to disk
collection = client.get_or_create_collection("research_papers")

texts = [c["text"] for c in chunks]
embeddings = model.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"paper": c["paper"]} for c in chunks],
    ids=[c["id"] for c in chunks],
)

print(f"Indexed {len(chunks)} chunks into ./chroma_store")