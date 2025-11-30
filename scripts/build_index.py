# scripts/build_index_pinecone.py
import os
import json
from sentence_transformers import SentenceTransformer
import pinecone

# Paths
# KB_PATH = "../data/legal_kb.json"
KB_PATH = "data/legal_kb.json"

# Pinecone config
PINECONE_API_KEY = os.getenv("pcsk_2qo78V_BzsKLxqxdPeB9y4ctmUZAwAVfLn6pa9bu7dvdvxtoWbPs7J1j6bB9wE1aLbmvxY")
PINECONE_ENV = os.getenv("lexibot")
INDEX_NAME = "lexibot-legal"

# Load KB
with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create index if not exists
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(name=INDEX_NAME, dimension=384)  # 384 for MiniLM embeddings
index = pinecone.Index(INDEX_NAME)

# Clear index if needed
index.delete(delete_all=True)

# Prepare entries
vectors = []
metadata_list = []

for entry in kb["entries"]:
    text = entry["text"]
    # simple chunking if text is long
    chunk_size = 1000
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i+chunk_size]
        emb = model.encode(chunk_text).tolist()  # convert numpy to list
        meta = {
            "id": entry["id"],
            "act_name": entry["act_name"],
            "section_number": entry["section_number"],
            "title": entry["title"],
            "source_url": entry.get("source_url", ""),
            "chunk_index": i // chunk_size,
            "text": chunk_text
        }
        vectors.append((f"{entry['id']}_{i//chunk_size}", emb, meta))

# Upsert into Pinecone
batch_size = 50
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    ids = [v[0] for v in batch]
    embs = [v[1] for v in batch]
    metas = [v[2] for v in batch]
    index.upsert(vectors=list(zip(ids, embs, metas)))

print(f"Pinecone index '{INDEX_NAME}' ready with {len(vectors)} vectors.")
