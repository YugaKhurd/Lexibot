import os
import json
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# --- Config ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # optional environment name from Pinecone console
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lexibot-index")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")  # used for ServerlessSpec

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in environment")

# --- Initialize Pinecone (new client API) ---
if PINECONE_ENV:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
# pc.list_indexes() may return a response object with .names(), fall back to iterable
list_resp = pc.list_indexes()
existing_indexes = list_resp.names() if hasattr(list_resp, "names") else list(list_resp)
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # embedding size for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
    )

index = pc.Index(PINECONE_INDEX_NAME)


def load_kb(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"legal_kb.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_meta(meta):
    """
    Support shapes:
     - {"docs": [...], "metas": [...]}
     - {"data": [...]} where items are {"text":..., "meta":...} or simple strings
     - {"entries": [...]} where items contain id/act_name/section_number/title/text
     - list of texts or list of {"text":..., "meta":...}
    """
    docs = []
    metas = []

    if isinstance(meta, dict):
        if "docs" in meta and "metas" in meta:
            docs = list(meta["docs"])
            metas = list(meta["metas"])
        elif "data" in meta and isinstance(meta["data"], list):
            for item in meta["data"]:
                if isinstance(item, dict):
                    docs.append(item.get("text") or item.get("doc") or "")
                    metas.append(item.get("meta", {}))
                else:
                    docs.append(str(item))
                    metas.append({})
        elif "entries" in meta and isinstance(meta["entries"], list):
            for item in meta["entries"]:
                if isinstance(item, dict):
                    docs.append(item.get("text") or item.get("title") or "")
                    # keep id/act_name/section_number etc. as metadata
                    m = {k: v for k, v in item.items() if k not in ("text", "title")}
                    metas.append(m)
                else:
                    docs.append(str(item))
                    metas.append({})
        else:
            raise ValueError(
                "Unexpected JSON structure in legal_kb.json. Top-level keys: "
                + ", ".join(list(meta.keys()))
                + ". Expected keys: 'docs'/'metas', 'data' or 'entries'."
            )
    elif isinstance(meta, list):
        for item in meta:
            if isinstance(item, dict):
                docs.append(item.get("text") or item.get("doc") or "")
                metas.append(item.get("meta", {}))
            else:
                docs.append(str(item))
                metas.append({})
    else:
        raise ValueError("Unsupported JSON root type: " + type(meta).__name__)

    if len(docs) != len(metas):
        metas = metas + [{}] * (len(docs) - len(metas))
    return docs, metas


def embeddings_to_vectors(embs, metas):
    vectors = []
    for i, emb in enumerate(embs):
        values = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        vectors.append({"id": str(i), "values": values, "metadata": metas[i]})
    return vectors


def main():
    data_file = Path(__file__).resolve().parents[1] / "data" / "legal_kb.json"
    raw = load_kb(data_file)
    docs, metas = normalize_meta(raw)

    if not docs:
        raise RuntimeError("No documents to embed (empty docs list)")

    print(f"Encoding {len(docs)} documents with model {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(docs, show_progress_bar=True)

    vectors = embeddings_to_vectors(embs, metas)
    print(f"Uploading {len(vectors)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
    index.upsert(vectors=vectors)
    print("✅ Upload complete!")


if __name__ == "__main__":
    main()