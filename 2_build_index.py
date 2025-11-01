import os, json, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_JSONL = os.path.join("data","posts.jsonl")
INDEX_PATH  = os.path.join("data","index.faiss")
META_PATH   = os.path.join("data","meta.pkl")
MODEL_NAME  = "all-MiniLM-L6-v2"
CHUNK_TOKENS = 400        # ~60–80 words per 100 tokens (rough)
OVERLAP_TOKENS = 80

# naive tokenizer by whitespace; fine for short social scripts
def chunk_text(text, size=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    words = text.split()
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += step
    return out

def load_docs(jsonl_path):
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(rec)
    return docs

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    docs = load_docs(DATA_JSONL)
    model = SentenceTransformer(MODEL_NAME)

    chunks, meta = [], []
    for d in docs:
        cks = chunk_text(d["text"])
        for c in cks:
            chunks.append(c)
            meta.append({"source_id": d["id"], "title": d["title"], "url": d["url"]})

    print(f"Embedding {len(chunks)} chunks…")
    emb = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle = {"chunks": chunks, "meta": meta, "model": MODEL_NAME}
        import pickle as p; p.dump(pickle, f)

    print("Saved:", INDEX_PATH, "and", META_PATH)

