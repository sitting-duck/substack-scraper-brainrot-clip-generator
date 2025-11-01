import os, re, pickle as pkl
import numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INDEX_PATH = os.path.join("data","index.faiss")
META_PATH  = os.path.join("data","meta.pkl")
OUT_DIR    = "generated_scripts"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = None  # will read from meta.pkl
MAX_WORDS  = 75    # ~30 seconds (60–75 words)

# starter topics — replace/extend freely or auto-generate from titles
DEFAULT_TOPICS = [
    "Explain cryonics in 30 seconds",
    "A quick case for radical life extension",
    "Quantified self: one practical habit",
    "How to think about mortality (fast take)",
    "Longevity tip most people miss",
]

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pkl.load(f)
    return index, meta

def retrieve(embedder, index, meta, query, k=6):
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q, k)
    chunks = [meta["chunks"][i] for i in I[0] if i < len(meta["chunks"])]
    cites  = [meta["meta"][i]   for i in I[0] if i < len(meta["meta"])]
    return chunks, cites

def tighten_to_words(text, max_words=MAX_WORDS):
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return text.strip()
    trimmed = " ".join(words[:max_words]).rstrip(",;:")
    if not trimmed.endswith((".", "!", "?")):
        trimmed += "."
    return trimmed

def draft_from_hits(topic, hits):
    # Heuristic: prefer first 2–3 chunks; then squeeze
    base = " ".join(hits[:3]) + f" Topic: {topic}"
    base = re.sub(r"\s+", " ", base).strip()
    # One-pass tighten
    body = tighten_to_words(base)
    # Add a simple hook + CTA for shorts
    title = re.sub(r"[^\w\s-]", "", topic).strip()
    script = f"{title}\n\n{body}\n\n— Source: Carrie Radomski on Substack"
    return script

if __name__ == "__main__":
    index, meta = load_index()
    MODEL_NAME = meta.get("model", "all-MiniLM-L6-v2")
    embedder = SentenceTransformer(MODEL_NAME)

    # Use post titles as topics too:
    title_topics = list({ m["title"] for m in meta["meta"] })[:60]  # limit for demo
    topics = DEFAULT_TOPICS + title_topics

    for t in tqdm(topics):
        hits, cites = retrieve(embedder, index, meta, t, k=6)
        script = draft_from_hits(t, hits)
        stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", t)[:120]
        (Path(OUT_DIR)/f"{stem}.txt").write_text(script, encoding="utf-8")

    print(f"Wrote {len(list(Path(OUT_DIR).glob('*.txt')))} scripts to {OUT_DIR}")

