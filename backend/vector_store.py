import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 1. SETUP EMBEDDINGS (GEMINI)
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db.faiss")
DOCS_STORE_PATH = os.path.join(BASE_DIR, "vector_db_docs.pkl")

# Global Cache
_index = None
_docs = None

# ---------- BUILD ----------
def build_vector_store(docs: list[str]):
    print("Generating embeddings... (this may take a moment)")
    doc_embeddings = embed_model.embed_documents(docs)
    emb_array = np.array(doc_embeddings, dtype="float32")

    dim = emb_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb_array)

    # Save to disk
    with open(DOCS_STORE_PATH, "wb") as f:
        pickle.dump(docs, f)
    faiss.write_index(index, VECTOR_DB_PATH)
    print(f" Vector store saved to {VECTOR_DB_PATH}")

# ---------- LOAD (Cached) ----------
def load_vector_index():
    global _index, _docs
    if _index is not None and _docs is not None:
        return _index, _docs

    if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(DOCS_STORE_PATH):
        print("⚠️ Vector DB not found. Please run build.py first.")
        return None, None

    print("Loading Vector DB from disk...")
    _index = faiss.read_index(VECTOR_DB_PATH)
    with open(DOCS_STORE_PATH, "rb") as f:
        _docs = pickle.load(f)
    return _index, _docs

# ---------- SEARCH ----------
def search(query: str, k=3):
    index, docs = load_vector_index()
    
    if not index or not docs:
        return []

    # Embed query
    q_emb = embed_model.embed_query(query)
    q_arr = np.array([q_emb], dtype="float32")

    # Search FAISS
    scores, ids = index.search(q_arr, k)

    results = []
    for i, s in zip(ids[0], scores[0]):
        if i < len(docs): # Safety check
            results.append({
                "text": docs[i],
                "score": float(s)
            })

    return results
