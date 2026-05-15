import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle

# ─────────────────────────────────────────────────────────────────────
# Embedding Model Choice: all-MiniLM-L6-v2
#
# Why this model:
# 1. Speed — it is one of the fastest sentence-transformer models,
#    critical for a cache system where query embedding must be near
#    real-time (< 50ms on CPU).
# 2. Quality — despite its small size, it performs strongly on
#    semantic textual similarity benchmarks (STSB, SICK-R).
# 3. Embedding size — 384 dimensions. Small enough for efficient
#    FAISS indexing and cosine similarity in the cache layer, but
#    rich enough to capture nuanced topic relationships.
# 4. No GPU required — runs well on CPU, making it portable and
#    easy to deploy without special hardware.
#
# Trade-off acknowledged:
# Larger models (e.g. all-mpnet-base-v2, 768-dim) would give better
# semantic accuracy but at 2-3x the latency. For this use case,
# MiniLM is the right balance.
# ─────────────────────────────────────────────────────────────────────

EMBEDDINGS_CACHE_PATH = "data/embeddings_cache.pkl"

def get_model():
    """Load the sentence transformer model."""
    print("🔄 Loading embedding model: all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model loaded.")
    return model


def embed_documents(documents: list[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of documents into embeddings.

    Args:
        documents:  List of raw text strings.
        model:      Loaded SentenceTransformer model.
        batch_size: Number of documents encoded per batch.
                    64 is a good default — large enough to use
                    vectorised ops, small enough to avoid OOM on CPU.

    Returns:
        numpy array of shape (n_documents, 384)
    """
    print(f"🔄 Embedding {len(documents)} documents in batches of {batch_size}...")

    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize so cosine similarity = dot product
                                   # This is important for the cache layer —
                                   # normalized vectors make similarity comparisons
                                   # faster and more numerically stable.
    )

    print(f"✅ Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, documents: list, labels: list):
    """
    Persist embeddings to disk so I don't recompute on every startup.
    Recomputing 5000 embeddings takes ~30-60s on CPU — caching this
    is essential for a fast development loop.
    """
    with open(EMBEDDINGS_CACHE_PATH, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "documents": documents,
            "labels": labels
        }, f)
    print(f"✅ Embeddings saved to {EMBEDDINGS_CACHE_PATH}")


def load_embeddings():
    """
    Load embeddings from disk if available.
    Returns None if not found.
    """
    if not os.path.exists(EMBEDDINGS_CACHE_PATH):
        return None, None, None

    print("📂 Loading cached embeddings from disk...")
    with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    print(f"✅ Loaded {len(data['documents'])} cached embeddings.")
    return data["embeddings"], data["documents"], data["labels"]
