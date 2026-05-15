from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import router
from app.dataset import load_documents
from app.embeddings import get_model, embed_documents, save_embeddings, load_embeddings
from app.vector_store import build_index, save_index, load_index
from app.clustering import (
    reduce_dimensions, fit_gmm, get_cluster_distributions,
    get_dominant_cluster, save_clustering, load_clustering,
    analyze_clusters
)
from app.cache import SemanticCache
from app.hybrid_search import BM25Index, HybridSearcher, save_bm25, load_bm25

# ─────────────────────────────────────────────────────────────────────
# App Lifespan (Startup / Shutdown context)
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Starting Semantic Search Service...\n")

    # Store objects natively on app.state
    
    # ── Step 1: Embedding model ──
    app.state.model = get_model()

    # ── Step 2: Documents + Embeddings ──
    embeddings, documents, labels = load_embeddings()

    if embeddings is None:
        print("⚙️  No cached embeddings found. Building from scratch...")
        documents, labels, label_names = load_documents()
        embeddings = embed_documents(documents, app.state.model)
        save_embeddings(embeddings, documents, labels)

    app.state.documents = documents
    app.state.embeddings = embeddings
    app.state.labels = labels

    # ── Step 3: FAISS Index (with metadata) ──
    index_data = load_index()
    if index_data is None:
        print("⚙️  No cached FAISS index found. Building...")
        index_data = build_index(embeddings, labels)
        save_index(index_data)
        
    app.state.index = index_data

    # ── Step 4: Clustering ──
    gmm, pca, cluster_probs, dominant_clusters = load_clustering()
    if gmm is None:
        print("⚙️  No cached clustering found. Fitting GMM...")
        reduced, pca = reduce_dimensions(embeddings)
        gmm = fit_gmm(reduced, n_clusters=15)
        cluster_probs = get_cluster_distributions(gmm, reduced)
        dominant_clusters = get_dominant_cluster(cluster_probs)
        save_clustering(gmm, pca, cluster_probs, dominant_clusters)
        analyze_clusters(documents, cluster_probs, dominant_clusters, embeddings)

    app.state.gmm = gmm
    app.state.pca = pca
    app.state.cluster_probs = cluster_probs
    app.state.dominant_clusters = dominant_clusters

    # ── Step 5: BM25 Index (for hybrid search) ──
    bm25 = load_bm25()
    if bm25 is None:
        print("⚙️  No cached BM25 index found. Building...")
        bm25 = BM25Index()
        bm25.fit(documents)
        save_bm25(bm25)
        
    app.state.bm25 = bm25

    # ── Step 6: Semantic Cache ──
    app.state.cache = SemanticCache(threshold=0.85)

    # ── Step 7: Hybrid Searcher ──
    app.state.hybrid_searcher = HybridSearcher(bm25_index=bm25)

    print("\n✅ Service ready.\n")

    yield  # Server runs here

    # Shutdown
    print("\n🛑 Shutting down...")
    app.state._state.clear()


# ─────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Search API: Powered by Cognitive RAG & Semantic Caching",
    description=(
        "An enterprise-grade Retrieval-Augmented Generation (RAG) backend designed for sub-millisecond context provisioning. "
        "Engineered with a dual-encoder hybrid search architecture (FAISS dense vector similarity + Okapi BM25 sparse lexical matching) "
        "and accelerated by a custom probabilistic GMM cluster-aware semantic cache to massively reduce LLM inference latency."
    ),
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(router)
