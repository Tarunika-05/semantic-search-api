from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np

# Create the router
router = APIRouter()

# ─────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: str
    dominant_cluster: int
    search_mode: str = "dense"  # "dense" | "hybrid"


class HybridQueryRequest(BaseModel):
    query: str
    alpha: float = 0.7  # fusion weight: 0.0 = pure BM25, 1.0 = pure dense


# ─────────────────────────────────────────────────────────────────────
# Handlers (Functions previously inside main.py)
# ─────────────────────────────────────────────────────────────────────

def process_query(request: Request, query: str):
    """
    Embed a query string and determine its dominant cluster.
    """
    state = request.app.state
    model = state.model
    pca = state.pca
    gmm = state.gmm

    # Embed (1, 384)
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Reduce for clustering (1, 50)
    query_reduced = pca.transform([query_embedding])

    # Soft cluster assignment (15,)
    cluster_probs = gmm.predict_proba(query_reduced)[0]
    dominant_cluster = int(np.argmax(cluster_probs))

    return query_embedding, dominant_cluster, cluster_probs


def get_result_from_corpus(request: Request, query_embedding: np.ndarray, category_filter: int | None = None) -> str:
    """
    Search FAISS index and return the most relevant document snippet.
    """
    from app.vector_store import search_index, search_with_filter
    
    state = request.app.state
    index_data = state.index
    documents = state.documents

    if category_filter is not None:
        distances, indices = search_with_filter(
            index_data, query_embedding, category_filter=category_filter, top_k=3
        )
    else:
        distances, indices = search_index(index_data, query_embedding, top_k=3)

    if len(indices) == 0:
        return "No matching documents found."

    # Build result from top match
    top_doc = documents[indices[0]]
    result = top_doc[:500].strip()
    return result


def get_hybrid_result(request: Request, query: str, query_embedding: np.ndarray, alpha: float = 0.7) -> tuple:
    """
    Search using hybrid (BM25 + Dense) scoring.
    """
    state = request.app.state
    hybrid = state.hybrid_searcher
    hybrid.alpha = alpha
    documents = state.documents
    index_data = state.index

    indices, scores, details = hybrid.search(
        query=query,
        query_embedding=query_embedding,
        faiss_index=index_data,
        documents=documents,
        top_k=3
    )

    if len(indices) == 0:
        return "No matching documents found.", []

    result = documents[indices[0]][:500].strip()
    return result, details


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = payload.query.strip()
    cache = request.app.state.cache

    query_embedding, dominant_cluster, cluster_probs = process_query(request, query)

    cached_entry = cache.lookup(query_embedding, dominant_cluster, cluster_probs)

    if cached_entry is not None:
        similarity = float(np.dot(query_embedding, cached_entry.embedding))
        return QueryResponse(
            query=query,
            cache_hit=True,
            matched_query=cached_entry.query,
            similarity_score=round(similarity, 4),
            result=cached_entry.result,
            dominant_cluster=dominant_cluster,
            search_mode="dense"
        )

    result = get_result_from_corpus(request, query_embedding)

    cache.store(
        query=query,
        embedding=query_embedding,
        result=result,
        dominant_cluster=dominant_cluster,
        cluster_probs=cluster_probs
    )

    return QueryResponse(
        query=query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        dominant_cluster=dominant_cluster,
        search_mode="dense"
    )


@router.post("/hybrid-query")
async def hybrid_query_endpoint(request: Request, payload: HybridQueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not (0.0 <= payload.alpha <= 1.0):
        raise HTTPException(status_code=400, detail="Alpha must be between 0 and 1.")

    query = payload.query.strip()
    query_embedding, dominant_cluster, _ = process_query(request, query)

    result, score_details = get_hybrid_result(request, query, query_embedding, alpha=payload.alpha)

    return {
        "query": query,
        "result": result,
        "search_mode": "hybrid",
        "alpha": payload.alpha,
        "dominant_cluster": dominant_cluster,
        "score_breakdown": score_details[:3]
    }


@router.post("/filtered-query")
async def filtered_query_endpoint(request: Request, payload: QueryRequest, category: int = Query(None, ge=0, le=19)):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = payload.query.strip()
    query_embedding, dominant_cluster, _ = process_query(request, query)
    result = get_result_from_corpus(request, query_embedding, category_filter=category)

    return {
        "query": query,
        "result": result,
        "search_mode": "filtered",
        "category_filter": category,
        "dominant_cluster": dominant_cluster,
    }


@router.get("/cache/stats")
async def cache_stats(request: Request):
    return request.app.state.cache.get_stats()


@router.delete("/cache")
async def clear_cache(request: Request):
    request.app.state.cache.flush()
    return {"message": "Cache cleared successfully.", "status": "ok"}


@router.patch("/cache/threshold")
async def update_threshold(request: Request, threshold: float):
    if not (0.0 < threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")
    request.app.state.cache.set_threshold(threshold)
    return {"message": f"Threshold updated to {threshold}", "threshold": threshold}


@router.get("/clusters/analysis")
async def cluster_analysis(request: Request):
    from app.clustering import get_full_analysis
    state = request.app.state
    return get_full_analysis(
        documents=state.documents,
        cluster_probs=state.cluster_probs,
        dominant_clusters=state.dominant_clusters,
        embeddings=state.embeddings,
    )


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@router.get("/health")
async def health(request: Request):
    state = request.app.state
    return {
        "status": "ok",
        "documents_loaded": len(getattr(state, "documents", [])),
        "cache_entries": len(getattr(state, "cache", [])),
        "bm25_vocab_size": len(getattr(state, "bm25", __import__("app.hybrid_search", fromlist=["BM25Index"]).BM25Index()).df),
        "search_modes": ["dense", "hybrid", "filtered"],
    }
