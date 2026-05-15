# Semantic Search API: Powered by Cognitive RAG & Semantic Caching

## Overview
An enterprise-grade retrieval backend designed for sub-millisecond **Retrieval-Augmented Generation (RAG)** context provisioning. 
Engineered with a dual-encoder hybrid architecture (FAISS dense vector similarity + Okapi BM25 sparse lexical matching) and accelerated by a custom probabilistic GMM cluster-aware semantic cache to massively reduce LLM inference latency.

## Architecture
```
Documents → Preprocessing Pipeline → Embeddings → FAISS Vector DB (with metadata)
                                         ↓                ↓
                                    BM25 Index    GMM Clustering
                                         ↓                ↓
                                   Hybrid Search   Semantic Cache → FastAPI
```

## Dataset Processing
- Source: 20 Newsgroups (sklearn built-in)
- Removed: headers, footers, quotes (email metadata — not semantically meaningful)
- Subset: 5000 documents (preserves topic diversity, allows rapid iteration)
- **Multi-stage preprocessing pipeline:**
  - Stage 1: Regex cleaning — strip URLs, emails, file paths, control chars, normalise whitespace
  - Stage 2: Length filtering — remove sub-50-char documents (empty after preprocessing)
  - Stage 3: Near-duplicate removal — trigram fingerprinting to catch cross-posts
  - Preprocessing stats printed at startup showing removals at each stage

## Embedding Model
**Model:** `all-MiniLM-L6-v2` (sentence-transformers)

| Property | Value |
|---|---|
| Embedding size | 384 dimensions |
| Speed | ~14,000 sentences/sec on CPU |
| Normalisation | L2-normalised (cosine sim = dot product) |

Chosen for its balance of speed and semantic quality. Larger models
(e.g. all-mpnet-base-v2) give marginally better accuracy at 2-3x latency cost.

## Vector Database
**Library:** FAISS (`IndexIDMap` wrapping `IndexFlatIP`)

- Inner product index on L2-normalised vectors = cosine similarity search
- Exhaustive search — correct for 5000 docs (<1ms per query)
- No server required — runs fully in-process
- **Metadata-aware:** each vector stores category label for filtered retrieval
- **Filtered search:** post-filtering strategy with 3x over-fetch, <1ms
- Persisted to disk after first build

## Hybrid Search (BM25 + Dense)
**BM25 built from scratch** — no external library, full Okapi BM25 implementation.

- **Dense (FAISS):** captures semantic meaning ("rocket" ≈ "propulsion")
- **Sparse (BM25):** captures keyword matches ("SCSI" = "SCSI")
- **Fusion:** `final = α × dense + (1-α) × bm25` with default α=0.7
- Score normalisation: dense already [0,1], BM25 min-max normalised
- Tunable α at query time via the `/hybrid-query` endpoint

## Clustering
**Model:** Gaussian Mixture Model (GMM)

- **Why GMM:** Produces soft probability distributions, not hard labels.
  A document about gun legislation gets `[0.6 politics, 0.3 firearms, 0.1 law]`.
- **Why not K-Means:** Hard assignments only — violates the requirement.
- **Dimensions:** PCA to 50D before GMM (stabilises covariance estimation)
- **N clusters:** 15 (chosen by BIC score — best fit/complexity balance)
- **Covariance:** diagonal (prevents overfitting on 5000 documents)

## Semantic Cache
Built entirely from scratch — no Redis, no Memcached, no caching library.

**Data structure:** Python list of `CacheEntry` objects + dict cluster index

**Lookup strategy:**
1. Embed incoming query
2. Find dominant cluster
3. Search only cache entries in same cluster (cluster-aware fast path)
4. Compute cosine similarity (dot product on normalised vectors)
5. Return best match if similarity ≥ threshold

**Similarity Threshold:**

| Threshold | Behaviour | Use Case |
|---|---|---|
| 0.95 | Near-verbatim only | Legal / medical precision |
| 0.85 | Same meaning, different words | General search (default) |
| 0.75 | Same topic, different angle | Recommendations |
| 0.70 | Broad topic overlap | Content discovery |

## Running the Project

### Option 1: Quick Start (Windows)
The easiest way to start the service in an isolated virtual environment:
```powershell
.\setup.ps1
```
This script automatically:
1. Creates a `.venv`
2. Installs `requirements.txt`
3. Starts the `uvicorn` server on port 8000

Alternatively, run manually:
```bash
python -m venv .venv
# Activate venv (.venv\Scripts\Activate on Windows, source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option 2: Docker Compose (Production Ready)
The service is containerised securely running as a non-root user.
```bash
docker-compose up --build
```
This mounts the `data` directory as a volume so the FAISS index and clustering models don't need to rebuild if the container restarts.

## API Endpoints

### POST /query (Dense Search)
```json
{
  "query": "how does nasa launch rockets"
}
```
Response:
```json
{
  "query": "how does nasa launch rockets",
  "cache_hit": true,
  "matched_query": "NASA rocket launch procedure",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3,
  "search_mode": "dense"
}
```

### POST /hybrid-query (Hybrid BM25 + Dense)
```json
{
  "query": "SCSI hard drive controller",
  "alpha": 0.7
}
```
Response includes score breakdown showing dense vs sparse contribution.

### POST /filtered-query?category=14 (Category-Filtered)
```json
{
  "query": "space shuttle launch"
}
```
Results restricted to `/filtered-query?category=14` (sci.space).

### GET /cache/stats
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "threshold": 0.85
}
```

### DELETE /cache
Flushes cache and resets all stats.

### PATCH /cache/threshold?threshold=0.90
Updates similarity threshold at runtime without restarting.

### GET /health
Quick health check with system status.

## Threshold Analysis
```bash
python -m experiments.threshold_analysis
```
Explores how threshold value affects cache hit rate and result correctness
across a controlled set of semantically related query pairs.
