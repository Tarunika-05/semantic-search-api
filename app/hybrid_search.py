import numpy as np
import pickle
import os
import re
from math import log

# ─────────────────────────────────────────────────────────────────────
# Hybrid Search: BM25 (Sparse) + Dense (FAISS) Score Fusion
#
# Why hybrid search matters:
#
# Pure semantic (dense) search excels at understanding meaning:
#   "How do rockets work?" ≈ "explain propulsion systems" → HIGH similarity
#
# But it fails on exact-match keywords and proper nouns:
#   "SCSI controller" vs "Small Computer System Interface" → LOW similarity
#   (The model may not associate the acronym with the full term)
#
# BM25 (sparse keyword matching) handles these cases perfectly because
# it matches on exact token overlap. The trade-off:
#   - BM25 misses semantic paraphrasing ("car" ≠ "automobile")
#   - Dense search misses keyword specificity ("SCSI" ≠ "SCSI")
#
# Hybrid search combines both: final = α × dense + (1-α) × bm25
#   - α = 0.7 (default): favours semantics, supplements with keywords
#   - This covers both failure modes with minimal accuracy loss on either
#
# Implementation: I build BM25 from scratch using standard TF-IDF-like
# scoring (Okapi BM25) rather than importing a library. This keeps the
# codebase self-contained and demonstrates understanding of the algorithm.
# ─────────────────────────────────────────────────────────────────────

BM25_INDEX_PATH = "data/bm25_index.pkl"

# BM25 hyperparameters — standard Okapi BM25 values.
# k1 controls term frequency saturation: higher = longer docs get higher scores.
# b controls length normalisation: 0 = no normalisation, 1 = full normalisation.
# These are the widely-accepted defaults from the original Robertson et al. paper.
BM25_K1 = 1.5
BM25_B = 0.75

# Default fusion weight: how much to favour dense (semantic) over sparse (keyword).
# 0.7 means 70% semantic, 30% keyword — appropriate for a semantic search system
# where meaning matters more than exact wording, but keywords still help.
DEFAULT_ALPHA = 0.7


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.

    Why not NLTK or spaCy?
    - I need fast, lightweight tokenization for BM25, not linguistic analysis
    - BM25 works on raw tokens — stemming and lemmatization actually hurt
      performance on technical corpora (newsgroups) because acronyms and
      technical terms get mangled (e.g. "SCSI" → "scsi" is fine, but
      stemming "graphics" → "graphic" loses the plural distinction)
    - A simple regex split on non-alphanumeric characters is sufficient

    Returns:
        List of lowercase tokens, minimum 2 characters each.
    """
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Filter tokens < 2 chars — single characters are almost never
    # meaningful search terms and inflate the vocabulary unnecessarily.
    return [t for t in tokens if len(t) >= 2]


class BM25Index:
    """
    Okapi BM25 sparse index built from scratch.

    BM25 scoring formula per document d for query q:
        score(q, d) = Σ IDF(qi) × (tf(qi, d) × (k1 + 1)) / (tf(qi, d) + k1 × (1 - b + b × |d|/avgdl))

    Where:
        tf(qi, d)  = term frequency of query term qi in document d
        IDF(qi)    = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
        N          = total documents
        df(qi)     = document frequency of term qi
        |d|        = length of document d (in tokens)
        avgdl      = average document length across corpus

    This gives higher scores to:
    - Documents containing rare query terms (IDF weighting)
    - Documents with higher term frequency (with saturation via k1)
    - Shorter documents containing the terms (length normalisation via b)
    """

    def __init__(self):
        self.doc_term_freqs = []   # list of {term: count} per document
        self.doc_lengths = []       # token count per document
        self.avg_doc_length = 0.0
        self.n_docs = 0
        self.df = {}               # document frequency: term → count of docs containing it
        self.idf = {}              # pre-computed IDF scores

    def fit(self, documents: list[str]):
        """
        Build the BM25 index from a list of documents.

        Args:
            documents: list of raw text strings (already preprocessed by dataset.py)
        """
        print(f"🔄 Building BM25 sparse index over {len(documents)} documents...")

        self.n_docs = len(documents)
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.df = {}

        for doc in documents:
            tokens = _tokenize(doc)
            self.doc_lengths.append(len(tokens))

            # Compute term frequencies for this document
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)

            # Update document frequencies (count each term once per doc)
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths) / max(self.n_docs, 1)

        # Pre-compute IDF for all terms — avoids repeated computation at query time.
        # Using the BM25+ variant IDF formula which adds 1 inside the log to prevent
        # negative IDF values for very common terms.
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

        vocab_size = len(self.df)
        print(f"✅ BM25 index built. Vocabulary: {vocab_size} terms, Avg doc length: {self.avg_doc_length:.1f} tokens")

    def score(self, query: str, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Score all documents against a query using BM25.

        Args:
            query:  raw query string
            top_k:  number of top results to return

        Returns:
            scores:  numpy array of BM25 scores for top_k documents
            indices: numpy array of document indices for top_k documents
        """
        query_tokens = _tokenize(query)
        scores = np.zeros(self.n_docs, dtype=np.float64)

        for term in query_tokens:
            if term not in self.idf:
                continue  # Query term not in corpus — skip

            idf = self.idf[term]

            for doc_idx in range(self.n_docs):
                tf = self.doc_term_freqs[doc_idx].get(term, 0)
                if tf == 0:
                    continue  # Term not in this document — skip

                doc_len = self.doc_lengths[doc_idx]

                # Okapi BM25 scoring formula
                numerator = tf * (BM25_K1 + 1)
                denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self.avg_doc_length)
                scores[doc_idx] += idf * numerator / denominator

        # Get top-k indices sorted by score (descending)
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]

        return top_scores, top_indices


class HybridSearcher:
    """
    Combines dense (FAISS) and sparse (BM25) search with score fusion.

    Score fusion strategy: linear interpolation.
        final_score = α × normalised_dense + (1 - α) × normalised_bm25

    Why linear interpolation over other fusion methods (e.g. RRF)?
    - Reciprocal Rank Fusion (RRF) only uses rank positions, discarding
      the actual similarity/BM25 scores. This loses information.
    - Linear interpolation preserves score magnitudes, letting truly
      high-confidence dense matches dominate when appropriate.
    - The α parameter is interpretable: α=1.0 → pure dense, α=0.0 → pure BM25.
    - Easy to tune at runtime via the API.

    Score normalisation:
    - Dense scores (cosine similarity) are already in [0, 1] for normalised vectors
    - BM25 scores vary wildly (0 to 50+), so I min-max normalise to [0, 1]
    - This ensures both signals contribute proportionally
    """

    def __init__(self, bm25_index: BM25Index, alpha: float = DEFAULT_ALPHA):
        self.bm25 = bm25_index
        self.alpha = alpha

    def search(self, query: str, query_embedding: np.ndarray,
               faiss_index, documents: list[str], top_k: int = 5) -> tuple[list, list, list]:
        """
        Perform hybrid search combining dense and sparse signals.

        Args:
            query:           raw query string (for BM25)
            query_embedding: L2-normalised query vector (for FAISS)
            faiss_index:     FAISS index or dict with 'index' key
            documents:       list of document texts
            top_k:           number of results to return

        Returns:
            final_indices:  document indices sorted by hybrid score
            final_scores:   hybrid scores for each result
            score_details:  list of dicts with dense/sparse/hybrid breakdown
        """
        from app.vector_store import search_index

        # ── Dense search (FAISS) ──
        # Over-fetch to have enough candidates for fusion
        n_candidates = min(top_k * 3, len(documents))
        dense_distances, dense_indices = search_index(faiss_index, query_embedding, n_candidates)

        # ── Sparse search (BM25) ──
        bm25_scores, bm25_indices = self.bm25.score(query, n_candidates)

        # ── Build unified candidate set ──
        # Merge candidates from both retrievers to avoid missing documents
        # that rank highly in one system but not the other.
        candidates = set()
        for idx in dense_indices:
            if idx >= 0:
                candidates.add(int(idx))
        for idx in bm25_indices:
            if idx >= 0:
                candidates.add(int(idx))

        # ── Score normalisation ──
        # Dense scores: cosine similarity already in [-1, 1], shift to [0, 1]
        dense_score_map = {}
        for dist, idx in zip(dense_distances, dense_indices):
            if idx >= 0:
                dense_score_map[int(idx)] = float((dist + 1) / 2)  # [-1,1] → [0,1]

        # BM25 scores: min-max normalise to [0, 1]
        bm25_score_map = {}
        bm25_max = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        bm25_min = min(bm25_scores) if len(bm25_scores) > 0 else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
        for score, idx in zip(bm25_scores, bm25_indices):
            if idx >= 0:
                bm25_score_map[int(idx)] = float((score - bm25_min) / bm25_range)

        # ── Fusion ──
        scored_candidates = []
        for idx in candidates:
            dense_s = dense_score_map.get(idx, 0.0)
            sparse_s = bm25_score_map.get(idx, 0.0)
            hybrid = self.alpha * dense_s + (1 - self.alpha) * sparse_s

            scored_candidates.append({
                "index": idx,
                "dense_score": round(dense_s, 4),
                "sparse_score": round(sparse_s, 4),
                "hybrid_score": round(hybrid, 4),
            })

        # Sort by hybrid score descending
        scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_results = scored_candidates[:top_k]

        final_indices = [r["index"] for r in top_results]
        final_scores = [r["hybrid_score"] for r in top_results]

        return final_indices, final_scores, top_results


def save_bm25(bm25_index: BM25Index):
    """Persist BM25 index to disk."""
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"✅ BM25 index saved to {BM25_INDEX_PATH}")


def load_bm25() -> BM25Index | None:
    """Load BM25 index from disk if available."""
    if not os.path.exists(BM25_INDEX_PATH):
        return None
    print("📂 Loading cached BM25 index from disk...")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    print(f"✅ BM25 index loaded. Vocabulary: {len(bm25.df)} terms")
    return bm25
