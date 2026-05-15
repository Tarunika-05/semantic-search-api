import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import time

# ─────────────────────────────────────────────────────────────────────
# Semantic Cache — Built From Scratch
#
# A traditional cache uses exact string matching:
#   "how do rockets work" ≠ "explain rocket propulsion"
# Both questions want the same answer. A string cache misses this.
#
# my semantic cache uses cosine similarity between query embeddings:
#   embed("how do rockets work") ≈ embed("explain rocket propulsion")
#   similarity = 0.91 → cache HIT
#
# Data structure: a plain Python list of CacheEntry objects.
# No Redis, no Memcached, no external library. Just list + numpy.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Cluster-Aware Multi-Cluster Lookup
#
# The basic approach: search only the dominant cluster.
# The problem: a query about "gun legislation" has dominant_cluster=3
# (politics), but a cached query about "firearm regulations" might
# sit in cluster=7 (firearms). Same semantic intent, different clusters.
#
# my improvement: use the query's SOFT cluster probabilities to
# search the top-N clusters by probability, not just the dominant one.
# This directly leverages Part 2's fuzzy clustering output.
#
# Example:
#   Query cluster probs: [... 0.45 politics, 0.35 firearms, 0.12 law ...]
#   I search clusters: politics(0.45), firearms(0.35), law(0.12)
#   Total comparisons: ~150 (across 3 clusters) vs 1000 (naive full scan)
#   Result: catches cross-cluster matches without falling back to O(n)
#
# The number of clusters to search (MAX_LOOKUP_CLUSTERS) is a secondary
# tunable: 3 covers >90% of probability mass for most queries.
# ─────────────────────────────────────────────────────────────────────

MAX_LOOKUP_CLUSTERS = 3  # Search this many top clusters during lookup

# ─────────────────────────────────────────────────────────────────────
# Similarity Threshold — The Core Tunable Decision
#
# This is the single most important parameter in the entire system.
# It defines what the system considers "the same question".
#
# threshold = 0.95 → very strict
#   Only nearly identical queries hit the cache.
#   Safe but defeats the purpose of semantic caching.
#   Use case: legal/medical systems where a wrong cached result is costly.
#
# threshold = 0.85 → balanced (my default)
#   Queries phrased differently but meaning the same thing hit.
#   "space shuttle launch" ↔ "NASA rocket takeoff" → HIT (sim ≈ 0.91)
#   Good balance for general semantic search.
#
# threshold = 0.75 → loose
#   Same topic, different angle triggers a hit.
#   Risk: semantically related but distinct queries get same result.
#   Use case: recommendations, content discovery.
#
# threshold = 0.70 → too loose
#   Broad topic overlap causes false positives.
#   "python programming" ↔ "snake biology" might hit.
#   Breaks result correctness for most use cases.
#
# The insight: threshold is NOT a performance dial. It is a definition
# of what "the same question" means to your system. Choosing it requires
# understanding user intent, not just measuring hit rates.
# ─────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.85

# ─────────────────────────────────────────────────────────────────────
# TTL (Time-To-Live) and LRU (Least Recently Used) Eviction
#
# A real cache must manage its own size. Without eviction:
# - Memory grows unbounded as queries accumulate
# - Stale entries pollute results if underlying data changes
#
# I implement two eviction strategies:
#
# 1. TTL — entries older than MAX_AGE_SECONDS auto-expire.
#    Default: 1 hour. Stale entries are skipped during lookup
#    and lazily cleaned during store() to avoid scan overhead.
#
# 2. LRU — when cache exceeds MAX_ENTRIES, the least-recently-used
#    entry is evicted. "Used" means either stored or returned as a hit.
#    I track this via last_accessed on each CacheEntry.
#
# Why both? TTL handles staleness (data freshness), LRU handles
# memory pressure. They solve different problems.
# ─────────────────────────────────────────────────────────────────────

MAX_ENTRIES = 500       # Maximum cache size before LRU eviction kicks in
MAX_AGE_SECONDS = 3600  # 1 hour TTL — entries older than this are stale


@dataclass
class CacheEntry:
    """
    A single entry in the semantic cache.

    Fields:
        query:            Original natural language query string
        embedding:        L2-normalized embedding vector (384-dim)
        result:           The computed result/answer for this query
        dominant_cluster: Cluster index — used for fast lookup filtering
        timestamp:        Unix time when entry was created (for TTL)
        last_accessed:    Unix time when entry was last returned as a hit (for LRU)
        cluster_probs:    Full soft cluster distribution (for multi-cluster lookup)
        access_count:     Number of times this entry was returned as a hit
    """
    query: str
    embedding: np.ndarray
    result: str
    dominant_cluster: int
    timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    cluster_probs: Optional[np.ndarray] = None
    access_count: int = 0


class SemanticCache:
    """
    Cluster-aware semantic cache with multi-cluster lookup, TTL, and LRU eviction.
    Built entirely from scratch — no Redis, no Memcached, no caching library.

    Internally maintains:
    - A list of CacheEntry objects (the cache store)
    - A dict mapping cluster_id → list of indices into the store
      (the cluster index for fast lookup)
    - Hit/miss/eviction counters for observability
    - Latency tracking to prove cluster-aware lookup is faster than naive
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 max_entries: int = MAX_ENTRIES,
                 max_age: float = MAX_AGE_SECONDS):
        """
        Args:
            threshold:   Cosine similarity threshold for cache hit (0, 1].
            max_entries: Maximum cache entries before LRU eviction.
            max_age:     TTL in seconds — entries older than this are stale.
        """
        self.threshold = threshold
        self.max_entries = max_entries
        self.max_age = max_age

        # Core cache store — list of CacheEntry
        self._store: list[CacheEntry] = []

        # Cluster index — maps cluster_id → [store indices]
        # This is what makes lookup O(cluster_size) not O(total_cache)
        self._cluster_index: dict[int, list[int]] = {}

        # Stats
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._eviction_count: int = 0
        self._ttl_expiry_count: int = 0

        # Latency tracking — proves cluster-aware lookup is faster
        # Stores the number of comparisons per lookup, not wall-clock time,
        # because comparison count is deterministic and hardware-independent.
        self._comparison_counts: list[int] = []

    # ─────────────────────────────────────────────────────────────────
    # Core lookup logic — Multi-Cluster Aware
    # ─────────────────────────────────────────────────────────────────

    def lookup(self, query_embedding: np.ndarray, dominant_cluster: int,
               cluster_probs: np.ndarray | None = None) -> Optional[CacheEntry]:
        """
        Search the cache for a semantically similar query.

        Multi-cluster lookup strategy (uses Part 2's fuzzy clustering):
        ─────────────────────────────────────────────────────────────
        1. If cluster_probs are provided, find the top-3 clusters by
           probability and search all cached entries in those clusters.
           This catches cross-cluster matches that single-cluster lookup
           would miss (e.g. "gun legislation" cached in politics cluster,
           query about "firearm regulations" assigned to firearms cluster).

        2. If cluster_probs are not provided (backward compat), fall back
           to single-cluster lookup.

        3. If the top clusters have < 3 entries total, fall back to full
           scan (cache is too sparse for cluster filtering to help).

        TTL enforcement:
        - Entries older than max_age are skipped during lookup.
        - This is lazy expiry — I don't proactively scan and delete,
          I just ignore stale entries. Cleanup happens during store().

        Args:
            query_embedding:  L2-normalized query vector (384-dim)
            dominant_cluster: Predicted dominant cluster for query
            cluster_probs:    Optional soft cluster distribution (from GMM)

        Returns:
            Best matching CacheEntry if similarity >= threshold, else None
        """
        if len(self._store) == 0:
            self._miss_count += 1
            return None

        now = time.time()
        comparisons = 0
        best_score = -1.0
        best_entry = None

        # ── Determine which clusters to search ──
        if cluster_probs is not None:
            # Multi-cluster lookup: search top-N clusters by probability.
            # This is the key novelty — uses Part 2's soft assignments
            # to search beyond the dominant cluster.
            top_clusters = np.argsort(cluster_probs)[-MAX_LOOKUP_CLUSTERS:][::-1]
            candidate_indices = []
            for c in top_clusters:
                candidate_indices.extend(self._cluster_index.get(int(c), []))
        else:
            # Single-cluster fallback
            candidate_indices = self._cluster_index.get(dominant_cluster, [])

        # ── Sparse fallback ──
        # If the selected clusters have very few entries, the cache is
        # too sparse for cluster filtering to be useful. Fall back to
        # full scan to avoid false misses in the cold-start phase.
        if len(candidate_indices) < 3:
            candidate_indices = list(range(len(self._store)))

        # ── Search candidates ──
        for idx in candidate_indices:
            if idx >= len(self._store):
                continue

            entry = self._store[idx]

            # TTL check — skip stale entries
            if (now - entry.timestamp) > self.max_age:
                continue

            comparisons += 1
            score = self._cosine_similarity(query_embedding, entry.embedding)

            if score > best_score:
                best_score = score
                best_entry = entry

        # Track comparisons for performance analysis
        self._comparison_counts.append(comparisons)

        if best_score >= self.threshold and best_entry is not None:
            self._hit_count += 1
            # Update LRU tracking
            best_entry.last_accessed = now
            best_entry.access_count += 1
            return best_entry
        else:
            self._miss_count += 1
            return None

    # ─────────────────────────────────────────────────────────────────
    # Store a new entry (with LRU eviction)
    # ─────────────────────────────────────────────────────────────────

    def store(
        self,
        query: str,
        embedding: np.ndarray,
        result: str,
        dominant_cluster: int,
        cluster_probs: Optional[np.ndarray] = None
    ) -> CacheEntry:
        """
        Add a new query+result to the cache.

        Before inserting:
        1. Evict TTL-expired entries (lazy cleanup)
        2. If cache is at max capacity, evict the LRU entry

        After inserting:
        - Updates the cluster index for fast cluster-aware lookups

        Args:
            query:            Raw query string
            embedding:        L2-normalized embedding (384-dim)
            result:           Computed result to cache
            dominant_cluster: Cluster this query belongs to
            cluster_probs:    Full soft distribution (for multi-cluster lookup)

        Returns:
            The newly created CacheEntry
        """
        # ── Evict expired entries (lazy TTL cleanup) ──
        # I do this during store() rather than lookup() to avoid
        # adding latency to the read path. Writes are less frequent
        # than reads (every miss triggers a store, every query triggers a lookup).
        self._evict_expired()

        # ── LRU eviction if at capacity ──
        if len(self._store) >= self.max_entries:
            self._evict_lru()

        entry = CacheEntry(
            query=query,
            embedding=embedding,
            result=result,
            dominant_cluster=dominant_cluster,
            cluster_probs=cluster_probs
        )

        # Add to main store
        store_idx = len(self._store)
        self._store.append(entry)

        # Update cluster index
        if dominant_cluster not in self._cluster_index:
            self._cluster_index[dominant_cluster] = []
        self._cluster_index[dominant_cluster].append(store_idx)

        return entry

    # ─────────────────────────────────────────────────────────────────
    # Eviction strategies
    # ─────────────────────────────────────────────────────────────────

    def _evict_expired(self):
        """
        Remove entries older than max_age (TTL expiry).

        This is a full rebuild of the store — expensive, but only runs
        during store() which happens on cache misses. In a steady-state
        system with good hit rates, this runs infrequently.
        """
        now = time.time()
        expired_count = sum(1 for e in self._store if (now - e.timestamp) > self.max_age)

        if expired_count == 0:
            return

        # Rebuild store and cluster index without expired entries
        new_store = []
        new_cluster_index: dict[int, list[int]] = {}

        for entry in self._store:
            if (now - entry.timestamp) > self.max_age:
                self._ttl_expiry_count += 1
                continue

            new_idx = len(new_store)
            new_store.append(entry)

            if entry.dominant_cluster not in new_cluster_index:
                new_cluster_index[entry.dominant_cluster] = []
            new_cluster_index[entry.dominant_cluster].append(new_idx)

        self._store = new_store
        self._cluster_index = new_cluster_index

    def _evict_lru(self):
        """
        Remove the least-recently-used entry when cache is at capacity.

        LRU entry = the one with the oldest last_accessed timestamp.
        After removal, rebuild cluster index to keep indices consistent.
        """
        if len(self._store) == 0:
            return

        # Find LRU entry
        lru_idx = min(range(len(self._store)), key=lambda i: self._store[i].last_accessed)

        # Remove it
        self._store.pop(lru_idx)
        self._eviction_count += 1

        # Rebuild cluster index (indices shifted after pop)
        self._rebuild_cluster_index()

    def _rebuild_cluster_index(self):
        """Rebuild cluster_id → [store indices] mapping after eviction."""
        self._cluster_index = {}
        for idx, entry in enumerate(self._store):
            if entry.dominant_cluster not in self._cluster_index:
                self._cluster_index[entry.dominant_cluster] = []
            self._cluster_index[entry.dominant_cluster].append(idx)

    # ─────────────────────────────────────────────────────────────────
    # Similarity computation
    # ─────────────────────────────────────────────────────────────────

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Since my embeddings are L2-normalized (done in embeddings.py),
        cosine similarity reduces to a simple dot product:
            cosine_sim(a, b) = a · b  (when ||a|| = ||b|| = 1)

        This is O(384) per comparison — very fast.

        Returns:
            float in [-1, 1], where 1 = identical direction
        """
        return float(np.dot(a, b))

    # ─────────────────────────────────────────────────────────────────
    # Cache management
    # ─────────────────────────────────────────────────────────────────

    def flush(self):
        """
        Clear all cache entries and reset all stats.
        Called by DELETE /cache endpoint.
        """
        self._store.clear()
        self._cluster_index.clear()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._ttl_expiry_count = 0
        self._comparison_counts.clear()

    def get_stats(self) -> dict:
        """
        Return current cache statistics including performance metrics.
        Called by GET /cache/stats endpoint.

        Includes:
        - Hit/miss counts and rate
        - Eviction and TTL expiry counts
        - Average comparisons per lookup (proves cluster-aware efficiency)
        - Cluster distribution of cached entries
        """
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0

        # Comparison stats — this is the evidence that cluster-aware
        # lookup reduces work. A naive cache would compare against ALL
        # entries every time. my cluster-aware cache compares against
        # only the entries in the relevant clusters.
        avg_comparisons = (
            round(sum(self._comparison_counts) / len(self._comparison_counts), 1)
            if self._comparison_counts else 0
        )

        return {
            "total_entries": len(self._store),
            "max_entries": self.max_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 3),
            "threshold": self.threshold,
            "evictions": {
                "lru_evictions": self._eviction_count,
                "ttl_expiries": self._ttl_expiry_count,
            },
            "performance": {
                "avg_comparisons_per_lookup": avg_comparisons,
                "total_lookups": len(self._comparison_counts),
                "comparison_vs_naive": f"{avg_comparisons} vs {len(self._store)} (full scan)",
            },
            "cluster_distribution": {
                str(k): len(v)
                for k, v in self._cluster_index.items()
            }
        }

    def set_threshold(self, threshold: float):
        """
        Update the similarity threshold at runtime.
        Useful for threshold exploration without restarting the server.
        """
        assert 0.0 < threshold <= 1.0, "Threshold must be in (0, 1]"
        self.threshold = threshold

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return (
            f"SemanticCache("
            f"entries={len(self._store)}, "
            f"threshold={self.threshold}, "
            f"hits={self._hit_count}, "
            f"misses={self._miss_count})"
        )
