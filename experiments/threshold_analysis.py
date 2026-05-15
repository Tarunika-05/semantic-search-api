import requests
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

BASE_URL = "http://127.0.0.1:8000"

queries = [
    "What causes global warming?",
    "Why does climate change happen?",
    "Reasons for global warming",
    "How do rockets reach space?",
    "How does a spacecraft launch?",
    "How does 3D graphics rendering work?",
    "Explain computer graphics rendering"
]

results = []


def run_queries():
    print("\nRunning query experiments...\n")

    for q in queries:

        start = time.time()

        r = requests.post(
            f"{BASE_URL}/query",
            json={"query": q}
        )

        latency = time.time() - start

        data = r.json()

        results.append({
            "query": q,
            "cache_hit": data["cache_hit"],
            "similarity": data.get("similarity_score", 0),
            "cluster": data["dominant_cluster"],
            "latency": latency
        })


def visualize_cache_hits():

    df = pd.DataFrame(results)

    hits = df["cache_hit"].sum()
    misses = len(df) - hits

    plt.figure()
    plt.bar(["Cache Hits", "Cache Misses"], [hits, misses])
    plt.title("Semantic Cache Performance")
    plt.ylabel("Number of Queries")
    plt.show()


def visualize_latency():

    df = pd.DataFrame(results)

    hit_latency = df[df["cache_hit"]]["latency"]
    miss_latency = df[~df["cache_hit"]]["latency"]

    plt.figure()
    
    # Safeguard against empty data arrays causing Matplotlib to crash
    data_to_plot = []
    labels = []
    if not hit_latency.empty:
        data_to_plot.append(hit_latency)
        labels.append("Cache Hit")
    if not miss_latency.empty:
        data_to_plot.append(miss_latency)
        labels.append("Cache Miss")
        
    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels)
    plt.title("Latency Comparison")
    plt.ylabel("Response Time (seconds)")
    plt.show()


def visualize_clusters():

    df = pd.DataFrame(results)

    plt.figure()

    clusters = df["cluster"]
    x = np.arange(len(clusters))

    plt.scatter(x, clusters)

    plt.title("Query Cluster Distribution")
    plt.xlabel("Query Index")
    plt.ylabel("Dominant Cluster")
    plt.show()


def threshold_experiment():

    thresholds = [0.95, 0.90, 0.85, 0.80, 0.70]

    hit_rates = []

    for t in thresholds:

        requests.patch(
            f"{BASE_URL}/cache/threshold",
            params={"threshold": t}
        )

        requests.delete(f"{BASE_URL}/cache")

        local_hits = 0

        for q in queries:

            r = requests.post(
                f"{BASE_URL}/query",
                json={"query": q}
            )

            data = r.json()

            if data["cache_hit"]:
                local_hits += 1

        hit_rates.append(local_hits / len(queries))

    plt.figure()
    plt.plot(thresholds, hit_rates, marker="o")
    plt.title("Similarity Threshold vs Cache Hit Rate")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Cache Hit Rate")
    plt.gca().invert_xaxis()
    plt.show()


if __name__ == "__main__":

    run_queries()

    visualize_cache_hits()
    visualize_latency()
    visualize_clusters()
    threshold_experiment()