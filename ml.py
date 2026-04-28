# unsupervised ML
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_theme(style="darkgrid")


def _find_optimal_k(df, num_cols):
    print("\n--- Elbow & Silhouette ---")

    k_range = range(2, 11)
    inertias = []
    silhouette_scores = []

    print("Testar k-värden (2-10)...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(df[num_cols])
        inertias.append(km.inertia_)
        score = silhouette_score(df[num_cols], labels, sample_size=10000, random_state=42)
        silhouette_scores.append(score)
        print(f"  k={k} → Silhouette: {score:.3f}")

    best_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nBästa k enligt Silhouette Score: {best_k}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_range, inertias, marker="o", color="steelblue")
    axes[0].set_title("Elbow-metoden")
    axes[0].set_xlabel("Antal kluster (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_range, silhouette_scores, marker="o", color="tomato")
    axes[1].axvline(x=best_k, linestyle="--", color="gray", label=f"Bästa k={best_k}")
    axes[1].set_title("Silhouette Score per k")
    axes[1].set_xlabel("Antal kluster (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("kmeans_elbow_silhouette.png")
    plt.show()

    return best_k


def perform_ml(df):
    print("=" * 60)
    print("UNSUPERVISED MACHINE LEARNING - NSL-KDD")
    print("=" * 60)

    num_cols = [
        c for c in df.select_dtypes(include=["float64", "int64"]).columns
        if "label" not in c
    ]

    best_k = _find_optimal_k(df, num_cols)

    print(f"\nOptimalt k = {best_k}. Redo att köra K-Means!")
    return df, best_k