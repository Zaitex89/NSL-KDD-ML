import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

sns.set_theme(style="darkgrid")


def _find_optimal_k(df, num_cols):
    print("\n--- Elbow & Silhouette ---")

    k_range = range(2, 11)
    inertias = []
    silhouette_scores = []

    print("Testar k-värden (2–10)...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(df[num_cols])
        inertias.append(km.inertia_)
        score = silhouette_score(df[num_cols], labels, sample_size=10000, random_state=42)
        silhouette_scores.append(score)
        print(f"  k={k} → Silhouette: {score:.3f}")

    best_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nBästa k enligt Silhouette Score: {best_k}")

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


def _run_kmeans(df, num_cols, k):
    print(f"\n--- K-Means (k={k}) ---")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["kmeans_cluster"] = kmeans.fit_predict(df[num_cols])

    print("\nKlusterfördelning:")
    print(df["kmeans_cluster"].value_counts().sort_index().to_string())

    print("\nKluster vs label:")
    print(pd.crosstab(df["kmeans_cluster"], df["label_binary"],
                      rownames=["Kluster"],
                      colnames=["0=Normal, 1=Attack"]).to_string())

    pca = PCA(n_components=2)
    coords = pca.fit_transform(df[num_cols])

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=df["kmeans_cluster"], cmap="tab10",
        alpha=0.3, s=1
    )
    plt.colorbar(scatter, label="Kluster")
    plt.title(f"K-Means (k={k}) – visualiserat med PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png")
    plt.show()

    return df


def _run_autoencoder(df, num_cols):
    print("\n--- Autoencoder ---")

    normal_data = df[df["label_binary"] == 0][num_cols]
    print(f"Tränar på {len(normal_data)} normala anslutningar...")

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16, 32, 64),
        activation="relu",
        max_iter=50,
        random_state=42,
        verbose=True
    )
    model.fit(normal_data, normal_data)

    reconstructed = model.predict(df[num_cols])
    errors = np.mean((df[num_cols].values - reconstructed) ** 2, axis=1)
    df["ae_error"] = errors

    normal_errors = errors[df["label_binary"] == 0]
    threshold = np.percentile(normal_errors, 95)
    print(f"Threshold (95:e percentilen): {threshold:.4f}")

    df["ae_pred"] = (errors > threshold).astype(int)

    print("\nKlassificeringsrapport:")
    print(classification_report(
        df["label_binary"], df["ae_pred"],
        target_names=["Normal", "Attack"]
    ))

    pca = PCA(n_components=2)
    coords = pca.fit_transform(df[num_cols])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(coords[:, 0], coords[:, 1],
                    c=df["label_binary"], cmap="coolwarm", alpha=0.3, s=1)
    axes[0].set_title("Faktiska labels\n(Blå=Normal, Röd=Attack)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(coords[:, 0], coords[:, 1],
                    c=df["ae_pred"], cmap="coolwarm", alpha=0.3, s=1)
    axes[1].set_title("Autoencoder förutsägelser\n(Blå=Normal, Röd=Anomali)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    axes[2].boxplot(
        [errors[df["label_binary"] == 0], errors[df["label_binary"] == 1]],
        labels=["Normal", "Attack"]
    )
    axes[2].axhline(y=threshold, linestyle="--", color="red", label=f"Threshold={threshold:.3f}")
    axes[2].set_title("Rekonstruktionsfel per klass")
    axes[2].set_ylabel("MSE")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("autoencoder.png")
    plt.show()

    return df


def perform_ml(df):
    print("=" * 60)
    print("UNSUPERVISED MACHINE LEARNING – NSL-KDD")
    print("=" * 60)

    num_cols = [
        c for c in df.select_dtypes(include=["float64", "int64"]).columns
        if "label" not in c
    ]

    best_k = _find_optimal_k(df, num_cols)
    df = _run_kmeans(df, num_cols, best_k)
    df = _run_autoencoder(df, num_cols)

    print("\nML klar! Grafer sparade som PNG.")
    return df