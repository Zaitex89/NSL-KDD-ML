import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# Global style
sns.set_theme(style="darkgrid")
FIGSIZE = (10, 6)


def _plot_label_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Attacktypes
    df["label"].value_counts().head(10).plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Topp 10 attacktyper")
    axes[0].set_ylabel("Antal")
    axes[0].tick_params(axis="x", rotation=45)

    # Normal vs Attack
    df["label_binary"].value_counts().plot(
        kind="pie", ax=axes[1],
        labels=["Attack", "Normal"],
        autopct="%1.1f%%",
        colors=["tomato", "steelblue"]
    )
    axes[1].set_title("Normal vs Attack")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig("label_distribution.png")
    plt.show()


def _plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=["label_binary"], errors="ignore"
    )
    # Choose the top 20 most varying for readability
    top_cols = numeric_df.std().nlargest(20).index
    corr = numeric_df[top_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5, center=0)
    plt.title("Korrelationsmatris (topp 20 variabler)")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()


def _plot_pca(df):
    num_cols = [
        c for c in df.select_dtypes(include=["float64", "int64"]).columns
        if "label" not in c
    ]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[num_cols])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_result[:, 0], pca_result[:, 1],
        c=df["label_binary"], cmap="coolwarm",
        alpha=0.3, s=1
    )
    plt.colorbar(scatter, label="0 = Normal, 1 = Attack")
    plt.title(f"PCA - Normal vs Attack\nFörklarad varians: {pca.explained_variance_ratio_.round(3)}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("pca.png")
    plt.show()


def perform_eda(df):
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - NSL-KDD")
    print("=" * 60)

    # Basic info
    print(f"\nShape:      {df.shape}")
    print(f"Features:   {df.shape[1]}")
    print(f"Missing:    {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    # Labels
    print("\n--- Labelfördelning ---")
    print(df["label"].value_counts().head(15).to_string())
    print("\nNormal vs Attack (%):")
    print((df["label_binary"].value_counts(normalize=True) * 100).round(1).to_string())

    # Numerical stats
    print("\n--- Numerisk statistik (urval) ---")
    print(df.describe().iloc[:, :8].round(2).to_string())

    # Protocoll-encoding
    print("\n--- Protokoll (one-hot) ---")
    for col in [c for c in df.columns if c.startswith("protocol_type_")]:
        print(f"  {col}: {int(df[col].sum())}")

    # Graphs
    print("\nSkapar grafer...")
    _plot_label_distribution(df)
    _plot_correlation_heatmap(df)
    _plot_pca(df)

    print("\nEDA klar! Grafer sparade som PNG.")
    return df