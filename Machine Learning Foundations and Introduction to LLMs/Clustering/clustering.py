import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    AffinityPropagation,
    HDBSCAN,
)
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------
# Data
# -----------------------------
def load_and_preprocess() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Wine dataset, standardize features."""
    data = load_wine()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, list(data.feature_names)


# -----------------------------
# Results
# -----------------------------
@dataclass
class ClusteringResult:
    name: str
    labels: np.ndarray
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    ari: float
    noise_ratio: float


# -----------------------------
# Evaluation
# -----------------------------
def compute_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Returns (silhouette, davies_bouldin, calinski_harabasz, ari, noise_ratio).
    Noise points (label == -1) are excluded from internal metric computation.
    Returns sentinel values if fewer than 2 valid clusters exist.
    """
    noise_mask = labels == -1
    noise_ratio = float(noise_mask.sum()) / len(labels)

    valid_mask = ~noise_mask
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    y_valid = y_true[valid_mask]

    n_clusters = len(np.unique(labels_valid))

    if n_clusters < 2 or len(X_valid) < 2:
        return -1.0, float("inf"), 0.0, 0.0, noise_ratio

    sil = silhouette_score(X_valid, labels_valid)
    db = davies_bouldin_score(X_valid, labels_valid)
    ch = calinski_harabasz_score(X_valid, labels_valid)
    ari = adjusted_rand_score(y_valid, labels_valid)

    return sil, db, ch, ari, noise_ratio


# -----------------------------
# Clustering Algorithms
# -----------------------------
def run_kmeans_plus_plus(X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """k-Means with k-Means++ initialization."""
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=seed,
    )
    return km.fit_predict(X)


def run_agglomerative(
    X: np.ndarray, n_clusters: int, linkage_method: str = "ward"
) -> np.ndarray:
    """Agglomerative Hierarchical Clustering."""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    return agg.fit_predict(X)


def run_birch(
    X: np.ndarray, n_clusters: int, threshold: float = 0.5
) -> np.ndarray:
    """BIRCH: Balanced Iterative Reducing and Clustering using Hierarchies."""
    birch = Birch(threshold=threshold, branching_factor=50, n_clusters=n_clusters)
    return birch.fit_predict(X)


def run_dbscan(
    X: np.ndarray, eps: float = 1.5, min_samples: int = 5
) -> np.ndarray:
    """DBSCAN: Density-Based Spatial Clustering of Applications with Noise."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(X)


def run_hdbscan(
    X: np.ndarray, min_cluster_size: int = 10, min_samples: int = 5
) -> np.ndarray:
    """HDBSCAN: Hierarchical DBSCAN."""
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    return hdb.fit_predict(X)


def run_affinity_propagation(
    X: np.ndarray, damping: float = 0.9, seed: int = 42
) -> np.ndarray:
    """Affinity Propagation: exemplar-based message-passing clustering."""
    ap = AffinityPropagation(damping=damping, random_state=seed, max_iter=500)
    return ap.fit_predict(X)


# -----------------------------
# Benchmark runner
# -----------------------------
def run_benchmark(
    X: np.ndarray,
    y_true: np.ndarray,
    n_clusters: int,
    seed: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    birch_threshold: float,
    ap_damping: float,
) -> List[ClusteringResult]:

    experiments = [
        ("k-Means++",               run_kmeans_plus_plus(X, n_clusters, seed)),
        ("Agglomerative (Ward)",    run_agglomerative(X, n_clusters, "ward")),
        ("Agglomerative (Complete)",run_agglomerative(X, n_clusters, "complete")),
        ("BIRCH",                   run_birch(X, n_clusters, birch_threshold)),
        ("DBSCAN",                  run_dbscan(X, dbscan_eps, dbscan_min_samples)),
        ("HDBSCAN",                 run_hdbscan(X, hdbscan_min_cluster_size, hdbscan_min_samples)),
        ("Affinity Propagation",    run_affinity_propagation(X, ap_damping, seed)),
    ]

    results: List[ClusteringResult] = []
    for name, labels in experiments:
        noise_mask = labels == -1
        n_unique = len(np.unique(labels[~noise_mask]))
        sil, db, ch, ari, noise_ratio = compute_metrics(X, labels, y_true)
        results.append(ClusteringResult(
            name=name,
            labels=labels,
            n_clusters=n_unique,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
            ari=ari,
            noise_ratio=noise_ratio,
        ))
        print(
            f"  {name:<28} k={n_unique:2d}  "
            f"Sil={sil:6.3f}  DB={db:6.3f}  CH={ch:8.1f}  "
            f"ARI={ari:6.3f}  Noise={noise_ratio:.1%}"
        )

    return results


def print_summary_table(results: List[ClusteringResult]) -> None:
    header = (
        f"{'Algorithm':<28} {'k':>4} {'Silhouette':>12} "
        f"{'Davies-Bouldin':>16} {'Calinski-Harabasz':>18} {'ARI':>8} {'Noise%':>8}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        db_str = f"{r.davies_bouldin:16.4f}" if r.davies_bouldin != float("inf") else f"{'N/A':>16}"
        print(
            f"{r.name:<28} {r.n_clusters:>4} "
            f"{r.silhouette:>12.4f} {db_str} "
            f"{r.calinski_harabasz:>18.1f} {r.ari:>8.4f} {r.noise_ratio:>7.1%}"
        )
    print(sep)


# -----------------------------
# Visualization
# -----------------------------
def reduce_to_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Reduce to 2D via PCA for visualization."""
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X)


def plot_clusters(
    X_2d: np.ndarray,
    results: List[ClusteringResult],
    y_true: np.ndarray,
    save_path: str,
) -> None:
    """Grid of scatter plots — one per algorithm plus ground truth."""
    n_plots = len(results) + 1
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    # Ground truth
    ax = axes[0]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="tab10", s=20, alpha=0.85)
    ax.set_title("Ground Truth", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    for i, result in enumerate(results, 1):
        ax = axes[i]
        labels = result.labels
        noise_mask = labels == -1

        if noise_mask.any():
            ax.scatter(
                X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                c="lightgrey", s=10, alpha=0.5, label="Noise",
            )
        valid_mask = ~noise_mask
        if valid_mask.any():
            ax.scatter(
                X_2d[valid_mask, 0], X_2d[valid_mask, 1],
                c=labels[valid_mask], cmap="tab10", s=20, alpha=0.85,
            )

        db_str = f"{result.davies_bouldin:.3f}" if result.davies_bouldin != float("inf") else "N/A"
        title = (
            f"{result.name}\n"
            f"k={result.n_clusters}  Sil={result.silhouette:.3f}  ARI={result.ari:.3f}"
        )
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Clustering Comparison — Wine Dataset (PCA 2D)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Cluster scatter plots  → {save_path}")


def plot_metrics_comparison(
    results: List[ClusteringResult], save_path: str
) -> None:
    """Bar charts comparing all four metrics across algorithms."""
    names = [r.name for r in results]
    silhouettes = [r.silhouette for r in results]
    dbs = [
        r.davies_bouldin if r.davies_bouldin != float("inf") else 0.0
        for r in results
    ]
    chs = [r.calinski_harabasz for r in results]
    aris = [r.ari for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    panels = [
        (silhouettes, "Silhouette Score  (↑ better)", axes[0, 0], "steelblue"),
        (dbs,         "Davies-Bouldin Index  (↓ better)", axes[0, 1], "coral"),
        (chs,         "Calinski-Harabasz Index  (↑ better)", axes[1, 0], "mediumseagreen"),
        (aris,        "Adjusted Rand Index  (↑ better)", axes[1, 1], "orchid"),
    ]

    for values, title, ax, color in panels:
        bars = ax.bar(
            names, values, color=color, alpha=0.82,
            edgecolor="black", linewidth=0.5,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=35)
        max_val = max(abs(v) for v in values) if values else 1.0
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * (max_val or 1),
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
            )

    plt.suptitle(
        "Metric Comparison Across Clustering Algorithms",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Metrics comparison     → {save_path}")


def plot_silhouette_analysis(
    X: np.ndarray,
    results: List[ClusteringResult],
    save_path: str,
) -> None:
    """Per-cluster silhouette coefficient plots for each algorithm."""
    valid_results = [r for r in results if r.silhouette > -1.0]
    if not valid_results:
        return

    ncols = 3
    nrows = (len(valid_results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, result in enumerate(valid_results):
        ax = axes[i]
        noise_mask = result.labels == -1
        X_valid = X[~noise_mask]
        labels_valid = result.labels[~noise_mask]
        unique_clusters = np.unique(labels_valid)
        n_clusters = len(unique_clusters)

        sil_vals = silhouette_samples(X_valid, labels_valid)
        colors = cm.nipy_spectral(np.linspace(0.1, 0.9, n_clusters))
        y_lower = 10

        for cluster_id, color in zip(unique_clusters, colors):
            cluster_sil = np.sort(sil_vals[labels_valid == cluster_id])
            y_upper = y_lower + len(cluster_sil)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper), 0, cluster_sil,
                facecolor=color, edgecolor=color, alpha=0.75,
            )
            ax.text(-0.06, y_lower + 0.5 * len(cluster_sil), str(cluster_id), fontsize=7)
            y_lower = y_upper + 8

        ax.axvline(
            x=result.silhouette, color="red", linestyle="--", lw=1.5,
            label=f"avg = {result.silhouette:.3f}",
        )
        ax.set_xlim([-0.25, 1.0])
        ax.set_title(result.name, fontsize=8.5)
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("Cluster")
        ax.legend(fontsize=7)

    for j in range(len(valid_results), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Silhouette Analysis per Algorithm", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Silhouette analysis    → {save_path}")


def plot_dendrogram(X: np.ndarray, save_path: str, n_samples: int = 120) -> None:
    """Truncated Ward-linkage dendrogram."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sub = X[idx]

    Z = linkage(X_sub, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True,
        color_threshold=0.7 * max(Z[:, 2]),
    )
    ax.set_title(
        "Hierarchical Clustering Dendrogram (Ward linkage, truncated)",
        fontweight="bold",
    )
    ax.set_xlabel("Sample index or (cluster size)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Dendrogram             → {save_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    # -------------------------
    # Constants (all here)
    # -------------------------
    SEED = 42

    # Wine dataset has 3 cultivar classes — use as target number of clusters
    N_CLUSTERS = 3

    # DBSCAN (tuned for standardized 13-dimensional wine features;
    # inter-point distances scale with sqrt(d), so eps must be larger than in 2D)
    DBSCAN_EPS = 2.5
    DBSCAN_MIN_SAMPLES = 5

    # HDBSCAN
    HDBSCAN_MIN_CLUSTER_SIZE = 10
    HDBSCAN_MIN_SAMPLES = 5

    # BIRCH
    BIRCH_THRESHOLD = 0.5

    # Affinity Propagation
    AP_DAMPING = 0.9

    # Output
    OUTPUT_DIR = "./outputs/clustering"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------
    # Data
    # -------------------------
    print("Loading and preprocessing Wine dataset...")
    X, y_true, feature_names = load_and_preprocess()
    print(
        f"  {X.shape[0]} samples | {X.shape[1]} features | "
        f"{len(np.unique(y_true))} cultivar classes"
    )

    # -------------------------
    # Benchmark
    # -------------------------
    print("\nRunning clustering algorithms...")
    results = run_benchmark(
        X=X,
        y_true=y_true,
        n_clusters=N_CLUSTERS,
        seed=SEED,
        dbscan_eps=DBSCAN_EPS,
        dbscan_min_samples=DBSCAN_MIN_SAMPLES,
        hdbscan_min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        hdbscan_min_samples=HDBSCAN_MIN_SAMPLES,
        birch_threshold=BIRCH_THRESHOLD,
        ap_damping=AP_DAMPING,
    )

    print_summary_table(results)

    # -------------------------
    # Visualization
    # -------------------------
    print("\nGenerating visualizations...")
    X_2d = reduce_to_2d(X, seed=SEED)

    plot_clusters(
        X_2d, results, y_true,
        save_path=os.path.join(OUTPUT_DIR, "cluster_comparison.png"),
    )
    plot_metrics_comparison(
        results,
        save_path=os.path.join(OUTPUT_DIR, "metrics_comparison.png"),
    )
    plot_silhouette_analysis(
        X, results,
        save_path=os.path.join(OUTPUT_DIR, "silhouette_analysis.png"),
    )
    plot_dendrogram(
        X,
        save_path=os.path.join(OUTPUT_DIR, "dendrogram.png"),
    )

    print(f"\nDone. All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
