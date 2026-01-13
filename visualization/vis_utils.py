"""
Visualization utilities for PYRA experiments.
Includes metric computation, dimensionality reduction, and plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def compute_clustering_metrics(
    features: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.

    Args:
        features: Feature vectors, shape [N, D]
        labels: Ground truth labels, shape [N]

    Returns:
        Dictionary with metric scores:
            - silhouette: Silhouette coefficient (-1 to 1, higher is better)
            - davies_bouldin: Davies-Bouldin index (lower is better)
            - calinski_harabasz: Calinski-Harabasz index (higher is better)
    """
    metrics = {}

    # Silhouette score
    try:
        metrics['silhouette'] = silhouette_score(features, labels)
    except Exception as e:
        print(f"Warning: Could not compute silhouette score: {e}")
        metrics['silhouette'] = np.nan

    # Davies-Bouldin index
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
    except Exception as e:
        print(f"Warning: Could not compute Davies-Bouldin score: {e}")
        metrics['davies_bouldin'] = np.nan

    # Calinski-Harabasz index
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
    except Exception as e:
        print(f"Warning: Could not compute Calinski-Harabasz score: {e}")
        metrics['calinski_harabasz'] = np.nan

    return metrics


def compute_knn_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    test_split: float = 0.3
) -> float:
    """
    Compute k-NN classification accuracy.

    Args:
        features: Feature vectors, shape [N, D]
        labels: Ground truth labels, shape [N]
        k: Number of neighbors
        test_split: Fraction of data to use for testing

    Returns:
        k-NN accuracy
    """
    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_split, stratify=labels, random_state=42
    )

    # Train k-NN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Evaluate
    accuracy = knn.score(X_test, y_test)

    return accuracy


def compute_linear_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    test_split: float = 0.3
) -> float:
    """
    Compute linear probe classification accuracy.

    Args:
        features: Feature vectors, shape [N, D]
        labels: Ground truth labels, shape [N]
        test_split: Fraction of data to use for testing

    Returns:
        Linear probe accuracy
    """
    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_split, stratify=labels, random_state=42
    )

    # Train linear classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    accuracy = clf.score(X_test, y_test)

    return accuracy


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Visualization",
    save_path: Optional[str] = None,
    perplexity: int = 30,
    n_iter: int = 1000
) -> plt.Figure:
    """
    Create t-SNE visualization of features.

    Args:
        features: Feature vectors, shape [N, D]
        labels: Ground truth labels, shape [N]
        title: Plot title
        save_path: Path to save the figure (optional)
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations

    Returns:
        Matplotlib figure
    """
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each class with different color
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            label=f'Class {label}',
            alpha=0.6,
            s=30
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")

    return fig


def plot_umap(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "UMAP Visualization",
    save_path: Optional[str] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> plt.Figure:
    """
    Create UMAP visualization of features.

    Args:
        features: Feature vectors, shape [N, D]
        labels: Ground truth labels, shape [N]
        title: Plot title
        save_path: Path to save the figure (optional)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter

    Returns:
        Matplotlib figure
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None

    # Compute UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    features_2d = reducer.fit_transform(features)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each class with different color
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            label=f'Class {label}',
            alpha=0.6,
            s=30
        )

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved UMAP plot to {save_path}")

    return fig


def plot_layer_wise_metrics(
    metrics_dict: Dict[str, List[float]],
    metric_name: str,
    title: str,
    save_path: Optional[str] = None,
    ylabel: Optional[str] = None
) -> plt.Figure:
    """
    Plot layer-wise metrics for multiple conditions.

    Args:
        metrics_dict: Dictionary mapping condition names to metric values per layer
        metric_name: Name of the metric being plotted
        title: Plot title
        save_path: Path to save the figure (optional)
        ylabel: Y-axis label (optional, defaults to metric_name)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each condition
    for condition, values in metrics_dict.items():
        layers = list(range(len(values)))
        ax.plot(layers, values, marker='o', linewidth=2, markersize=6, label=condition)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(ylabel or metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer-wise plot to {save_path}")

    return fig


def plot_similarity_distribution(
    similarities_dict: Dict[str, np.ndarray],
    title: str = "Token Merge Similarity Distribution",
    save_path: Optional[str] = None,
    plot_type: str = 'violin'
) -> plt.Figure:
    """
    Plot distribution of token merge similarities.

    Args:
        similarities_dict: Dictionary mapping condition names to similarity arrays
        title: Plot title
        save_path: Path to save the figure (optional)
        plot_type: 'violin', 'hist', or 'box'

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if plot_type == 'violin':
        # Prepare data for violin plot
        data = []
        labels = []
        for condition, sims in similarities_dict.items():
            # Flatten all similarities
            flat_sims = sims.flatten()
            data.append(flat_sims)
            labels.append(condition)

        # Create violin plot
        parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right')

    elif plot_type == 'hist':
        # Create histogram
        for condition, sims in similarities_dict.items():
            flat_sims = sims.flatten()
            ax.hist(flat_sims, bins=50, alpha=0.5, label=condition, density=True)
        ax.legend(fontsize=11)

    elif plot_type == 'box':
        # Prepare data for box plot
        data = []
        labels = []
        for condition, sims in similarities_dict.items():
            flat_sims = sims.flatten()
            data.append(flat_sims)
            labels.append(condition)

        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_xticklabels(labels, rotation=15, ha='right')

    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved similarity distribution plot to {save_path}")

    return fig


def plot_layer_wise_similarity(
    similarities_dict: Dict[str, List[np.ndarray]],
    title: str = "Layer-wise Token Merge Similarity",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot mean similarity across layers for multiple conditions.

    Args:
        similarities_dict: Dictionary mapping condition names to list of similarity arrays per layer
        title: Plot title
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each condition
    for condition, sims_list in similarities_dict.items():
        # Compute mean similarity per layer
        mean_sims = [sims.mean() if len(sims) > 0 else 0 for sims in sims_list]
        std_sims = [sims.std() if len(sims) > 0 else 0 for sims in sims_list]

        layers = list(range(len(mean_sims)))
        ax.plot(layers, mean_sims, marker='o', linewidth=2, markersize=6, label=condition)
        ax.fill_between(
            layers,
            np.array(mean_sims) - np.array(std_sims),
            np.array(mean_sims) + np.array(std_sims),
            alpha=0.2
        )

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Cosine Similarity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer-wise similarity plot to {save_path}")

    return fig


def save_metrics_to_file(
    metrics: Dict[str, any],
    save_path: str
):
    """
    Save metrics to a text file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the metrics
    """
    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, (list, np.ndarray)):
                f.write(f"{key}:\n")
                for i, v in enumerate(value):
                    f.write(f"  Layer {i}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"Saved metrics to {save_path}")


def plot_comparison_grid(
    features_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    title_prefix: str = "",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grid of t-SNE plots for comparing multiple conditions.

    Args:
        features_dict: Dictionary mapping condition names to feature arrays
        labels: Ground truth labels
        title_prefix: Prefix for subplot titles
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib figure
    """
    n_conditions = len(features_dict)
    n_cols = min(3, n_conditions)
    n_rows = (n_conditions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_conditions == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (condition, features) in enumerate(features_dict.items()):
        ax = axes[idx]

        # Compute t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(features) // 5), n_iter=1000, random_state=42)
        features_2d = tsne.fit_transform(features)

        # Plot
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                label=f'Class {label}',
                alpha=0.6,
                s=20
            )

        ax.set_title(f"{title_prefix} {condition}", fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=8, loc='best')

    # Hide unused subplots
    for idx in range(n_conditions, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison grid to {save_path}")

    return fig
