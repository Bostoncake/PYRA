"""
PYRA Compression Visualization Package

This package provides tools for visualizing and quantifying compression effects
in vision transformers with token merging.
"""

from .feature_extractor import FeatureExtractor, create_subset_dataloader
from .vis_utils import (
    compute_clustering_metrics,
    compute_knn_accuracy,
    compute_linear_probe_accuracy,
    plot_tsne,
    plot_umap,
    plot_layer_wise_metrics,
    plot_similarity_distribution,
    plot_layer_wise_similarity,
    plot_comparison_grid,
    save_metrics_to_file,
)

__version__ = "1.0.0"
__all__ = [
    "FeatureExtractor",
    "create_subset_dataloader",
    "compute_clustering_metrics",
    "compute_knn_accuracy",
    "compute_linear_probe_accuracy",
    "plot_tsne",
    "plot_umap",
    "plot_layer_wise_metrics",
    "plot_similarity_distribution",
    "plot_layer_wise_similarity",
    "plot_comparison_grid",
    "save_metrics_to_file",
]
