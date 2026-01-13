"""
Test script for PYRA visualization system.
Tests the visualization utilities with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.vis_utils import (
    compute_clustering_metrics,
    compute_knn_accuracy,
    compute_linear_probe_accuracy,
    plot_tsne,
    plot_umap,
    plot_layer_wise_metrics,
    plot_similarity_distribution,
    plot_layer_wise_similarity,
    plot_comparison_grid,
)


def generate_synthetic_data(n_samples=200, n_features=512, n_classes=5, seed=42):
    """Generate synthetic feature data for testing."""
    np.random.seed(seed)

    # Create cluster centers
    centers = np.random.randn(n_classes, n_features) * 5

    # Generate samples around centers
    features = []
    labels = []

    samples_per_class = n_samples // n_classes

    for i in range(n_classes):
        class_features = centers[i] + np.random.randn(samples_per_class, n_features)
        features.append(class_features)
        labels.extend([i] * samples_per_class)

    features = np.vstack(features)
    labels = np.array(labels)

    return features, labels


def test_clustering_metrics():
    """Test clustering metrics computation."""
    print("Testing clustering metrics...")

    features, labels = generate_synthetic_data()
    metrics = compute_clustering_metrics(features, labels)

    print(f"  Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f}")

    assert -1 <= metrics['silhouette'] <= 1, "Silhouette score out of range"
    assert metrics['davies_bouldin'] >= 0, "Davies-Bouldin index should be non-negative"
    assert metrics['calinski_harabasz'] >= 0, "Calinski-Harabasz index should be non-negative"

    print("  ✓ Clustering metrics test passed!")


def test_knn_accuracy():
    """Test k-NN accuracy computation."""
    print("\nTesting k-NN accuracy...")

    features, labels = generate_synthetic_data()
    accuracy = compute_knn_accuracy(features, labels, k=5)

    print(f"  k-NN Accuracy: {accuracy:.4f}")

    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert accuracy > 0.5, "k-NN accuracy should be > 0.5 for well-separated clusters"

    print("  ✓ k-NN accuracy test passed!")


def test_linear_probe_accuracy():
    """Test linear probe accuracy computation."""
    print("\nTesting linear probe accuracy...")

    features, labels = generate_synthetic_data()
    accuracy = compute_linear_probe_accuracy(features, labels)

    print(f"  Linear Probe Accuracy: {accuracy:.4f}")

    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert accuracy > 0.5, "Linear probe accuracy should be > 0.5 for well-separated clusters"

    print("  ✓ Linear probe accuracy test passed!")


def test_tsne_plot():
    """Test t-SNE visualization."""
    print("\nTesting t-SNE plot...")

    features, labels = generate_synthetic_data(n_samples=100)
    fig = plot_tsne(
        features,
        labels,
        title="Test t-SNE",
        save_path="./test_tsne.png",
        perplexity=15,
        n_iter=500
    )

    assert os.path.exists("./test_tsne.png"), "t-SNE plot not saved"
    print("  ✓ t-SNE plot test passed!")

    plt.close(fig)
    os.remove("./test_tsne.png")


def test_umap_plot():
    """Test UMAP visualization."""
    print("\nTesting UMAP plot...")

    try:
        import umap

        features, labels = generate_synthetic_data(n_samples=100)
        fig = plot_umap(
            features,
            labels,
            title="Test UMAP",
            save_path="./test_umap.png",
            n_neighbors=10,
            min_dist=0.1
        )

        assert os.path.exists("./test_umap.png"), "UMAP plot not saved"
        print("  ✓ UMAP plot test passed!")

        plt.close(fig)
        os.remove("./test_umap.png")
    except ImportError:
        print("  ⚠ UMAP not installed, skipping UMAP test")


def test_layer_wise_metrics_plot():
    """Test layer-wise metrics plot."""
    print("\nTesting layer-wise metrics plot...")

    # Generate synthetic layer-wise metrics
    num_layers = 24
    metrics_dict = {
        'Condition A': np.linspace(0.3, 0.6, num_layers) + np.random.randn(num_layers) * 0.02,
        'Condition B': np.linspace(0.2, 0.5, num_layers) + np.random.randn(num_layers) * 0.02,
    }

    fig = plot_layer_wise_metrics(
        metrics_dict,
        metric_name='Silhouette Score',
        title='Test Layer-wise Metrics',
        save_path='./test_layerwise.png'
    )

    assert os.path.exists("./test_layerwise.png"), "Layer-wise plot not saved"
    print("  ✓ Layer-wise metrics plot test passed!")

    plt.close(fig)
    os.remove("./test_layerwise.png")


def test_similarity_distribution_plot():
    """Test similarity distribution plot."""
    print("\nTesting similarity distribution plot...")

    # Generate synthetic similarity data
    similarities_dict = {
        'High Compression': np.random.beta(2, 5, 1000) * 2 - 1,  # Shift to [-1, 1]
        'Low Compression': np.random.beta(5, 2, 1000) * 2 - 1,
    }

    fig = plot_similarity_distribution(
        similarities_dict,
        title='Test Similarity Distribution',
        save_path='./test_similarity.png',
        plot_type='hist'
    )

    assert os.path.exists("./test_similarity.png"), "Similarity distribution plot not saved"
    print("  ✓ Similarity distribution plot test passed!")

    plt.close(fig)
    os.remove("./test_similarity.png")


def test_layer_wise_similarity_plot():
    """Test layer-wise similarity plot."""
    print("\nTesting layer-wise similarity plot...")

    # Generate synthetic layer-wise similarity data
    num_layers = 24
    similarities_dict = {
        'PYRA': [np.random.beta(5, 2, 50) for _ in range(num_layers)],
        'ToMe+LoRA': [np.random.beta(3, 3, 50) for _ in range(num_layers)],
    }

    fig = plot_layer_wise_similarity(
        similarities_dict,
        title='Test Layer-wise Similarity',
        save_path='./test_layerwise_sim.png'
    )

    assert os.path.exists("./test_layerwise_sim.png"), "Layer-wise similarity plot not saved"
    print("  ✓ Layer-wise similarity plot test passed!")

    plt.close(fig)
    os.remove("./test_layerwise_sim.png")


def test_comparison_grid():
    """Test comparison grid plot."""
    print("\nTesting comparison grid plot...")

    # Generate synthetic data for multiple conditions
    features_dict = {}
    labels = None

    for i, condition in enumerate(['Condition A', 'Condition B', 'Condition C']):
        features, labels_i = generate_synthetic_data(n_samples=50, seed=42+i)
        features_dict[condition] = features
        if labels is None:
            labels = labels_i

    fig = plot_comparison_grid(
        features_dict,
        labels,
        title_prefix='Test',
        save_path='./test_comparison_grid.png'
    )

    assert os.path.exists("./test_comparison_grid.png"), "Comparison grid plot not saved"
    print("  ✓ Comparison grid plot test passed!")

    plt.close(fig)
    os.remove("./test_comparison_grid.png")


def main():
    """Run all tests."""
    print("="*60)
    print("PYRA Visualization System Tests")
    print("="*60)

    try:
        test_clustering_metrics()
        test_knn_accuracy()
        test_linear_probe_accuracy()
        test_tsne_plot()
        test_umap_plot()
        test_layer_wise_metrics_plot()
        test_similarity_distribution_plot()
        test_layer_wise_similarity_plot()
        test_comparison_grid()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        print("\nThe visualization system is ready to use.")
        print("Run 'bash scripts/run_visualization_example.sh' to get started.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
