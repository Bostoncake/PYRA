"""
Main visualization script for PYRA compression evidence experiments.
Compares PYRA vs ToMe+LoRA and different compression rates.
"""

import os
import sys
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, '/home/user/PYRA')

from feature_extractor import FeatureExtractor, create_subset_dataloader
from vis_utils import (
    compute_clustering_metrics,
    compute_knn_accuracy,
    compute_linear_probe_accuracy,
    plot_tsne,
    plot_umap,
    plot_layer_wise_metrics,
    plot_similarity_distribution,
    plot_layer_wise_similarity,
    plot_comparison_grid,
    save_metrics_to_file
)


def load_model(
    cfg_path: str,
    checkpoint_path: str,
    model_name: str,
    device: str = 'cuda',
    use_tome: bool = True,
    tome_schedule: str = 'high',
    use_pyra: bool = False
):
    """
    Load a trained model from checkpoint.

    Args:
        cfg_path: Path to config YAML file
        checkpoint_path: Path to model checkpoint
        model_name: Model architecture name
        device: Device to load model on
        use_tome: Whether to apply ToMe
        tome_schedule: ToMe compression schedule ('low', 'high')
        use_pyra: Whether to use PYRA

    Returns:
        Loaded model
    """
    import model as models
    from model.tome import apply_tome, get_merging_schedule
    from types import SimpleNamespace

    # Load config
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Convert to namespace for attribute access
    cfg = SimpleNamespace(**cfg_dict)

    # Determine PYRA schedule
    if use_tome:
        tome_r = get_merging_schedule(model_name, tome_schedule)
        pyra_r = tome_r if use_pyra else None
    else:
        tome_r = None
        pyra_r = None

    # Build model (same as train.py)
    model = models.__dict__[model_name](
        img_size=224,
        drop_rate=0.0,
        drop_path_rate=0.0,
        prompt_tuning_dim=cfg.VISUAL_PROMPT_DIM if hasattr(cfg, 'VISUAL_PROMPT_DIM') else 0,
        LoRA_dim=cfg.LORA_DIM if hasattr(cfg, 'LORA_DIM') else 0,
        adapter_dim=cfg.ADAPTER_DIM if hasattr(cfg, 'ADAPTER_DIM') else 0,
        prefix_dim=cfg.PREFIX_DIM if hasattr(cfg, 'PREFIX_DIM') else 0,
        drop_rate_LoRA=0.0,
        drop_rate_prompt=0.0,
        drop_rate_adapter=0.0,
        IS_not_position_VPT=False,
        pyra_r=pyra_r
    )

    # Apply ToMe if requested
    if use_tome:
        apply_tome(model)
        model.r = tome_r
        print(f"Applied ToMe with schedule '{tome_schedule}': {tome_r}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_ema' in checkpoint:
        state_dict = checkpoint['model_ema']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"  Warning: Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys: {len(unexpected_keys)}")

    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Config: {cfg_path}")
    print(f"  ToMe: {use_tome}, Schedule: {tome_schedule if use_tome else 'N/A'}")
    print(f"  PYRA: {use_pyra}")

    return model


def extract_and_visualize(
    model,
    dataloader,
    condition_name: str,
    output_dir: str,
    device: str = 'cuda',
    track_similarities: bool = True
):
    """
    Extract features and create visualizations for a single condition.

    Args:
        model: Trained model
        dataloader: DataLoader with images
        condition_name: Name of this condition (for plot titles)
        output_dir: Directory to save results
        device: Device to run on
        track_similarities: Whether to track token merge similarities

    Returns:
        Dictionary with extracted features and computed metrics
    """
    print(f"\n{'='*60}")
    print(f"Processing: {condition_name}")
    print(f"{'='*60}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract features
    extractor = FeatureExtractor(model, track_merge_similarity=track_similarities)
    features_data = extractor.extract_features(dataloader, device=device)

    cls_features = features_data['cls_features']
    labels = features_data['labels']
    predictions = features_data['predictions']
    merge_similarities = features_data['merge_similarities']

    print(f"Extracted features from {len(cls_features)} layers")
    print(f"Number of samples: {len(labels)}")
    print(f"Number of classes: {len(np.unique(labels))}")

    # Compute metrics for last layer
    last_layer_features = cls_features[-1]
    print(f"\n--- Last Layer Metrics ---")

    clustering_metrics = compute_clustering_metrics(last_layer_features, labels)
    for metric, value in clustering_metrics.items():
        print(f"{metric}: {value:.4f}")

    knn_acc = compute_knn_accuracy(last_layer_features, labels, k=5)
    print(f"k-NN accuracy (k=5): {knn_acc:.4f}")

    linear_probe_acc = compute_linear_probe_accuracy(last_layer_features, labels)
    print(f"Linear probe accuracy: {linear_probe_acc:.4f}")

    # Compute layer-wise metrics
    print(f"\n--- Layer-wise Metrics ---")
    layer_silhouette = []
    layer_knn_acc = []
    layer_linear_acc = []

    for layer_idx, layer_features in enumerate(cls_features):
        # Silhouette score
        try:
            sil = compute_clustering_metrics(layer_features, labels)['silhouette']
        except:
            sil = np.nan
        layer_silhouette.append(sil)

        # k-NN accuracy (skip if too few samples)
        if len(labels) >= 10:
            try:
                knn = compute_knn_accuracy(layer_features, labels, k=5)
            except:
                knn = np.nan
        else:
            knn = np.nan
        layer_knn_acc.append(knn)

        # Linear probe accuracy
        if len(labels) >= 10:
            try:
                linear = compute_linear_probe_accuracy(layer_features, labels)
            except:
                linear = np.nan
        else:
            linear = np.nan
        layer_linear_acc.append(linear)

        if layer_idx % 4 == 0:
            print(f"Layer {layer_idx:2d}: Sil={sil:.3f}, kNN={knn:.3f}, Linear={linear:.3f}")

    # Create visualizations
    print(f"\n--- Creating Visualizations ---")

    # t-SNE visualization
    plot_tsne(
        last_layer_features,
        labels,
        title=f"t-SNE: {condition_name}",
        save_path=os.path.join(output_dir, f"{condition_name}_tsne.png")
    )

    # UMAP visualization (if available)
    try:
        plot_umap(
            last_layer_features,
            labels,
            title=f"UMAP: {condition_name}",
            save_path=os.path.join(output_dir, f"{condition_name}_umap.png")
        )
    except Exception as e:
        print(f"Could not create UMAP plot: {e}")

    # Save all metrics
    all_metrics = {
        'condition': condition_name,
        'last_layer': clustering_metrics,
        'last_layer_knn_acc': knn_acc,
        'last_layer_linear_acc': linear_probe_acc,
        'layer_wise_silhouette': layer_silhouette,
        'layer_wise_knn_acc': layer_knn_acc,
        'layer_wise_linear_acc': layer_linear_acc,
    }

    save_metrics_to_file(
        all_metrics,
        os.path.join(output_dir, f"{condition_name}_metrics.txt")
    )

    # Save as JSON for easy loading
    # Convert numpy types to Python types for JSON serialization
    json_metrics = {}
    for k, v in all_metrics.items():
        if isinstance(v, dict):
            json_metrics[k] = {kk: float(vv) if not np.isnan(vv) else None for kk, vv in v.items()}
        elif isinstance(v, list):
            json_metrics[k] = [float(x) if not np.isnan(x) else None for x in v]
        else:
            json_metrics[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v

    with open(os.path.join(output_dir, f"{condition_name}_metrics.json"), 'w') as f:
        json.dump(json_metrics, f, indent=2)

    return {
        'cls_features': cls_features,
        'merge_similarities': merge_similarities,
        'labels': labels,
        'predictions': predictions,
        'metrics': all_metrics,
    }


def compare_conditions(
    results_dict: dict,
    output_dir: str
):
    """
    Create comparison plots across multiple conditions.

    Args:
        results_dict: Dictionary mapping condition names to results
        output_dir: Directory to save comparison plots
    """
    print(f"\n{'='*60}")
    print("Creating Comparison Plots")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Extract data for comparison
    conditions = list(results_dict.keys())

    # Compare layer-wise silhouette scores
    silhouette_dict = {
        cond: results['metrics']['layer_wise_silhouette']
        for cond, results in results_dict.items()
    }
    plot_layer_wise_metrics(
        silhouette_dict,
        metric_name='Silhouette Score',
        title='Layer-wise Silhouette Score Comparison',
        save_path=os.path.join(output_dir, 'comparison_silhouette.png'),
        ylabel='Silhouette Score (higher is better)'
    )

    # Compare layer-wise k-NN accuracy
    knn_dict = {
        cond: results['metrics']['layer_wise_knn_acc']
        for cond, results in results_dict.items()
    }
    plot_layer_wise_metrics(
        knn_dict,
        metric_name='k-NN Accuracy',
        title='Layer-wise k-NN Accuracy Comparison',
        save_path=os.path.join(output_dir, 'comparison_knn_accuracy.png'),
        ylabel='k-NN Accuracy'
    )

    # Compare layer-wise linear probe accuracy
    linear_dict = {
        cond: results['metrics']['layer_wise_linear_acc']
        for cond, results in results_dict.items()
    }
    plot_layer_wise_metrics(
        linear_dict,
        metric_name='Linear Probe Accuracy',
        title='Layer-wise Linear Probe Accuracy Comparison',
        save_path=os.path.join(output_dir, 'comparison_linear_accuracy.png'),
        ylabel='Linear Probe Accuracy'
    )

    # Compare merge similarities (if available)
    sim_dict = {}
    for cond, results in results_dict.items():
        if results['merge_similarities']:
            # Concatenate all layers
            all_sims = np.concatenate([s.flatten() for s in results['merge_similarities']])
            sim_dict[cond] = all_sims

    if sim_dict:
        # Histogram comparison
        plot_similarity_distribution(
            sim_dict,
            title='Token Merge Similarity Distribution (All Layers)',
            save_path=os.path.join(output_dir, 'comparison_similarity_hist.png'),
            plot_type='hist'
        )

        # Violin plot comparison
        plot_similarity_distribution(
            sim_dict,
            title='Token Merge Similarity Distribution (All Layers)',
            save_path=os.path.join(output_dir, 'comparison_similarity_violin.png'),
            plot_type='violin'
        )

        # Layer-wise similarity comparison
        sim_layer_dict = {
            cond: results['merge_similarities']
            for cond, results in results_dict.items()
            if results['merge_similarities']
        }
        plot_layer_wise_similarity(
            sim_layer_dict,
            title='Layer-wise Mean Token Merge Similarity',
            save_path=os.path.join(output_dir, 'comparison_similarity_layerwise.png')
        )

    # Create t-SNE comparison grid for last layer
    last_layer_features_dict = {
        cond: results['cls_features'][-1]
        for cond, results in results_dict.items()
    }
    labels = results_dict[conditions[0]]['labels']  # Same labels for all conditions

    plot_comparison_grid(
        last_layer_features_dict,
        labels,
        title_prefix='Last Layer -',
        save_path=os.path.join(output_dir, 'comparison_tsne_grid.png')
    )

    print("Comparison plots created successfully!")


def main():
    parser = argparse.ArgumentParser(description='PYRA Compression Visualization Experiments')

    # Data arguments
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., cifar, caltech101)')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--samples-per-class', type=int, default=10, help='Samples per class for visualization')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='vit_large_patch16_224_in21k', help='Model architecture')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help='List of checkpoint paths')
    parser.add_argument('--configs', type=str, nargs='+', required=True, help='List of config paths (same length as checkpoints)')
    parser.add_argument('--condition-names', type=str, nargs='+', required=True, help='Names for each condition')
    parser.add_argument('--tome-schedules', type=str, nargs='+', default=None, help='ToMe schedules for each checkpoint (low/high)')
    parser.add_argument('--use-pyra', action='store_true', nargs='+', default=None, help='Whether to use PYRA for each checkpoint')
    parser.add_argument('--use-tome', action='store_true', nargs='+', default=None, help='Whether to use ToMe for each checkpoint')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./visualization_results', help='Output directory')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

    args = parser.parse_args()

    # Validate inputs
    num_conditions = len(args.checkpoints)
    assert len(args.configs) == num_conditions, "Number of configs must match checkpoints"
    assert len(args.condition_names) == num_conditions, "Number of condition names must match checkpoints"

    # Set default values for optional lists
    if args.tome_schedules is None:
        args.tome_schedules = ['high'] * num_conditions
    else:
        assert len(args.tome_schedules) == num_conditions

    if args.use_pyra is None:
        args.use_pyra = [False] * num_conditions
    else:
        assert len(args.use_pyra) == num_conditions

    if args.use_tome is None:
        args.use_tome = [True] * num_conditions
    else:
        assert len(args.use_tome) == num_conditions

    print(f"{'='*60}")
    print("PYRA Compression Visualization Experiments")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Number of conditions: {num_conditions}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataloader
    print(f"\n{'='*60}")
    print("Creating Dataloader")
    print(f"{'='*60}")

    dataloader, num_classes = create_subset_dataloader(
        args.dataset,
        args.data_path,
        split=args.split,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size
    )

    print(f"Created dataloader with {len(dataloader.dataset)} samples")
    print(f"Number of classes: {num_classes}")

    # Process each condition
    results_dict = {}

    for i in range(num_conditions):
        condition_name = args.condition_names[i]
        checkpoint_path = args.checkpoints[i]
        config_path = args.configs[i]
        tome_schedule = args.tome_schedules[i]
        use_pyra = args.use_pyra[i]
        use_tome = args.use_tome[i]

        # Load model
        model = load_model(
            config_path,
            checkpoint_path,
            args.model_name,
            device=args.device,
            use_tome=use_tome,
            tome_schedule=tome_schedule,
            use_pyra=use_pyra
        )

        # Extract and visualize
        condition_output_dir = os.path.join(args.output_dir, condition_name)
        results = extract_and_visualize(
            model,
            dataloader,
            condition_name,
            condition_output_dir,
            device=args.device,
            track_similarities=use_tome
        )

        results_dict[condition_name] = results

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Create comparison plots
    if len(results_dict) > 1:
        compare_conditions(results_dict, os.path.join(args.output_dir, 'comparisons'))

    print(f"\n{'='*60}")
    print("Visualization Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
