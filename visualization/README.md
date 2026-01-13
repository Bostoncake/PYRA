# PYRA Compression Visualization Experiments

This directory contains code for visualizing and quantifying "adverse compression" effects in vision transformers with token merging.

## Overview

The visualization experiments aim to transform "adverse compression" from an observed phenomenon into quantitative evidence in representation space. The system provides:

1. **Representation Space Analysis**: Extract and visualize [CLS] token representations from all transformer layers
2. **Clustering Quality Metrics**: Quantify representation quality using Silhouette score, Davies-Bouldin index, and Calinski-Harabasz index
3. **Layer-wise Analysis**: Track how representation quality evolves across layers
4. **Token Merge Similarity Tracking**: Record cosine similarity of merged token pairs to understand merge quality
5. **Comparison Visualizations**: Compare PYRA vs ToMe+LoRA and different compression rates

## Installation

Install additional dependencies:

```bash
pip install scikit-learn umap-learn seaborn
```

## Quick Start

### Step 1: Prepare Your Data

Ensure your dataset follows the VTAB-1k structure:

```
data/vtab-1k/{dataset}/
├── train800val200.txt
├── test.txt
└── images/
```

### Step 2: Update Checkpoint Paths

Edit `scripts/run_visualization_example.sh` and update:

```bash
PYRA_CHECKPOINT="./output/your_dataset/checkpoint.pth"
BASELINE_CHECKPOINT="./output/your_dataset/baseline.pth"
```

### Step 3: Run Visualization

```bash
bash scripts/run_visualization_example.sh
```

## Experiments

### Experiment 1: Representation Space Visualization

**Goal**: Visualize how well different classes are separated in the representation space.

**Metrics**:
- **t-SNE/UMAP**: 2D visualization of [CLS] token representations
- **Silhouette Score**: Measures how similar samples are to their own cluster vs other clusters
  - Range: [-1, 1], higher is better
  - > 0.5: Strong cluster structure
  - 0.2 - 0.5: Weak cluster structure
  - < 0.2: No meaningful cluster structure
- **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances
  - Lower is better
  - 0 is best (no overlap)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
  - Higher is better

**Output Files**:
- `{condition}_tsne.png`: t-SNE visualization
- `{condition}_umap.png`: UMAP visualization
- `{condition}_metrics.txt`: Numerical metrics
- `{condition}_metrics.json`: Metrics in JSON format

### Experiment 2: Layer-wise Analysis

**Goal**: Track how representation quality evolves across transformer layers.

**Metrics Computed at Each Layer**:
- **Silhouette Score**: Clustering quality
- **k-NN Accuracy**: Classification accuracy using k-nearest neighbors (k=5)
- **Linear Probe Accuracy**: Accuracy of a linear classifier trained on the representations

**Insights**:
- Early layers: Should show gradual improvement as features are extracted
- Middle layers: Where token merging happens, may show quality drops
- Late layers: Final representations, should recover quality (PYRA hypothesis)

**Output Files**:
- `comparison_silhouette.png`: Layer-wise silhouette scores
- `comparison_knn_accuracy.png`: Layer-wise k-NN accuracy
- `comparison_linear_accuracy.png`: Layer-wise linear probe accuracy

### Experiment 3: Token Merge Similarity Analysis

**Goal**: Analyze the cosine similarity of tokens that are merged by ToMe.

**Hypothesis**:
- **Low compression**: Should merge only very similar tokens (high similarity)
- **High compression**: May be forced to merge dissimilar tokens (lower similarity)
- **PYRA vs ToMe+LoRA**: PYRA's adaptive re-activation may make the model more robust to merging dissimilar tokens

**Metrics**:
- **Mean Similarity**: Average cosine similarity of merged token pairs
- **Distribution**: Histogram/violin plot showing similarity distribution
- **Layer-wise Evolution**: How merge quality changes across layers

**Output Files**:
- `comparison_similarity_hist.png`: Histogram of merge similarities
- `comparison_similarity_violin.png`: Violin plot showing distributions
- `comparison_similarity_layerwise.png`: Mean similarity across layers

### Experiment 4: Comparison Grid

**Goal**: Visual comparison of representation spaces across multiple conditions.

**Output Files**:
- `comparison_tsne_grid.png`: Grid of t-SNE plots for each condition

## File Structure

```
visualization/
├── README.md                      # This file
├── feature_extractor.py           # Feature extraction from models
├── vis_utils.py                   # Visualization and metric computation utilities
└── visualize_compression.py       # Main experiment script

scripts/
├── run_visualization.sh           # Full experiment suite
└── run_visualization_example.sh   # Simple example script
```

## Usage Examples

### Example 1: Compare PYRA vs ToMe+LoRA

```bash
python visualization/visualize_compression.py \
    --dataset cifar \
    --data-path ./data/vtab-1k/cifar \
    --split train \
    --samples-per-class 10 \
    --checkpoints model1.pth model2.pth \
    --configs config1.yaml config2.yaml \
    --condition-names "PYRA" "ToMe+LoRA" \
    --tome-schedules high high \
    --use-pyra True False \
    --use-tome True True \
    --output-dir ./results
```

### Example 2: Compare Low vs High Compression

```bash
python visualization/visualize_compression.py \
    --dataset cifar \
    --data-path ./data/vtab-1k/cifar \
    --split train \
    --samples-per-class 10 \
    --checkpoints model_low.pth model_high.pth \
    --configs config.yaml config.yaml \
    --condition-names "Low-Compression" "High-Compression" \
    --tome-schedules low high \
    --use-pyra True True \
    --use-tome True True \
    --output-dir ./results
```

### Example 3: Analyze Single Model

```bash
python visualization/visualize_compression.py \
    --dataset cifar \
    --data-path ./data/vtab-1k/cifar \
    --split train \
    --samples-per-class 20 \
    --checkpoints model.pth \
    --configs config.yaml \
    --condition-names "My-Model" \
    --tome-schedules high \
    --use-pyra True \
    --use-tome True \
    --output-dir ./results
```

## Command-line Arguments

### Data Arguments
- `--dataset`: Dataset name (e.g., cifar, caltech101, flowers102)
- `--data-path`: Path to dataset directory
- `--split`: Dataset split to use ('train' or 'val')
- `--samples-per-class`: Number of samples per class (default: 10)
- `--batch-size`: Batch size for inference (default: 32)

### Model Arguments
- `--model-name`: Model architecture (default: vit_large_patch16_224_in21k)
- `--checkpoints`: List of checkpoint paths (space-separated)
- `--configs`: List of config YAML paths (space-separated, same length as checkpoints)
- `--condition-names`: Names for each condition (space-separated)
- `--tome-schedules`: ToMe schedules ('low' or 'high' for each checkpoint)
- `--use-pyra`: Whether to use PYRA for each checkpoint (True/False)
- `--use-tome`: Whether to use ToMe for each checkpoint (True/False)

### Output Arguments
- `--output-dir`: Directory to save results (default: ./visualization_results)
- `--device`: Device to use ('cuda' or 'cpu')

## Output Structure

```
visualization_results/
├── {dataset}/
│   ├── condition1/
│   │   ├── condition1_tsne.png
│   │   ├── condition1_umap.png
│   │   ├── condition1_metrics.txt
│   │   └── condition1_metrics.json
│   ├── condition2/
│   │   └── ...
│   └── comparisons/
│       ├── comparison_silhouette.png
│       ├── comparison_knn_accuracy.png
│       ├── comparison_linear_accuracy.png
│       ├── comparison_similarity_hist.png
│       ├── comparison_similarity_violin.png
│       ├── comparison_similarity_layerwise.png
│       └── comparison_tsne_grid.png
```

## Tips for Paper Figures

### For Clear Visualizations
1. Use `--samples-per-class 10-15` for uncluttered t-SNE plots
2. Use `--split train` for consistency (larger sample pool)
3. Choose datasets with clear visual distinctions between classes

### For Strong Evidence
1. Compare multiple compression rates (low, medium, high)
2. Always include baseline (ToMe+LoRA without PYRA)
3. Use layer-wise metrics to show where PYRA helps most
4. Use merge similarity distributions to explain "adverse compression"

### Recommended Datasets
- **CIFAR-100**: Many classes, good for showing clustering
- **Caltech101**: Visually distinct classes
- **Flowers102**: High inter-class similarity (challenging case)
- **DTD**: Texture classification (different feature space)

## Interpreting Results

### Good Representation Quality
- **High Silhouette** (> 0.5): Classes are well-separated
- **Low Davies-Bouldin** (< 1.0): Clusters are compact and far apart
- **High k-NN Accuracy**: Nearest neighbors are from same class
- **High Linear Probe Accuracy**: Linearly separable

### Adverse Compression Indicators
- **Decreasing layer-wise metrics**: Quality drops as compression increases
- **Low merge similarity**: Forced to merge dissimilar tokens
- **High variance in similarity**: Inconsistent merge quality

### PYRA Advantage Indicators
- **Better recovery in late layers**: PYRA re-activation helps
- **Higher metrics under high compression**: More robust to aggressive merging
- **More consistent merge quality**: Adaptive weights improve merge decisions

## Troubleshooting

### UMAP not available
Install with: `pip install umap-learn`

### Memory issues
- Reduce `--samples-per-class`
- Reduce `--batch-size`
- Use CPU with `--device cpu`

### Checkpoint loading errors
- Ensure config matches training config
- Check that model architecture matches checkpoint
- Verify checkpoint path is correct

## Citation

If you use this visualization code in your research, please cite:

```bibtex
@inproceedings{wang2024pyra,
  title={PYRA: Parallel Yielding Re-Activation for Training-Efficient Model Compression},
  author={Wang, Yonggan and others},
  booktitle={ECCV},
  year={2024}
}
```
