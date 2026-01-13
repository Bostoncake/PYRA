# PYRA Visualization Experiments Guide

This guide explains the visualization experiments created for your TPAMI paper rebuttal.

## 📋 Overview

The visualization system provides comprehensive evidence for "adverse compression" by transforming it from a phenomenon into quantitative representation space evidence. It includes:

### Core Experiments

1. **Representation Space Visualization** (t-SNE/UMAP)
   - Visualizes how well classes are separated in the [CLS] token representation space
   - Compares PYRA vs ToMe+LoRA and different compression rates

2. **Quantitative Clustering Metrics**
   - **Silhouette Score**: Measures cluster separation quality
   - **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances
   - **Calinski-Harabasz Index**: Between-cluster vs within-cluster variance

3. **Layer-wise Analysis**
   - Tracks representation quality evolution across all 24 transformer layers
   - Metrics: Silhouette score, k-NN accuracy, linear probe accuracy
   - Shows where compression hurts and where PYRA recovers

4. **Token Merge Similarity Tracking**
   - Records cosine similarity of every merged token pair
   - Compares distributions: low vs high compression, PYRA vs ToMe+LoRA
   - Shows whether models are forced to merge dissimilar tokens

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r visualization/requirements.txt
```

### Step 2: Test the System

```bash
cd /home/user/PYRA
python visualization/test_visualization.py
```

This will verify all visualization functions work correctly.

### Step 3: Prepare Your Checkpoints

You need trained model checkpoints for comparison. Typical setup:

```
output/
├── cifar_pyra_high/checkpoint.pth       # PYRA with high compression
├── cifar_pyra_low/checkpoint.pth        # PYRA with low compression
├── cifar_tome_lora_high/checkpoint.pth  # ToMe+LoRA with high compression
└── cifar_tome_lora_low/checkpoint.pth   # ToMe+LoRA with low compression
```

### Step 4: Run Visualizations

Edit `scripts/run_visualization_example.sh` to point to your checkpoints:

```bash
# Update these paths
PYRA_CHECKPOINT="./output/cifar_pyra_high/checkpoint.pth"
BASELINE_CHECKPOINT="./output/cifar_tome_lora_high/checkpoint.pth"
```

Then run:

```bash
bash scripts/run_visualization_example.sh
```

## 📊 Understanding the Results

### Directory Structure

```
visualization_results/
└── cifar/
    ├── PYRA/
    │   ├── PYRA_tsne.png              # t-SNE visualization
    │   ├── PYRA_umap.png              # UMAP visualization
    │   ├── PYRA_metrics.txt           # Human-readable metrics
    │   └── PYRA_metrics.json          # Machine-readable metrics
    ├── ToMe+LoRA/
    │   └── ...
    └── comparisons/
        ├── comparison_silhouette.png           # Layer-wise silhouette comparison
        ├── comparison_knn_accuracy.png         # Layer-wise k-NN accuracy
        ├── comparison_linear_accuracy.png      # Layer-wise linear probe accuracy
        ├── comparison_similarity_hist.png      # Merge similarity histogram
        ├── comparison_similarity_violin.png    # Merge similarity violin plot
        ├── comparison_similarity_layerwise.png # Layer-wise merge similarity
        └── comparison_tsne_grid.png            # Side-by-side t-SNE comparison
```

### Key Metrics

#### Silhouette Score
- **Range**: [-1, 1]
- **Interpretation**:
  - > 0.5: Strong cluster structure
  - 0.2 - 0.5: Weak cluster structure
  - < 0.2: No meaningful clusters
- **For your paper**: Higher score = better representation quality

#### Davies-Bouldin Index
- **Range**: [0, ∞)
- **Interpretation**: Lower is better
- **For your paper**: Show that PYRA has lower DB index (tighter clusters)

#### Calinski-Harabasz Index
- **Range**: [0, ∞)
- **Interpretation**: Higher is better
- **For your paper**: Show that PYRA has higher CH index (better separation)

#### Token Merge Similarity
- **Range**: [-1, 1] (cosine similarity)
- **Interpretation**:
  - High (> 0.8): Merging similar tokens (good)
  - Low (< 0.5): Forced to merge dissimilar tokens (adverse compression)
- **For your paper**: Show that:
  - High compression → lower similarity
  - PYRA maintains higher similarity or is more robust to low similarity

## 🎯 Recommended Experiments for TPAMI

### Experiment 1: PYRA vs ToMe+LoRA (Same Compression)

**Goal**: Show PYRA preserves better representation quality

```bash
python visualization/visualize_compression.py \
    --dataset cifar \
    --data-path ./data/vtab-1k/cifar \
    --checkpoints pyra_high.pth tome_lora_high.pth \
    --configs config.yaml config.yaml \
    --condition-names "PYRA" "ToMe+LoRA" \
    --tome-schedules high high \
    --use-pyra True False \
    --use-tome True True
```

**Expected Results**:
- PYRA should have higher silhouette score
- PYRA's layer-wise metrics should recover better in late layers
- PYRA's merge similarities should be more consistent

### Experiment 2: Low vs High Compression

**Goal**: Show adverse compression effects quantitatively

```bash
python visualization/visualize_compression.py \
    --dataset cifar \
    --data-path ./data/vtab-1k/cifar \
    --checkpoints pyra_low.pth pyra_high.pth \
    --configs config.yaml config.yaml \
    --condition-names "Low-Compression" "High-Compression" \
    --tome-schedules low high \
    --use-pyra True True \
    --use-tome True True
```

**Expected Results**:
- High compression should show lower clustering metrics
- High compression should have lower merge similarities
- Layer-wise curves should show where compression hurts most

### Experiment 3: Comprehensive 4-Way Comparison

**Goal**: Complete picture of all conditions

```bash
bash scripts/run_visualization.sh
```

This runs all experiments and creates a comprehensive comparison.

## 📝 Tips for Your TPAMI Paper

### Figure Selection

1. **Main Figure (2x2 grid)**:
   - Top: t-SNE for PYRA vs ToMe+LoRA (high compression)
   - Bottom: Layer-wise silhouette + merge similarity distribution
   - **Message**: PYRA maintains better separation and merge quality

2. **Supplementary Figure 1**:
   - Layer-wise metrics comparison (all three: silhouette, k-NN, linear probe)
   - **Message**: PYRA's adaptive re-activation helps across all layers

3. **Supplementary Figure 2**:
   - Merge similarity analysis: histogram + layer-wise curves
   - **Message**: Quantify "adverse compression" via merge similarity

### Narrative for Rebuttal

> "We provide quantitative evidence for adverse compression effects through comprehensive representation space analysis. We extract [CLS] token representations from all 24 layers of ViT-Large and analyze them using clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and classification probes (k-NN, linear).
>
> **Key Findings:**
>
> 1. **Representation Quality** (Figure X): t-SNE visualization shows PYRA maintains better class separation under high compression (Silhouette: 0.XX vs 0.XX for ToMe+LoRA).
>
> 2. **Layer-wise Evolution** (Figure Y): Our analysis reveals representation quality degradation in middle layers (where token merging occurs), with PYRA showing superior recovery in late layers (layers 18-24) due to adaptive re-activation.
>
> 3. **Token Merge Quality** (Figure Z): By tracking cosine similarity of all merged token pairs, we quantify 'adverse compression': high compression rates force merging of dissimilar tokens (mean similarity: 0.XX), while PYRA's adaptive weights maintain more consistent merge quality (mean similarity: 0.XX).
>
> These experiments transform 'adverse compression' from a qualitative observation into measurable representation space evidence."

## 🔧 Advanced Usage

### Customize Sampling

For datasets with many classes, use stratified sampling:

```bash
--samples-per-class 5  # Fewer samples for cleaner t-SNE
```

For single-model deep analysis:

```bash
--samples-per-class 20  # More samples for better statistics
```

### Multiple Datasets

Run experiments on multiple datasets to show generalization:

```bash
for dataset in cifar caltech101 flowers102 dtd; do
    bash scripts/run_visualization_example.sh
done
```

### Custom Configurations

Modify `visualize_compression.py` to:
- Change t-SNE perplexity (line in plot_tsne call)
- Adjust k in k-NN (default k=5)
- Use different train/test splits for linear probe

## 🐛 Troubleshooting

### Issue: "Cannot load checkpoint"

**Solution**: Verify checkpoint paths and model architecture match:

```python
import torch
ckpt = torch.load('checkpoint.pth', map_location='cpu')
print(ckpt.keys())  # Should show 'model' or 'state_dict'
```

### Issue: "UMAP not installed"

**Solution**: Install UMAP (optional):

```bash
pip install umap-learn
```

Or skip UMAP plots (t-SNE is sufficient).

### Issue: "Out of memory"

**Solutions**:
1. Reduce samples: `--samples-per-class 5`
2. Reduce batch size: `--batch-size 16`
3. Use CPU: `--device cpu`

### Issue: "Silhouette score is NaN"

**Cause**: Too few samples or only one class

**Solution**: Increase `--samples-per-class` or check dataset split file

## 📚 File Reference

### Core Files

- `visualization/feature_extractor.py`: Extracts features and merge similarities
- `visualization/vis_utils.py`: Plotting and metric computation
- `visualization/visualize_compression.py`: Main experiment orchestrator
- `visualization/test_visualization.py`: System tests

### Scripts

- `scripts/run_visualization_example.sh`: Simple single comparison
- `scripts/run_visualization.sh`: Full experiment suite

### Documentation

- `visualization/README.md`: Detailed API documentation
- `VISUALIZATION_GUIDE.md`: This file (user guide)

## 🎓 Citation

```bibtex
@inproceedings{wang2024pyra,
  title={PYRA: Parallel Yielding Re-Activation for Training-Efficient Model Compression},
  author={Wang, Yonggan and others},
  booktitle={ECCV},
  year={2024}
}
```

## ❓ Questions?

If you encounter issues:

1. Run test script: `python visualization/test_visualization.py`
2. Check README: `visualization/README.md`
3. Verify checkpoint paths are correct
4. Ensure dataset split files exist

Good luck with your TPAMI rebuttal! 🚀
