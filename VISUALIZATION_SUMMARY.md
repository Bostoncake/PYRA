# Visualization Experiments Summary

## Created Files

### Core Modules
1. **`visualization/feature_extractor.py`** (331 lines)
   - `FeatureExtractor`: Class for extracting [CLS] tokens and token merge similarities
   - `create_subset_dataloader()`: Create balanced subset for visualization

2. **`visualization/vis_utils.py`** (418 lines)
   - Clustering metrics: `compute_clustering_metrics()`, `compute_knn_accuracy()`, `compute_linear_probe_accuracy()`
   - Visualizations: `plot_tsne()`, `plot_umap()`, `plot_layer_wise_metrics()`, `plot_similarity_distribution()`, `plot_layer_wise_similarity()`, `plot_comparison_grid()`
   - Utilities: `save_metrics_to_file()`

3. **`visualization/visualize_compression.py`** (327 lines)
   - Main experiment orchestrator
   - `load_model()`: Load trained checkpoints
   - `extract_and_visualize()`: Extract features and create visualizations
   - `compare_conditions()`: Create comparison plots

### Scripts
4. **`scripts/run_visualization.sh`** (Executable)
   - Full experiment suite with 4 experiment templates
   - PYRA vs ToMe+LoRA comparison
   - Low vs High compression comparison
   - Comprehensive 4-way comparison
   - Single model analysis

5. **`scripts/run_visualization_example.sh`** (Executable)
   - Simple quick-start example
   - Easy to modify for first-time users

### Testing & Documentation
6. **`visualization/test_visualization.py`** (331 lines)
   - Comprehensive test suite for all visualization functions
   - Uses synthetic data to verify correctness
   - Run with: `python visualization/test_visualization.py`

7. **`visualization/README.md`** (471 lines)
   - Detailed API documentation
   - Command-line argument reference
   - Usage examples
   - Output structure explanation

8. **`VISUALIZATION_GUIDE.md`** (461 lines)
   - User-friendly guide for TPAMI paper
   - Quick start instructions
   - Interpretation guidelines
   - Recommended experiments for paper
   - Troubleshooting section

9. **`visualization/requirements.txt`**
   - Additional dependencies for visualization

10. **`visualization/__init__.py`**
    - Package initialization

## Key Features

### 1. Representation Space Analysis
- Extracts [CLS] tokens from all 24 transformer layers
- Visualizes with t-SNE and UMAP
- Quantifies clustering quality with:
  - Silhouette Score (higher is better)
  - Davies-Bouldin Index (lower is better)
  - Calinski-Harabasz Index (higher is better)

### 2. Layer-wise Analysis
- Tracks metrics across all layers:
  - Silhouette score
  - k-NN accuracy (k=5)
  - Linear probe accuracy
- Shows where compression hurts and where PYRA recovers

### 3. Token Merge Similarity Tracking
- Records cosine similarity of every merged token pair
- Visualizes distributions (histogram, violin plot)
- Layer-wise evolution curves
- Compares PYRA vs ToMe+LoRA and low vs high compression

### 4. Flexible Comparison System
- Compare multiple conditions simultaneously
- Automatic comparison plot generation
- Side-by-side t-SNE grids
- Overlay curves for quantitative comparison

## Usage

### Quick Start
```bash
# Test installation
python visualization/test_visualization.py

# Run simple example (edit checkpoint paths first)
bash scripts/run_visualization_example.sh
```

### Full Experiments
```bash
# Run all experiments
bash scripts/run_visualization.sh
```

### Custom Experiment
```bash
python visualization/visualize_compression.py \
    --dataset cifar \
    --data-path ./data/vtab-1k/cifar \
    --checkpoints model1.pth model2.pth \
    --configs config1.yaml config2.yaml \
    --condition-names "PYRA" "Baseline" \
    --tome-schedules high high \
    --use-pyra True False \
    --use-tome True True
```

## Output Structure

```
visualization_results/
└── {dataset}/
    ├── {condition}/
    │   ├── {condition}_tsne.png          # t-SNE visualization
    │   ├── {condition}_umap.png          # UMAP visualization
    │   ├── {condition}_metrics.txt       # Human-readable metrics
    │   └── {condition}_metrics.json      # Machine-readable metrics
    └── comparisons/
        ├── comparison_silhouette.png           # Layer-wise silhouette
        ├── comparison_knn_accuracy.png         # Layer-wise k-NN accuracy
        ├── comparison_linear_accuracy.png      # Layer-wise linear probe
        ├── comparison_similarity_hist.png      # Similarity histogram
        ├── comparison_similarity_violin.png    # Similarity violin plot
        ├── comparison_similarity_layerwise.png # Layer-wise similarity
        └── comparison_tsne_grid.png            # Side-by-side t-SNE
```

## For TPAMI Paper

### Recommended Figures

1. **Main Figure**: t-SNE comparison + layer-wise silhouette + similarity distribution
   - Shows PYRA maintains better separation
   - Quantifies "adverse compression"

2. **Supplementary Figure 1**: Layer-wise metrics (silhouette, k-NN, linear probe)
   - Shows PYRA's recovery in late layers

3. **Supplementary Figure 2**: Token merge similarity analysis
   - Histogram + layer-wise curves
   - Evidence of forced dissimilar token merging

### Key Narrative Points

1. **Quantitative Evidence**: Transform "adverse compression" from observation to measurement
2. **Layer-wise Evolution**: Show where compression hurts (middle layers) and where PYRA helps (late layers)
3. **Merge Quality**: Demonstrate that high compression forces merging of dissimilar tokens
4. **PYRA Advantage**: Show adaptive re-activation maintains better representation quality

## Technical Details

### Feature Extraction
- Hooks into each transformer block's forward pass
- Extracts [CLS] token (position 0) after each block
- Patches `bipartite_soft_matching` to track merge similarities
- Handles both PYRA and baseline ToMe+LoRA models

### Data Sampling
- Stratified sampling: fixed number of samples per class
- Default: 10 samples per class for clear visualizations
- Adjustable for single-model deep analysis (20+ samples)

### Metrics Computation
- Uses scikit-learn for clustering metrics
- Train/test split for k-NN and linear probe accuracy
- Layer-wise computation for evolutionary analysis

### Visualization
- Matplotlib + Seaborn for publication-quality figures
- t-SNE with perplexity=30, 1000 iterations
- UMAP with n_neighbors=15, min_dist=0.1
- 300 DPI output for paper submission

## Dependencies

```bash
pip install torch torchvision numpy pyyaml matplotlib seaborn scikit-learn umap-learn tqdm
```

Or use:
```bash
pip install -r visualization/requirements.txt
```

## Testing

All visualization functions tested with synthetic data:
```bash
python visualization/test_visualization.py
```

Expected output: All tests passed ✓

## Next Steps

1. **Train Models** (if not already done):
   - PYRA with low and high compression
   - ToMe+LoRA with low and high compression

2. **Update Checkpoint Paths**:
   - Edit `scripts/run_visualization_example.sh`
   - Point to your trained model checkpoints

3. **Run Experiments**:
   - Start with example script
   - Expand to full experiment suite

4. **Create Paper Figures**:
   - Select best visualizations from results
   - Combine into multi-panel figures
   - Add to TPAMI submission

## Support

- **Documentation**: See `visualization/README.md`
- **User Guide**: See `VISUALIZATION_GUIDE.md`
- **Testing**: Run `python visualization/test_visualization.py`
- **Examples**: Check `scripts/` directory

---

Created for TPAMI paper rebuttal - PYRA compression evidence visualization experiments.
