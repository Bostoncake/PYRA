"""
Feature extraction module for PYRA visualization experiments.
Extracts [CLS] tokens from each layer and tracks token merging similarities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


class FeatureExtractor:
    """
    Extracts features from a Vision Transformer model during forward pass.
    Captures:
    - [CLS] token representations from each layer
    - Cosine similarity of merged token pairs in ToMe
    """

    def __init__(self, model: nn.Module, track_merge_similarity: bool = True):
        """
        Args:
            model: Vision Transformer model (with or without ToMe)
            track_merge_similarity: Whether to track token merge similarities
        """
        self.model = model
        self.track_merge_similarity = track_merge_similarity

        # Storage for extracted features
        self.cls_features = []  # List of [CLS] tokens from each layer
        self.merge_similarities = []  # List of merge similarities from each layer
        self.hooks = []

    def _register_hooks(self):
        """Register forward hooks to extract [CLS] tokens from each block."""
        self.cls_features = []
        self.hooks = []

        # Hook into each transformer block to extract [CLS] token
        for idx, block in enumerate(self.model.blocks):
            hook = block.register_forward_hook(self._make_cls_hook(idx))
            self.hooks.append(hook)

    def _make_cls_hook(self, layer_idx: int):
        """Create a hook function for extracting [CLS] token at a specific layer."""
        def hook_fn(module, input, output):
            # Extract [CLS] token (first token) and store
            cls_token = output[:, 0, :].detach().cpu()
            self.cls_features.append(cls_token)
        return hook_fn

    def _patch_tome_for_similarity_tracking(self):
        """
        Patch ToMe blocks to track cosine similarity of merged tokens.
        This modifies the bipartite_soft_matching function to record similarities.
        """
        if not self.track_merge_similarity:
            return

        # Import here to avoid circular dependency
        from model.adaptive_merge import bipartite_soft_matching as original_bsm

        # Store original function
        self._original_bsm = original_bsm

        # Create patched version
        def patched_bipartite_soft_matching(
            metric: torch.Tensor,
            r: int,
            class_token: bool = False,
            distill_token: bool = False,
        ):
            """
            Patched version that tracks merge similarities.
            """
            B, N, _ = metric.shape

            # Protect tokens (same as original)
            protected = 0
            if class_token:
                protected += 1
            if distill_token:
                protected += 1

            # Compute similarity scores for all pairs
            # metric shape: [B, N, C] where C is the feature dimension
            with torch.no_grad():
                # Normalize for cosine similarity
                metric_norm = F.normalize(metric, p=2, dim=-1)

                # Compute pairwise cosine similarity
                # We'll compute it between src and dst after splitting
                node_max, node_idx = metric.max(dim=-1)
                edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

                unm_idx = edge_idx[..., protected:, :]  # Unmerged Tokens
                src_idx = edge_idx[..., protected::2, :]  # Source tokens (odd)
                dst_idx = edge_idx[..., protected + 1::2, :]  # Destination tokens (even)

                # Limit to r merges
                src_idx = src_idx[..., :r, :]
                dst_idx = dst_idx[..., :r, :]

                # Extract actual tokens for similarity computation
                def gather_tokens(metric, idx):
                    return metric.gather(
                        dim=1,
                        index=idx.expand(B, -1, metric.shape[-1])
                    )

                src_tokens = gather_tokens(metric_norm, src_idx)
                dst_tokens = gather_tokens(metric_norm, dst_idx)

                # Compute cosine similarity
                similarities = (src_tokens * dst_tokens).sum(dim=-1)  # [B, r]

                # Store similarities for this layer
                self.merge_similarities.append(similarities.detach().cpu())

            # Call original function for actual merging
            return original_bsm(metric, r, class_token, distill_token)

        # Monkey patch the function in the model's tome module
        import model.adaptive_merge as adaptive_merge_module
        adaptive_merge_module.bipartite_soft_matching = patched_bipartite_soft_matching

    def _unpatch_tome(self):
        """Restore original ToMe function."""
        if hasattr(self, '_original_bsm'):
            import model.adaptive_merge as adaptive_merge_module
            adaptive_merge_module.bipartite_soft_matching = self._original_bsm

    def extract_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        max_batches: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Extract features from the dataloader.

        Args:
            dataloader: DataLoader with images to process
            device: Device to run inference on
            max_batches: Maximum number of batches to process (None for all)

        Returns:
            Dictionary containing:
                - 'cls_features': List of [CLS] tokens per layer, shape [num_layers, N, D]
                - 'merge_similarities': List of merge similarities per layer, shape [num_layers, N, r]
                - 'labels': Ground truth labels, shape [N]
                - 'predictions': Model predictions, shape [N]
        """
        self.model.eval()
        self.model.to(device)

        # Register hooks
        self._register_hooks()

        # Patch ToMe if tracking similarities
        if self.track_merge_similarity:
            self._patch_tome_for_similarity_tracking()

        all_labels = []
        all_predictions = []
        layer_cls_features = defaultdict(list)
        layer_merge_similarities = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                images = images.to(device)
                labels = labels.to(device)

                # Clear storage for this batch
                self.cls_features = []
                self.merge_similarities = []

                # Forward pass (hooks will capture features)
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)

                # Store labels and predictions
                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())

                # Store [CLS] features by layer
                for layer_idx, cls_feat in enumerate(self.cls_features):
                    layer_cls_features[layer_idx].append(cls_feat)

                # Store merge similarities by layer
                if self.track_merge_similarity:
                    for layer_idx, merge_sim in enumerate(self.merge_similarities):
                        layer_merge_similarities[layer_idx].append(merge_sim)

        # Remove hooks
        for hook in self.hooks:
            hook.remove()

        # Unpatch ToMe
        if self.track_merge_similarity:
            self._unpatch_tome()

        # Concatenate all batches
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_predictions = torch.cat(all_predictions, dim=0).numpy()

        # Concatenate features by layer
        num_layers = len(layer_cls_features)
        cls_features_array = []
        for layer_idx in range(num_layers):
            layer_feat = torch.cat(layer_cls_features[layer_idx], dim=0).numpy()
            cls_features_array.append(layer_feat)

        # Concatenate merge similarities by layer
        merge_similarities_array = []
        if self.track_merge_similarity and layer_merge_similarities:
            num_merge_layers = len(layer_merge_similarities)
            for layer_idx in range(num_merge_layers):
                layer_sim = torch.cat(layer_merge_similarities[layer_idx], dim=0).numpy()
                merge_similarities_array.append(layer_sim)

        return {
            'cls_features': cls_features_array,  # List of [N, D] arrays
            'merge_similarities': merge_similarities_array,  # List of [N, r] arrays
            'labels': all_labels,
            'predictions': all_predictions,
        }


def create_subset_dataloader(
    dataset_name: str,
    data_path: str,
    split: str = 'train',
    samples_per_class: int = 10,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, int]:
    """
    Create a dataloader with a fixed number of samples per class.

    Args:
        dataset_name: Name of the dataset (e.g., 'cifar')
        data_path: Path to the dataset
        split: 'train' or 'val'
        samples_per_class: Number of samples to use per class
        batch_size: Batch size
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility

    Returns:
        DataLoader and number of classes
    """
    import sys
    sys.path.insert(0, '/home/user/PYRA')
    from lib.datasets import get_val_transforms
    from torchvision.datasets import ImageFolder
    import random

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Determine split file
    if split == 'train':
        split_file = 'train800val200.txt'
    else:
        split_file = 'test.txt'

    # Load dataset
    transform = get_val_transforms()

    # Read the split file to get image paths
    split_path = f'{data_path}/{split_file}'
    with open(split_path, 'r') as f:
        lines = f.readlines()

    # Parse image paths and labels
    samples_by_class = defaultdict(list)
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 2:
            img_path = parts[0]
            label = int(parts[1])
            full_path = f'{data_path}/{img_path}'
            samples_by_class[label].append(full_path)

    # Sample fixed number per class
    selected_samples = []
    for label, paths in samples_by_class.items():
        # Shuffle and select
        random.shuffle(paths)
        selected = paths[:samples_per_class]
        selected_samples.extend([(path, label) for path in selected])

    # Create custom dataset
    from torch.utils.data import Dataset
    from PIL import Image

    class SubsetDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

    dataset = SubsetDataset(selected_samples, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_classes = len(samples_by_class)

    return dataloader, num_classes
