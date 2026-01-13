#!/bin/bash

# ============================================================================
# PYRA Visualization - Quick Start Example
# ============================================================================
# This script provides a simple example to get started with visualization.
# Modify the paths below to match your trained models.
# ============================================================================

# STEP 1: Set your dataset
DATASET="cifar"  # Change this to your dataset name
DATA_PATH="./data/vtab-1k/${DATASET}"

# STEP 2: Set your checkpoint paths (IMPORTANT: Update these!)
# You need to provide paths to your trained model checkpoints
PYRA_CHECKPOINT="./output/${DATASET}_pyra_high/checkpoint.pth"
BASELINE_CHECKPOINT="./output/${DATASET}_tome_lora_high/checkpoint.pth"

# STEP 3: Set your config paths
# These should match the configs used during training
CONFIG_PATH="./experiments/LoRA/ViT-L_prompt_lora_12.yaml"

# STEP 4: Run visualization
python visualization/visualize_compression.py \
    --dataset ${DATASET} \
    --data-path ${DATA_PATH} \
    --split train \
    --samples-per-class 10 \
    --batch-size 32 \
    --model-name vit_large_patch16_224_in21k \
    --checkpoints ${PYRA_CHECKPOINT} ${BASELINE_CHECKPOINT} \
    --configs ${CONFIG_PATH} ${CONFIG_PATH} \
    --condition-names "PYRA" "ToMe+LoRA" \
    --tome-schedules high high \
    --use-pyra True False \
    --use-tome True True \
    --output-dir ./visualization_results/${DATASET} \
    --device cuda

echo ""
echo "Visualization complete! Check ./visualization_results/${DATASET}/"
