#!/bin/bash

# PYRA Compression Visualization Experiments
# This script provides examples for running visualization experiments

# ============================================================================
# Configuration
# ============================================================================

# Dataset settings
DATASET="cifar"  # Options: cifar, caltech101, dtd, flowers102, pets, svhn, sun397, etc.
DATA_PATH="./data/vtab-1k/${DATASET}"
SPLIT="train"  # Options: train, val
SAMPLES_PER_CLASS=10  # Number of samples per class for clear visualization
BATCH_SIZE=32

# Model settings
MODEL_NAME="vit_large_patch16_224_in21k"

# Output directory
OUTPUT_DIR="./visualization_results/${DATASET}"

# Device
DEVICE="cuda"  # Options: cuda, cpu

# ============================================================================
# Experiment 1: Compare PYRA vs ToMe+LoRA (same compression rate)
# ============================================================================

echo "========================================"
echo "Experiment 1: PYRA vs ToMe+LoRA"
echo "========================================"

# Define checkpoint paths (REPLACE WITH YOUR ACTUAL PATHS)
PYRA_CHECKPOINT="./output/${DATASET}/pyra_high/checkpoint.pth"
TOME_LORA_CHECKPOINT="./output/${DATASET}/tome_lora_high/checkpoint.pth"

# Define config paths
PYRA_CONFIG="./experiments/LoRA/ViT-L_prompt_lora_12.yaml"
TOME_LORA_CONFIG="./experiments/LoRA/ViT-L_prompt_lora_12.yaml"

python visualization/visualize_compression.py \
    --dataset ${DATASET} \
    --data-path ${DATA_PATH} \
    --split ${SPLIT} \
    --samples-per-class ${SAMPLES_PER_CLASS} \
    --batch-size ${BATCH_SIZE} \
    --model-name ${MODEL_NAME} \
    --checkpoints ${PYRA_CHECKPOINT} ${TOME_LORA_CHECKPOINT} \
    --configs ${PYRA_CONFIG} ${TOME_LORA_CONFIG} \
    --condition-names "PYRA-High" "ToMe+LoRA-High" \
    --tome-schedules high high \
    --use-pyra True False \
    --use-tome True True \
    --output-dir ${OUTPUT_DIR}/pyra_vs_tome \
    --device ${DEVICE}

# ============================================================================
# Experiment 2: Compare Low vs High Compression Rates
# ============================================================================

echo ""
echo "========================================"
echo "Experiment 2: Low vs High Compression"
echo "========================================"

# Define checkpoint paths
LOW_COMPRESSION_CHECKPOINT="./output/${DATASET}/pyra_low/checkpoint.pth"
HIGH_COMPRESSION_CHECKPOINT="./output/${DATASET}/pyra_high/checkpoint.pth"

# Define config paths
CONFIG="./experiments/LoRA/ViT-L_prompt_lora_12.yaml"

python visualization/visualize_compression.py \
    --dataset ${DATASET} \
    --data-path ${DATA_PATH} \
    --split ${SPLIT} \
    --samples-per-class ${SAMPLES_PER_CLASS} \
    --batch-size ${BATCH_SIZE} \
    --model-name ${MODEL_NAME} \
    --checkpoints ${LOW_COMPRESSION_CHECKPOINT} ${HIGH_COMPRESSION_CHECKPOINT} \
    --configs ${CONFIG} ${CONFIG} \
    --condition-names "PYRA-Low" "PYRA-High" \
    --tome-schedules low high \
    --use-pyra True True \
    --use-tome True True \
    --output-dir ${OUTPUT_DIR}/low_vs_high \
    --device ${DEVICE}

# ============================================================================
# Experiment 3: Comprehensive Comparison (4 conditions)
# ============================================================================

echo ""
echo "========================================"
echo "Experiment 3: Comprehensive Comparison"
echo "========================================"

# Define all checkpoint paths
PYRA_LOW="./output/${DATASET}/pyra_low/checkpoint.pth"
PYRA_HIGH="./output/${DATASET}/pyra_high/checkpoint.pth"
TOME_LOW="./output/${DATASET}/tome_lora_low/checkpoint.pth"
TOME_HIGH="./output/${DATASET}/tome_lora_high/checkpoint.pth"

python visualization/visualize_compression.py \
    --dataset ${DATASET} \
    --data-path ${DATA_PATH} \
    --split ${SPLIT} \
    --samples-per-class ${SAMPLES_PER_CLASS} \
    --batch-size ${BATCH_SIZE} \
    --model-name ${MODEL_NAME} \
    --checkpoints ${PYRA_LOW} ${PYRA_HIGH} ${TOME_LOW} ${TOME_HIGH} \
    --configs ${CONFIG} ${CONFIG} ${CONFIG} ${CONFIG} \
    --condition-names "PYRA-Low" "PYRA-High" "ToMe+LoRA-Low" "ToMe+LoRA-High" \
    --tome-schedules low high low high \
    --use-pyra True True False False \
    --use-tome True True True True \
    --output-dir ${OUTPUT_DIR}/comprehensive \
    --device ${DEVICE}

# ============================================================================
# Experiment 4: Single Model Analysis
# ============================================================================

echo ""
echo "========================================"
echo "Experiment 4: Single Model Analysis"
echo "========================================"

# For analyzing a single trained model in detail
CHECKPOINT="./output/${DATASET}/pyra_high/checkpoint.pth"

python visualization/visualize_compression.py \
    --dataset ${DATASET} \
    --data-path ${DATA_PATH} \
    --split ${SPLIT} \
    --samples-per-class 20 \  # Use more samples for single model analysis
    --batch-size ${BATCH_SIZE} \
    --model-name ${MODEL_NAME} \
    --checkpoints ${CHECKPOINT} \
    --configs ${CONFIG} \
    --condition-names "PYRA-High" \
    --tome-schedules high \
    --use-pyra True \
    --use-tome True \
    --output-dir ${OUTPUT_DIR}/single_model \
    --device ${DEVICE}

echo ""
echo "========================================"
echo "All visualization experiments complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
