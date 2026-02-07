#!/bin/bash
# Vision Stage Training Script for SmolVLM2
# Usage: ./scripts/run_vision_stage.sh [256m|500m] [num_gpus]

set -e

MODEL_SIZE=${1:-256m}
NUM_GPUS=${2:-8}
OUTPUT_DIR="./checkpoints/vision_stage_${MODEL_SIZE}"

echo "========================================"
echo "SmolVLM2 Vision Stage Training"
echo "========================================"
echo "Model size: ${MODEL_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

# Set environment variables
export WANDB_PROJECT="smolvlm2-training"
export WANDB_RUN_NAME="smolvlm2-${MODEL_SIZE}-vision"
export TOKENIZERS_PARALLELISM=false

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch with accelerate
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes ${NUM_GPUS} \
    -m src.training.train_vision \
    --model-size ${MODEL_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --max-steps 50000 \
    --warmup-steps 1000 \
    --freeze-vision \
    --unfreeze-vision-after 10000 \
    --bf16 \
    --gradient-checkpointing \
    --flash-attention \
    --logging-steps 10 \
    --save-steps 1000

echo "Vision stage training complete!"
echo "Checkpoint saved to: ${OUTPUT_DIR}"
