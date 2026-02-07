#!/bin/bash
# Video Stage Training Script for SmolVLM2
# Usage: ./scripts/run_video_stage.sh [256m|500m] [num_gpus] [vision_checkpoint]

set -e

MODEL_SIZE=${1:-256m}
NUM_GPUS=${2:-8}
VISION_CHECKPOINT=${3:-"./checkpoints/vision_stage_${MODEL_SIZE}"}
OUTPUT_DIR="./checkpoints/video_stage_${MODEL_SIZE}"

echo "========================================"
echo "SmolVLM2 Video Stage Training"
echo "========================================"
echo "Model size: ${MODEL_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Vision checkpoint: ${VISION_CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

# Verify vision checkpoint exists
if [ ! -d "${VISION_CHECKPOINT}" ]; then
    echo "Error: Vision checkpoint not found at ${VISION_CHECKPOINT}"
    echo "Run vision stage training first: ./scripts/run_vision_stage.sh ${MODEL_SIZE}"
    exit 1
fi

# Set environment variables
export WANDB_PROJECT="smolvlm2-training"
export WANDB_RUN_NAME="smolvlm2-${MODEL_SIZE}-video"
export TOKENIZERS_PARALLELISM=false

# Memory optimization (video requires more memory)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch with accelerate
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes ${NUM_GPUS} \
    -m src.training.train_video \
    --vision-checkpoint ${VISION_CHECKPOINT} \
    --model-size ${MODEL_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --max-steps 30000 \
    --warmup-steps 500 \
    --max-frames 32 \
    --bf16 \
    --gradient-checkpointing \
    --logging-steps 10 \
    --save-steps 1000

echo "Video stage training complete!"
echo "Final model saved to: ${OUTPUT_DIR}"
