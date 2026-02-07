#!/bin/bash
# Evaluate SmolVLM2 model on benchmarks
# Usage: ./scripts/evaluate_model.sh [model_path] [benchmarks]

set -e

MODEL_PATH=${1:-"./checkpoints/video_stage_256m"}
BENCHMARKS=${2:-"video-mme,mlvu,mvbench"}
OUTPUT_DIR="./evaluation_results"

echo "========================================"
echo "SmolVLM2 Evaluation"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Benchmarks: ${BENCHMARKS}"
echo "========================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run evaluation
python -m src.evaluation.evaluate \
    --model-path ${MODEL_PATH} \
    --benchmarks ${BENCHMARKS} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 16 \
    --bf16

echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
