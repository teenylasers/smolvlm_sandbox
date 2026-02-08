#!/bin/bash
# Setup SmolVLM2 environment based on platform
# Usage: ./scripts/setup_environment.sh [cloud|local|auto]

set -e

MODE=${1:-auto}

# Detect platform
detect_platform() {
    if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        echo "apple_silicon"
    elif [[ "$(uname)" == "Linux" ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

PLATFORM=$(detect_platform)

echo "========================================"
echo "SmolVLM2 Environment Setup"
echo "========================================"
echo "Platform: ${PLATFORM}"
echo "Mode: ${MODE}"
echo "========================================"

# Auto-detect mode based on platform
if [[ "${MODE}" == "auto" ]]; then
    if [[ "${PLATFORM}" == "apple_silicon" ]]; then
        MODE="local"
    else
        MODE="cloud"
    fi
    echo "Auto-detected mode: ${MODE}"
fi

# Install base dependencies
echo ""
echo "Installing base dependencies..."
pip install -e .

if [[ "${MODE}" == "cloud" ]]; then
    echo ""
    echo "Installing cloud/GPU dependencies..."
    pip install -e ".[cloud]"

    # Install flash attention (optional, requires CUDA)
    if command -v nvcc &> /dev/null; then
        echo ""
        echo "CUDA detected. Installing flash-attn..."
        pip install flash-attn --no-build-isolation || echo "flash-attn installation failed (optional)"
    fi

elif [[ "${MODE}" == "local" ]]; then
    echo ""
    echo "Installing local/MLX dependencies..."
    pip install -e ".[local]"

    # Install mlx-vlm with SmolVLM2 support
    echo ""
    echo "Installing mlx-vlm with SmolVLM2 support..."
    pip install git+https://github.com/pcuenca/mlx-vlm.git@smolvlm || echo "mlx-vlm installation failed"

else
    echo "Unknown mode: ${MODE}"
    echo "Usage: ./scripts/setup_environment.sh [cloud|local|auto]"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"

# Print installed packages info
echo ""
echo "Installed video backends:"
python3 -c "
backends = []
try:
    import decord
    backends.append('decord')
except: pass
try:
    import torchvision.io
    backends.append('torchvision')
except: pass
try:
    import av
    backends.append('av')
except: pass
print('  ' + ', '.join(backends) if backends else '  None')
"

if [[ "${MODE}" == "local" ]]; then
    echo ""
    echo "MLX packages:"
    python3 -c "
try:
    import mlx
    print(f'  mlx: {mlx.__version__}')
except: print('  mlx: not installed')
try:
    import mlx_lm
    print(f'  mlx-lm: installed')
except: print('  mlx-lm: not installed')
try:
    import mlx_vlm
    print(f'  mlx-vlm: installed')
except: print('  mlx-vlm: not installed')
"
fi

echo ""
echo "Next steps:"
if [[ "${MODE}" == "cloud" ]]; then
    echo "  1. Download datasets: ./scripts/download_all_datasets.sh"
    echo "  2. Run training: ./scripts/run_vision_stage.sh 256m 8"
else
    echo "  1. Test inference: python mlx/test_inference.py"
    echo "  2. Test data loading: python mlx/test_data_loading.py"
    echo "  3. Run LoRA experiment: python mlx/lora_finetune.py --iters 100"
fi
