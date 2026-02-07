#!/bin/bash
# Download all datasets for SmolVLM2 training
# Usage: ./scripts/download_all_datasets.sh [output_dir] [stage]

set -e

OUTPUT_DIR=${1:-"./data"}
STAGE=${2:-""}  # vision, video, or empty for all

echo "========================================"
echo "SmolVLM2 Dataset Download"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Stage: ${STAGE:-all}"
echo "========================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Download command
DOWNLOAD_CMD="python -m src.data.download_datasets --output-dir ${OUTPUT_DIR}"

if [ -n "${STAGE}" ]; then
    DOWNLOAD_CMD="${DOWNLOAD_CMD} --stage ${STAGE}"
fi

# Preview first
echo "Previewing datasets..."
python -m src.data.download_datasets --preview

echo ""
read -p "Proceed with download? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting download..."
    ${DOWNLOAD_CMD}
    echo "Download complete!"
else
    echo "Download cancelled."
fi
