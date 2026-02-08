# SmolVLM2 Training Pipeline

Training pipeline for reproducing [SmolVLM2](https://huggingface.co/blog/smolvlm2) 256M and 500M vision-language models from scratch.

## Overview

SmolVLM2 is a family of compact vision-language models that achieve strong performance on image and video understanding tasks while being small enough to run on edge devices. This repository provides:

- **Full pretraining pipeline** from SigLIP + SmolLM2 components
- **Two-stage training**: Vision stage → Video stage
- **Multi-GPU distributed training** with FSDP
- **MLX support** for local testing on Apple Silicon

| Model | Parameters | Vision Encoder | Text Decoder | GPU RAM (Inference) |
|-------|------------|----------------|--------------|---------------------|
| SmolVLM2-256M | ~256M | SigLIP-base (93M) | SmolLM2-135M | 1.4 GB |
| SmolVLM2-500M | ~500M | SigLIP-base (93M) | SmolLM2-360M | 1.8 GB |

## Architecture

```
┌─────────────────┐     ┌────────────────────┐     ┌─────────────────┐
│  SigLIP Encoder │ ──▶ │  Pixel Shuffle     │ ──▶ │  SmolLM2 LLM    │
│  (384×384 img)  │     │  3×3 (9× compress) │     │  (135M/360M)    │
└─────────────────┘     └────────────────────┘     └─────────────────┘
    93M params                 MLP                  135M/360M params
```

- **Vision Encoder**: SigLIP base-patch16-512 (frozen initially, unfrozen after 10k steps)
- **Connector**: 3×3 pixel shuffle + MLP projection (81 visual tokens per 384×384 patch)
- **Text Decoder**: SmolLM2 with extended context (16k tokens, RoPE θ=273k)

## Requirements

### Cloud/GPU Training (Linux + CUDA)

- Python 3.10+
- CUDA 11.8+ with 8× A100 40GB (recommended) or equivalent
- ~500GB storage for datasets

### Local Testing (Apple Silicon)

- macOS with M1/M2/M3/M4 chip
- 16GB+ unified memory
- Python 3.10+

## Installation

### Quick Setup (Auto-detect Platform)

```bash
git clone <repo-url>
cd smolvlm_sandbox

# Auto-detect platform and install dependencies
./scripts/setup_environment.sh
```

### Manual Installation

**Cloud/GPU Environment:**
```bash
pip install -e ".[cloud]"

# Optional: Flash Attention (requires CUDA)
pip install flash-attn --no-build-isolation
```

**Local/Apple Silicon:**
```bash
pip install -e ".[local]"

# Install mlx-vlm with SmolVLM2 support
pip install git+https://github.com/pcuenca/mlx-vlm.git@smolvlm
```

## Quick Start

### 1. Test Locally (Apple Silicon)

```bash
# Test inference with pretrained model
python mlx/test_inference.py --model-size 256m

# Test with your own image
python mlx/test_inference.py --image photo.jpg --prompt "Describe this image"

# Validate data loading pipeline
python mlx/test_data_loading.py
```

### 2. Download Datasets

```bash
# Preview available datasets
python -m src.data.download_datasets --preview

# Download vision stage datasets
./scripts/download_all_datasets.sh ./data vision

# Download all datasets (vision + video)
./scripts/download_all_datasets.sh ./data
```

### 3. Train Models

```bash
# Vision Stage (Stage 1) - 8 GPUs
./scripts/run_vision_stage.sh 256m 8

# Video Stage (Stage 2) - continues from vision checkpoint
./scripts/run_video_stage.sh 256m 8 ./checkpoints/vision_stage_256m
```

## Project Structure

```
smolvlm_sandbox/
├── configs/
│   ├── accelerate_config.yaml     # FSDP distributed training config
│   ├── model_256m.yaml            # 256M model architecture
│   ├── model_500m.yaml            # 500M model architecture
│   ├── train_vision_stage.yaml    # Vision stage hyperparameters
│   └── train_video_stage.yaml     # Video stage hyperparameters
│
├── src/
│   ├── model/
│   │   ├── smolvlm_config.py      # Model configuration classes
│   │   ├── pixel_shuffle.py       # 3×3 pixel shuffle connector
│   │   └── model_init.py          # Initialize from pretrained components
│   │
│   ├── data/
│   │   ├── download_datasets.py   # Dataset download scripts
│   │   ├── dataset_loaders.py     # Load The Cauldron, Docmatix, etc.
│   │   ├── video_processor.py     # Multi-backend video processing
│   │   ├── data_collator.py       # Multi-modal batch collation
│   │   └── data_mixer.py          # Weighted dataset mixing
│   │
│   ├── training/
│   │   ├── trainer.py             # Custom trainer with vision unfreezing
│   │   ├── train_vision.py        # Stage 1: Vision training
│   │   └── train_video.py         # Stage 2: Video training
│   │
│   └── evaluation/
│       └── evaluate.py            # Benchmark evaluation
│
├── mlx/                           # Apple Silicon local testing
│   ├── requirements_mlx.txt       # MLX-specific dependencies
│   ├── test_inference.py          # Test pretrained models
│   ├── test_data_loading.py       # Validate data pipeline
│   └── lora_finetune.py           # LoRA fine-tuning experiments
│
├── scripts/
│   ├── setup_environment.sh       # Auto-setup for cloud/local
│   ├── download_all_datasets.sh   # Download training data
│   ├── run_vision_stage.sh        # Launch vision training
│   ├── run_video_stage.sh         # Launch video training
│   └── evaluate_model.sh          # Run benchmarks
│
├── pyproject.toml
├── requirements.txt               # Cloud/GPU dependencies
└── README.md
```

## Training Pipeline

### Stage 1: Vision Training

Trains on image understanding tasks using The Cauldron and Docmatix datasets.

| Setting        | Value                                 |
|----------------|---------------------------------------|
| Datasets       | The Cauldron (35%), Docmatix (41%), others (24%) |
| Batch Size     | 256 (8 × 4 × 8 GPUs)                  |
| Learning Rate  | 1e-4 (cosine schedule)                |
| Steps          | 50,000                                |
| Vision Encoder | Frozen first 10k steps, then unfrozen |

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    -m src.training.train_vision \
    --model-size 256m \
    --output-dir ./checkpoints/vision_stage
```

### Stage 2: Video Training

Fine-tunes on video understanding with 3.3M samples from 10 datasets.

| Setting       | Value                        |
|---------------|------------------------------|
| Data Mix      | Image 34.4%, Video 33.0%, Text 20.2%, Multi-image 12.3% |
| Batch Size    | 256 (4 × 8 × 8 GPUs)         |
| Learning Rate | 2e-5 (lower for fine-tuning) |
| Steps         | 30,000                       |
| Max Frames    | 32 per video                 |

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    -m src.training.train_video \
    --vision-checkpoint ./checkpoints/vision_stage \
    --output-dir ./checkpoints/video_stage
```

## Datasets

### Vision Stage
| Dataset      | Source                       | Purpose                |
|--------------|------------------------------|------------------------|
| The Cauldron | `HuggingFaceM4/the_cauldron` | Multi-task VQA         |
| Docmatix     | `HuggingFaceM4/Docmatix`     | Document understanding |

### Video Stage
| Dataset          | Source                               | Modality |
|------------------|--------------------------------------|----------|
| LLaVA-OneVision  | `lmms-lab/LLaVA-OneVision-Data`      | Image    |
| MAmmoTH-VL.      | `MAmmoTH-VL/MAmmoTH-VL-Instruct-12M` | Image    |
| M4-Instruct      | `lmms-lab/M4-Instruct-Data`          | Multi-image |
| LLaVA-Video-178K | `lmms-lab/LLaVA-Video-178K`          | Video    |
| FineVideo        | `HuggingFaceFV/finevideo`            | Video    |
| Video-STAR       | `orrzohar/Video-STaR`                | Video    |
| Vript            | `Mutonix/Vript`                      | Video    |
| Vista-400K       | `TIGER-Lab/VISTA-400K`               | Video    |
| MovieChat        | `Enxin/MovieChat-1K_train`           | Video    |
| ShareGPT4Video   | `ShareGPT4Video/ShareGPT4Video`      | Video    |

## MLX Local Testing (Apple Silicon)

Test the training pipeline locally before deploying to GPU cluster.

### Inference Testing

```bash
# Image inference
python mlx/test_inference.py \
    --model-size 256m \
    --image path/to/image.jpg \
    --prompt "Describe this image"

# Video inference
python mlx/test_inference.py \
    --model-size 500m \
    --video path/to/video.mp4 \
    --prompt "What happens in this video?"
```

### LoRA Fine-tuning Experiments

```bash
# Quick test (100 iterations)
python mlx/lora_finetune.py --model-size 256m --iters 100

# With custom data
python mlx/lora_finetune.py \
    --model-size 256m \
    --data ./my_training_data \
    --iters 500 \
    --batch-size 1

# Estimate memory usage
python mlx/lora_finetune.py --model-size 500m --estimate-memory
```

### Memory Requirements (Apple Silicon)

| Model        | Inference | LoRA Training |
|--------------|-----------|---------------|
| 256M (4-bit) | ~1 GB     | ~4 GB         |
| 500M (4-bit) | ~1.8 GB   | ~6 GB         |

## Video Processing Backends

The pipeline automatically selects the best video backend for your platform:

| Platform      | Primary     | Fallback    |
|---------------|-------------|-------------|
| Linux         | decord      | torchvision |
| Apple Silicon | torchvision | av (PyAV)   |

Note: `decord` does not work on Apple Silicon. The pipeline uses `torchvision.io` instead.

## Configuration

### Model Configuration (`configs/model_*.yaml`)

```yaml
model:
  vision_encoder:
    name: google/siglip-base-patch16-512
    image_size: 384
  text_decoder:
    name: HuggingFaceTB/SmolLM2-135M-Instruct
    max_position_embeddings: 16384
    rope_theta: 273000
  connector:
    type: pixel_shuffle
    ratio: 3  # 9× compression
```

### Training Configuration (`configs/train_*.yaml`)

```yaml
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  max_steps: 50000
  freeze_vision_encoder: true
  unfreeze_vision_after_steps: 10000
  bf16: true
  gradient_checkpointing: true
```

### Distributed Training (`configs/accelerate_config.yaml`)

```yaml
distributed_type: FSDP
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
mixed_precision: bf16
num_processes: 8
```

## Evaluation

Run benchmarks on trained models:

```bash
./scripts/evaluate_model.sh ./checkpoints/video_stage_256m "video-mme,mlvu,mvbench"
```

### Benchmarks

| Benchmark | Type                | Metrics  |
|-----------|---------------------|----------|
| Video-MME | Video understanding | Accuracy |
| MLVU      | Multi-turn video    | Accuracy |
| MVBench   | Video reasoning     | Accuracy |
| DocVQA    | Document QA         | ANLS     |
| TextVQA   | Text in images      | Accuracy |
| ChartQA   | Chart understanding | Accuracy |

## Troubleshooting

### Out of Memory (GPU)

```bash
# Reduce batch size
--batch-size 4 --gradient-accumulation-steps 8

# Enable gradient checkpointing (already default)
--gradient-checkpointing

# Use FSDP CPU offloading (slower but saves memory)
# Edit configs/accelerate_config.yaml:
fsdp_offload_params: true
```

### Video Processing Errors (Apple Silicon)

```bash
# Check available backends
python -c "
import platform
print(f'Platform: {platform.machine()}')

try:
    import torchvision.io
    print('torchvision: available')
except: print('torchvision: not available')

try:
    import av
    print('av: available')
except: print('av: not available')
"

# Install torchvision if missing
pip install torchvision>=0.16.0
```

### Flash Attention Installation

```bash
# Requires CUDA and matching PyTorch version
pip install flash-attn --no-build-isolation

# If it fails, training will fall back to eager attention
```

## Citation

```bibtex
@article{marafioti2025smolvlm,
  title={SmolVLM: Redefining small and efficient multimodal models},
  author={Marafioti, Andr{\'e}s and Zohar, Orr and Farr{\'e}, Miquel and others},
  journal={arXiv preprint arXiv:2504.05299},
  year={2025}
}
```

## References

- [SmolVLM2 Blog Post](https://huggingface.co/blog/smolvlm2)
- [SmolVLM 256M & 500M Blog](https://huggingface.co/blog/smolervlm)
- [SmolVLM2-256M Model Card](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)
- [SmolVLM2-500M Model Card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
- [Technical Paper (arXiv:2504.05299)](https://arxiv.org/abs/2504.05299)
- [HuggingFace SmolLM Repository](https://github.com/huggingface/smollm)
- [MLX-VLM (Apple Silicon)](https://github.com/Blaizzy/mlx-vlm)

## License

Apache 2.0
