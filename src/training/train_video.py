#!/usr/bin/env python3
"""Video Stage Training for SmolVLM2.

Stage 2 of SmolVLM2 training: Fine-tune on video understanding tasks.
Continues from vision stage checkpoint.

Usage:
    # Single GPU
    python -m src.training.train_video \
        --vision-checkpoint ./checkpoints/vision_stage \
        --output-dir ./checkpoints/video_stage

    # Multi-GPU with accelerate
    accelerate launch -m src.training.train_video \
        --vision-checkpoint ./checkpoints/vision_stage
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import set_seed, AutoProcessor, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import SmolVLMConfig
from src.data import (
    VideoStageDataset,
    VideoDataCollator,
    VideoProcessor,
    create_video_processor,
)
from src.training.trainer import SmolVLMTrainer, create_training_args

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SmolVLM2 Video Stage Training")

    # Model
    parser.add_argument(
        "--vision-checkpoint",
        type=str,
        required=True,
        help="Path to vision stage checkpoint",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["256m", "500m"],
        default=None,
        help="Model size (auto-detected from checkpoint if not specified)",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length (longer for video)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum video frames",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming datasets",
    )

    # Training
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/video_stage",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (smaller for video)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (lower for fine-tuning)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Warmup steps",
    )

    # Hardware
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing",
    )

    # Logging
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="smolvlm2-training",
        help="Weights & Biases project name",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def load_vision_checkpoint(checkpoint_path: str):
    """Load model from vision stage checkpoint.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Tuple of (model, processor, tokenizer)
    """
    from transformers import AutoModelForVision2Seq

    logger.info(f"Loading vision stage checkpoint from {checkpoint_path}")

    # Try loading as HuggingFace model
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        return model, processor, tokenizer

    except Exception as e:
        logger.warning(f"Could not load as HF model: {e}")

    # Try loading custom checkpoint
    from src.model import initialize_smolvlm_model

    # Detect model size from config
    config_path = Path(checkpoint_path) / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        # Determine size from hidden dim
        hidden_size = config.get("hidden_size", config.get("text_hidden_size", 576))
        model_size = "500m" if hidden_size >= 960 else "256m"
    else:
        model_size = "256m"

    model, processor, tokenizer = initialize_smolvlm_model(
        model_size=model_size,
        torch_dtype=torch.bfloat16,
    )

    # Load weights
    state_dict_path = Path(checkpoint_path) / "pytorch_model.bin"
    if state_dict_path.exists():
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded model weights")

    return model, processor, tokenizer


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("SmolVLM2 Video Stage Training")
    logger.info("=" * 60)
    logger.info(f"Vision checkpoint: {args.vision_checkpoint}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Max steps: {args.max_steps}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load vision stage checkpoint
    logger.info("Loading vision stage checkpoint...")
    model, processor, tokenizer = load_vision_checkpoint(args.vision_checkpoint)

    # Unfreeze all parameters for video stage
    for param in model.parameters():
        param.requires_grad = True

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Create video processor
    video_processor = create_video_processor(
        image_processor=processor,
        num_frames=args.max_frames,
    )

    # Create dataset
    logger.info("Loading video datasets...")
    train_dataset = VideoStageDataset(
        processor=processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        max_length=args.max_length,
        max_frames=args.max_frames,
        streaming=args.streaming,
    )

    # Create data collator
    from src.data.data_collator import VideoDataCollator
    data_collator = VideoDataCollator(
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_length,
        max_frames=args.max_frames,
    )

    # Create training arguments
    training_args = create_training_args(
        output_dir=args.output_dir,
        model_size=args.model_size or "256m",
        stage="video",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        freeze_vision_encoder=False,  # All unfrozen for video stage
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

    # Set wandb
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_RUN_NAME"] = f"smolvlm2-video-{args.model_size}"

    # Create trainer
    trainer = SmolVLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting video stage training...")
    train_result = trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Video stage training complete!")
    logger.info(f"Final model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
