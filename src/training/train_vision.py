#!/usr/bin/env python3
"""Vision Stage Training for SmolVLM2.

Stage 1 of SmolVLM2 training: Train on image understanding tasks.

Usage:
    # Single GPU
    python -m src.training.train_vision --model-size 256m --output-dir ./checkpoints

    # Multi-GPU with accelerate
    accelerate launch -m src.training.train_vision --model-size 256m

    # Resume from checkpoint
    python -m src.training.train_vision --resume-from ./checkpoints/checkpoint-10000
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import set_seed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import (
    SmolVLMConfig,
    SmolVLM256MConfig,
    SmolVLM500MConfig,
    initialize_smolvlm_model,
)
from src.data import SmolVLMDataCollator, VisionStageDataset
from src.training.trainer import SmolVLMTrainer, create_training_args

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SmolVLM2 Vision Stage Training")

    # Model
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["256m", "500m"],
        default="256m",
        help="Model size to train",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from checkpoint",
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
        default=2048,
        help="Maximum sequence length",
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
        default="./checkpoints/vision_stage",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Warmup steps",
    )

    # Vision encoder
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        default=True,
        help="Freeze vision encoder initially",
    )
    parser.add_argument(
        "--unfreeze-vision-after",
        type=int,
        default=10000,
        help="Unfreeze vision encoder after N steps",
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
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        default=True,
        help="Use flash attention 2",
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


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("SmolVLM2 Vision Stage Training")
    logger.info("=" * 60)
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Max steps: {args.max_steps}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    logger.info("Initializing model...")
    model, processor, tokenizer = initialize_smolvlm_model(
        model_size=args.model_size,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        use_flash_attention=args.flash_attention,
    )

    logger.info(f"Model parameters: {model.num_parameters():,}")
    logger.info(f"Trainable parameters: {model.num_parameters(trainable_only=True):,}")

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model.text_decoder, 'gradient_checkpointing_enable'):
            model.text_decoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Create dataset
    logger.info("Loading datasets...")
    train_dataset = VisionStageDataset(
        processor=processor,
        tokenizer=tokenizer,
        max_length=args.max_length,
        streaming=args.streaming,
    )

    # Create data collator
    data_collator = SmolVLMDataCollator(
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_length,
    )

    # Create training arguments
    training_args = create_training_args(
        output_dir=args.output_dir,
        model_size=args.model_size,
        stage="vision",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        freeze_vision_encoder=args.freeze_vision,
        unfreeze_vision_after_steps=args.unfreeze_vision_after,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

    # Set wandb project
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Create trainer
    trainer = SmolVLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Resume from checkpoint if specified
    resume_from = args.resume_from

    # Train
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete!")
    logger.info(f"Final model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
