#!/usr/bin/env python3
"""LoRA fine-tuning experiments on MLX (Apple Silicon).

Run small-scale LoRA fine-tuning experiments locally before
deploying full training to the cluster.

Usage:
    # Quick test with 100 iterations
    python mlx/lora_finetune.py --iters 100 --batch-size 1

    # Train on custom data
    python mlx/lora_finetune.py --data ./my_data --iters 500

    # Use 500M model
    python mlx/lora_finetune.py --model-size 500m --iters 200
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import os


MLX_MODELS = {
    "256m": "mlx-community/SmolVLM2-256M-Video-Instruct-mlx",
    "500m": "mlx-community/SmolVLM2-500M-Video-Instruct-mlx",
}


def create_sample_training_data(output_dir: Path, num_samples: int = 100) -> Path:
    """Create sample training data in JSONL format.

    Args:
        output_dir: Directory to save data
        num_samples: Number of samples to create

    Returns:
        Path to training data directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample data for vision-language training
    samples = [
        {
            "messages": [
                {"role": "user", "content": "<image>Describe this image."},
                {"role": "assistant", "content": "This is a sample description of the image."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "<image>What objects are in this image?"},
                {"role": "assistant", "content": "The image contains various objects including..."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "<image>Is there any text in this image?"},
                {"role": "assistant", "content": "Yes, I can see text that reads..."},
            ]
        },
    ]

    # Create train.jsonl
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for i in range(num_samples):
            sample = samples[i % len(samples)].copy()
            # Add variation
            sample["id"] = i
            f.write(json.dumps(sample) + "\n")

    print(f"Created {num_samples} training samples at {train_path}")

    # Create valid.jsonl (10% of train)
    valid_path = output_dir / "valid.jsonl"
    num_valid = max(10, num_samples // 10)
    with open(valid_path, "w") as f:
        for i in range(num_valid):
            sample = samples[i % len(samples)].copy()
            sample["id"] = f"valid_{i}"
            f.write(json.dumps(sample) + "\n")

    print(f"Created {num_valid} validation samples at {valid_path}")

    return output_dir


def run_lora_training(
    model_id: str,
    data_dir: str,
    output_dir: str,
    iters: int = 100,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    num_layers: int = 4,
    lora_rank: int = 8,
    use_gradient_checkpointing: bool = True,
) -> bool:
    """Run LoRA fine-tuning using mlx-lm.

    Args:
        model_id: HuggingFace model ID
        data_dir: Path to training data directory
        output_dir: Path to save adapters
        iters: Number of training iterations
        batch_size: Batch size (reduce for memory)
        learning_rate: Learning rate
        num_layers: Number of layers to fine-tune
        lora_rank: LoRA rank
        use_gradient_checkpointing: Enable gradient checkpointing

    Returns:
        True if training succeeded
    """
    print(f"\n{'='*60}")
    print(f"Starting LoRA Fine-tuning")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Iterations: {iters}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA layers: {num_layers}")
    print(f"LoRA rank: {lora_rank}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_id,
        "--train",
        "--data", str(data_dir),
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--learning-rate", str(learning_rate),
        "--num-layers", str(num_layers),
        "--lora-rank", str(lora_rank),
        "--adapter-path", str(output_dir),
    ]

    if use_gradient_checkpointing:
        cmd.append("--grad-checkpoint")

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error: {e}")
        return False


def test_finetuned_model(
    model_id: str,
    adapter_path: str,
    prompt: str = "Describe this image in detail.",
) -> bool:
    """Test the fine-tuned model with LoRA adapters.

    Args:
        model_id: Base model ID
        adapter_path: Path to LoRA adapters
        prompt: Test prompt

    Returns:
        True if test succeeded
    """
    print(f"\n{'='*60}")
    print(f"Testing Fine-tuned Model")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", model_id,
        "--adapter-path", adapter_path,
        "--prompt", prompt,
        "--max-tokens", "100",
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Output:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Test failed: {e}")
        print(e.stderr)
        return False


def estimate_memory_usage(model_size: str, batch_size: int, lora_rank: int) -> Dict:
    """Estimate memory usage for training.

    Args:
        model_size: "256m" or "500m"
        batch_size: Training batch size
        lora_rank: LoRA rank

    Returns:
        Dictionary with memory estimates
    """
    # Base memory estimates (GB)
    base_memory = {
        "256m": 1.0,  # 4-bit quantized
        "500m": 1.8,
    }

    # Per-batch memory (approximate)
    batch_memory = 0.5 * batch_size  # Rough estimate

    # LoRA overhead
    lora_memory = 0.1 * (lora_rank / 8)  # Scales with rank

    # Gradient memory (with checkpointing)
    grad_memory = 1.0 * batch_size

    total = base_memory[model_size] + batch_memory + lora_memory + grad_memory

    return {
        "model_memory_gb": base_memory[model_size],
        "batch_memory_gb": batch_memory,
        "lora_memory_gb": lora_memory,
        "gradient_memory_gb": grad_memory,
        "total_estimated_gb": total,
        "recommended_unified_memory_gb": max(16, int(total * 1.5)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning on MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test
    python mlx/lora_finetune.py --iters 100

    # Custom data
    python mlx/lora_finetune.py --data ./my_data --iters 500

    # Memory efficient
    python mlx/lora_finetune.py --batch-size 1 --num-layers 2
        """,
    )

    parser.add_argument(
        "--model-size",
        choices=["256m", "500m"],
        default="256m",
        help="Model size to fine-tune",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data directory (must contain train.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./adapters",
        help="Output directory for LoRA adapters",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (reduce for less memory)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers to fine-tune (reduce for less memory)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample training data",
    )
    parser.add_argument(
        "--estimate-memory",
        action="store_true",
        help="Only estimate memory usage, don't train",
    )
    parser.add_argument(
        "--test-only",
        type=str,
        help="Test existing adapter path instead of training",
    )

    args = parser.parse_args()

    model_id = MLX_MODELS[args.model_size]

    # Estimate memory
    if args.estimate_memory:
        estimates = estimate_memory_usage(
            args.model_size, args.batch_size, args.lora_rank
        )
        print("\nMemory Usage Estimates:")
        print("-" * 40)
        for key, value in estimates.items():
            print(f"  {key}: {value:.2f} GB" if isinstance(value, float) else f"  {key}: {value} GB")
        return

    # Test existing adapters
    if args.test_only:
        success = test_finetuned_model(model_id, args.test_only)
        sys.exit(0 if success else 1)

    # Create or verify training data
    if args.data:
        data_dir = Path(args.data)
        if not (data_dir / "train.jsonl").exists():
            print(f"train.jsonl not found in {data_dir}")
            if args.create_sample_data:
                create_sample_training_data(data_dir)
            else:
                print("Use --create-sample-data to create sample data")
                sys.exit(1)
    else:
        # Create sample data
        data_dir = Path("./sample_training_data")
        create_sample_training_data(data_dir)

    # Run training
    success = run_lora_training(
        model_id=model_id,
        data_dir=str(data_dir),
        output_dir=args.output,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        lora_rank=args.lora_rank,
    )

    if success:
        # Test the fine-tuned model
        test_finetuned_model(model_id, args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
