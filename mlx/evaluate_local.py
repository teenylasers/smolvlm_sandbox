#!/usr/bin/env python3
"""Local evaluation of video benchmarks on Apple Silicon.

This script evaluates SmolVLM2 (MLX) and PerceptionLM (PyTorch/MPS) models
on video understanding benchmarks locally on Apple Silicon devices.

Usage:
    # Evaluate SmolVLM2 256M model (MLX)
    python mlx/evaluate_local.py \
        --model-size 256m \
        --benchmarks video-mme \
        --num-samples 100

    # Evaluate PerceptionLM 1B model (PyTorch/MPS)
    python mlx/evaluate_local.py \
        --model-size plm-1b \
        --benchmarks video \
        --num-samples 100

    # Evaluate PerceptionLM 3B model
    python mlx/evaluate_local.py \
        --model-size plm-3b \
        --benchmarks mvbench \
        --num-samples 50

    # Quick test with fewer samples
    python mlx/evaluate_local.py \
        --model-size 256m \
        --benchmarks mvbench \
        --num-samples 10
"""

import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx.benchmark_loader import (
    BenchmarkLoader,
    BenchmarkSample,
    list_benchmarks,
    resolve_benchmarks,
)
from mlx.metrics import (
    EvaluationResults,
    evaluate_benchmark,
    extract_answer,
)
from mlx.mlx_inference import (
    MLX_MODELS,
    MLXModel,
    check_mlx_vlm_installed,
    install_mlx_vlm,
    list_models as list_mlx_models,
)
from mlx.pytorch_inference import (
    PYTORCH_MODELS,
    PyTorchModel,
    check_mps_available,
    list_models as list_pytorch_models,
)

# All available models
ALL_MODELS = {**MLX_MODELS, **PYTORCH_MODELS}


def is_mlx_model(model_size: str) -> bool:
    """Check if model uses MLX backend."""
    return model_size in MLX_MODELS


def is_pytorch_model(model_size: str) -> bool:
    """Check if model uses PyTorch backend."""
    return model_size in PYTORCH_MODELS


def list_all_models() -> str:
    """List all available models."""
    lines = ["Available Models:", ""]
    lines.append("SmolVLM2 (MLX - optimized for Apple Silicon):")
    for size, model_id in MLX_MODELS.items():
        lines.append(f"  {size}: {model_id}")
    lines.append("")
    lines.append("PerceptionLM (PyTorch/MPS):")
    for size, model_id in PYTORCH_MODELS.items():
        lines.append(f"  {size}: {model_id}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Local evaluation of video benchmarks on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # SmolVLM2 quick test (MLX)
    python mlx/evaluate_local.py --model-size 256m --benchmarks mvbench --num-samples 10

    # PerceptionLM evaluation (PyTorch/MPS)
    python mlx/evaluate_local.py --model-size plm-1b --benchmarks video --num-samples 100

    # Full evaluation (100 samples per benchmark)
    python mlx/evaluate_local.py --model-size 500m --benchmarks video --num-samples 100

Available models:
    SmolVLM2 (MLX): 256m, 500m, 2.2b
    PerceptionLM (PyTorch): plm-1b, plm-3b

Available benchmarks: video-mme, mvbench, mlvu, tempcompass
Benchmark groups: video (all video benchmarks)
        """,
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["256m", "500m", "2.2b", "plm-1b", "plm-3b"],
        default="256m",
        help="Model size to evaluate (256m/500m/2.2b for SmolVLM2, plm-1b/plm-3b for PerceptionLM)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="video",
        help="Comma-separated benchmark names or 'video' for all",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per benchmark (default: 100)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Maximum video frames (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results/local",
        help="Directory to save results",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install mlx-vlm if not present",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def evaluate_single_benchmark(
    model: Union[MLXModel, PyTorchModel],
    benchmark_name: str,
    num_samples: int,
    seed: int,
    verbose: bool = False,
) -> "BenchmarkResult":
    """Evaluate a single benchmark.

    Args:
        model: Loaded model (MLXModel or PyTorchModel).
        benchmark_name: Name of the benchmark.
        num_samples: Number of samples to evaluate.
        seed: Random seed for sampling.
        verbose: Print verbose output.

    Returns:
        BenchmarkResult with metrics.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark_name}")
    print(f"{'='*60}")

    # Load benchmark samples
    print(f"Loading {num_samples} samples...")
    loader = BenchmarkLoader(
        benchmark=benchmark_name,
        num_samples=num_samples,
        seed=seed,
    )
    samples = loader.load_samples()

    if not samples:
        print(f"  No samples loaded for {benchmark_name}")
        from mlx.metrics import BenchmarkResult
        return BenchmarkResult(
            benchmark=benchmark_name,
            accuracy=0.0,
            num_samples=0,
            num_correct=0,
        )

    print(f"Loaded {len(samples)} samples")

    # Generate responses
    print(f"Generating responses...")
    responses = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        # Build prompt with question and options
        prompt = sample.get_prompt()

        if verbose:
            print(f"\n  Sample {i+1}/{len(samples)}: {sample.sample_id}")
            print(f"  Question: {sample.question[:100]}...")

        # Generate response
        response = model.generate_response(
            video_path=sample.video_path,
            prompt=prompt,
        )
        responses.append(response)

        # Extract and show answer
        predicted = extract_answer(response)
        correct = predicted.upper() == sample.correct_answer.upper()

        if verbose:
            print(f"  Response: {response[:100]}...")
            print(f"  Predicted: {predicted}, Correct: {sample.correct_answer}, {'OK' if correct else 'WRONG'}")
        else:
            status = "OK" if correct else "X"
            print(f"  [{i+1}/{len(samples)}] {status}", end="\r")

        # Memory cleanup every 10 samples
        if (i + 1) % 10 == 0:
            gc.collect()
            # Also clear MPS cache if using PyTorch
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except (ImportError, AttributeError):
                pass

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(samples):.1f}s per sample)")

    # Compute metrics
    result = evaluate_benchmark(benchmark_name, samples, responses)

    print(f"Accuracy: {result.accuracy:.2%} ({result.num_correct}/{result.num_samples})")

    return result


def main():
    """Main entry point."""
    args = parse_args()

    # Handle list commands
    if args.list_benchmarks:
        print(list_benchmarks())
        return 0

    if args.list_models:
        print(list_all_models())
        return 0

    # Check dependencies based on model type
    if is_mlx_model(args.model_size):
        if not check_mlx_vlm_installed():
            if args.install:
                install_mlx_vlm()
            else:
                print("mlx-vlm not installed. Run with --install or:")
                print("  pip install git+https://github.com/pcuenca/mlx-vlm.git@smolvlm")
                return 1
    elif is_pytorch_model(args.model_size):
        if not check_mps_available():
            print("Warning: MPS not available, will use CPU (slower)")
        # Check for video decoder
        try:
            import decord
        except ImportError:
            try:
                import av
            except ImportError:
                print("No video decoder found. Install one:")
                print("  pip install decord  # or")
                print("  pip install av")
                return 1

    # Resolve benchmarks
    try:
        benchmark_names = resolve_benchmarks(args.benchmarks)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-benchmarks to see available options")
        return 1

    # Determine backend
    if is_mlx_model(args.model_size):
        backend = "MLX"
        model_family = "SmolVLM2"
    else:
        backend = "PyTorch/MPS"
        model_family = "PerceptionLM"

    # Print configuration
    print("\n" + "=" * 60)
    print(f"{model_family} Local Evaluation ({backend})")
    print("=" * 60)
    print(f"Model size:     {args.model_size}")
    print(f"Model ID:       {ALL_MODELS[args.model_size]}")
    print(f"Backend:        {backend}")
    print(f"Benchmarks:     {', '.join(benchmark_names)}")
    print(f"Samples/bench:  {args.num_samples}")
    print(f"Max tokens:     {args.max_tokens}")
    print(f"Max frames:     {args.max_frames}")
    print(f"Random seed:    {args.seed}")
    print(f"Output dir:     {args.output_dir}")
    print("=" * 60)

    # Create model based on backend
    print(f"\nLoading model...")
    if is_mlx_model(args.model_size):
        model = MLXModel(
            model_size=args.model_size,
            max_tokens=args.max_tokens,
            max_frames=args.max_frames,
        )
    else:
        model = PyTorchModel(
            model_size=args.model_size,
            max_tokens=args.max_tokens,
            max_frames=args.max_frames,
        )
    model.load()

    # Create results container
    results = EvaluationResults(
        model_size=args.model_size,
        model_id=ALL_MODELS[args.model_size],
        config={
            "num_samples": args.num_samples,
            "max_tokens": args.max_tokens,
            "max_frames": args.max_frames,
            "seed": args.seed,
            "backend": backend,
        },
    )

    # Evaluate each benchmark
    total_start = time.time()

    for benchmark_name in benchmark_names:
        try:
            result = evaluate_single_benchmark(
                model=model,
                benchmark_name=benchmark_name,
                num_samples=args.num_samples,
                seed=args.seed,
                verbose=args.verbose,
            )
            results.add_benchmark(result)
        except Exception as e:
            print(f"Error evaluating {benchmark_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Memory cleanup between benchmarks
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass

    total_elapsed = time.time() - total_start

    # Print summary
    results.print_summary()
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.model_size}_{timestamp}.json"
    results.save(output_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
