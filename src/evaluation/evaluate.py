"""Main evaluation entry point for SmolVLM2 and PerceptionLM models.

This module provides the CLI interface for running benchmark evaluations
using lmms-eval framework.

Usage:
    # Evaluate SmolVLM2 on video benchmarks
    python -m src.evaluation.evaluate \
        --model-path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
        --benchmarks video \
        --bf16

    # Evaluate PerceptionLM on PLM-VideoBench
    python -m src.evaluation.evaluate \
        --model-path facebook/Perception-LM-3B \
        --benchmarks plm-videobench \
        --bf16

    # Evaluate local checkpoint on all benchmarks
    python -m src.evaluation.evaluate \
        --model-path ./checkpoints/video_stage_256m \
        --benchmarks all \
        --output-dir ./results

    # Use the shell script
    ./scripts/evaluate_model.sh HuggingFaceTB/SmolVLM2-256M-Video-Instruct video
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .benchmarks.benchmark_configs import (
    list_available_benchmarks,
    resolve_benchmark_names,
)
from .models.model_registry import get_model_config, list_available_models
from .runner import EvaluationRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments (uses sys.argv if None).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLM2 and PerceptionLM models on VLM benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Video benchmarks on SmolVLM2
    %(prog)s --model-path HuggingFaceTB/SmolVLM2-2.2B-Instruct --benchmarks video

    # PLM-VideoBench on PerceptionLM
    %(prog)s --model-path facebook/Perception-LM-3B --benchmarks plm

    # All benchmarks on local checkpoint
    %(prog)s --model-path ./checkpoints/model --benchmarks all

Available benchmark groups: video, plm, image, all
        """,
    )

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="HuggingFace model ID or path to local checkpoint",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["256m", "500m", "2.2b", "1b", "3b", "8b"],
        help="Model size hint for local checkpoints",
    )

    # Benchmark arguments
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="video",
        help=(
            "Comma-separated benchmark names or groups. "
            "Groups: video, plm, image, all. "
            "Example: 'video-mme,mlvu' or 'video,plm'"
        ),
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

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        default=True,
        help="Save individual sample predictions",
    )
    parser.add_argument(
        "--no-log-samples",
        action="store_false",
        dest="log_samples",
        help="Don't save individual sample predictions",
    )

    # Execution arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision (default)",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use float32 precision instead of bf16",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for distributed evaluation",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum number of video frames to process",
    )

    # Advanced arguments
    parser.add_argument(
        "--use-python-api",
        action="store_true",
        help="Use lmms-eval Python API instead of subprocess",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parsed = parser.parse_args(args)

    # Handle bf16/fp32 conflict
    if parsed.fp32:
        parsed.bf16 = False

    return parsed


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments.

    Args:
        args: Parsed arguments.

    Returns:
        True if valid, False otherwise.
    """
    # Check model path
    model_path = Path(args.model_path)
    is_local = model_path.exists()
    is_hf = "/" in args.model_path and not is_local

    if not is_local and not is_hf:
        logger.warning(
            f"Model path '{args.model_path}' is not a local path or "
            "recognized HuggingFace model ID. Proceeding anyway."
        )

    # Resolve benchmarks to validate them
    try:
        benchmarks = resolve_benchmark_names(args.benchmarks)
        if not benchmarks:
            logger.error("No valid benchmarks specified")
            return False
        logger.info(f"Resolved benchmarks: {', '.join(benchmarks)}")
    except Exception as e:
        logger.error(f"Failed to resolve benchmarks: {e}")
        return False

    # Check output directory
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create output directory: {e}")
        return False

    return True


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for evaluation.

    Args:
        args: Command line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parsed_args = parse_args(args)

    # Handle list commands
    if parsed_args.list_benchmarks:
        print(list_available_benchmarks())
        return 0

    if parsed_args.list_models:
        print(list_available_models())
        return 0

    # Set logging level
    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not validate_args(parsed_args):
        return 1

    # Get model configuration
    try:
        model_config = get_model_config(
            parsed_args.model_path,
            parsed_args.model_size,
        )
        logger.info(f"Model: {model_config.name}")
        logger.info(f"Family: {model_config.family.value}")
        logger.info(f"Parameters: {model_config.parameters}")
    except Exception as e:
        logger.error(f"Failed to get model configuration: {e}")
        return 1

    # Resolve benchmarks
    benchmarks = resolve_benchmark_names(parsed_args.benchmarks)

    # Print evaluation summary
    print("\n" + "=" * 60)
    print("SmolVLM2 / PerceptionLM Evaluation")
    print("=" * 60)
    print(f"Model:      {parsed_args.model_path}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Output:     {parsed_args.output_dir}")
    print(f"Batch size: {parsed_args.batch_size}")
    print(f"Precision:  {'bfloat16' if parsed_args.bf16 else 'float32'}")
    print(f"Max frames: {parsed_args.max_frames}")
    print(f"GPUs:       {parsed_args.num_gpus}")
    print("=" * 60 + "\n")

    # Create and run evaluation
    try:
        runner = EvaluationRunner(
            model_path=parsed_args.model_path,
            benchmarks=benchmarks,
            output_dir=parsed_args.output_dir,
            batch_size=parsed_args.batch_size,
            bf16=parsed_args.bf16,
            num_gpus=parsed_args.num_gpus,
            max_frames=parsed_args.max_frames,
            log_samples=parsed_args.log_samples,
            use_subprocess=not parsed_args.use_python_api,
        )

        results = runner.run()
        runner.print_summary(results)

        # Check for failures
        failures = [
            task for task, result in results.items() if result.get("status") != "success"
        ]

        if failures:
            logger.warning(f"Some benchmarks failed: {', '.join(failures)}")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
