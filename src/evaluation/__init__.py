"""SmolVLM2 and PerceptionLM Evaluation Pipeline.

This module provides evaluation capabilities for vision-language models
using the lmms-eval framework. It supports SmolVLM2 (256M, 500M, 2.2B)
and PerceptionLM (1B, 3B) models on video and image benchmarks.

Example usage:
    # Command line
    python -m src.evaluation.evaluate \
        --model-path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
        --benchmarks video,plm \
        --bf16

    # Python API
    from src.evaluation import EvaluationRunner, resolve_benchmark_names

    benchmarks = resolve_benchmark_names("video")
    runner = EvaluationRunner(
        model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        benchmarks=benchmarks,
    )
    results = runner.run()
    runner.print_summary()

Available benchmark groups:
    - video: Video-MME, MLVU, MVBench, WorldSense, TempCompass
    - plm: PLM-VideoBench (FGQA, SGQA, RCap, RTLoc, RDCap)
    - image: TextVQA, DocVQA, ChartQA, MMMU, MathVista, etc.
    - all: All benchmarks

Available models:
    SmolVLM2:
        - HuggingFaceTB/SmolVLM2-256M-Video-Instruct
        - HuggingFaceTB/SmolVLM2-500M-Video-Instruct
        - HuggingFaceTB/SmolVLM2-2.2B-Instruct

    PerceptionLM:
        - facebook/Perception-LM-1B
        - facebook/Perception-LM-3B
"""

from .evaluate import main
from .runner import EvaluationRunner

# Benchmark utilities
from .benchmarks.benchmark_configs import (
    BENCHMARK_GROUPS,
    BENCHMARK_NAME_MAP,
    IMAGE_BENCHMARKS,
    PLM_VIDEOBENCH,
    VIDEO_BENCHMARKS,
    get_benchmark_info,
    list_available_benchmarks,
    resolve_benchmark_names,
)

# Model utilities
from .models.model_registry import (
    MODEL_REGISTRY,
    PERCEPTIONLM_MODELS,
    SMOLVLM2_MODELS,
    ModelConfig,
    ModelFamily,
    detect_model_family,
    get_model_config,
    list_available_models,
)

# Result utilities
from .utils.result_parser import (
    EvaluationResult,
    compare_models,
    format_results_table,
    generate_leaderboard,
    load_evaluation_results,
    merge_results,
    parse_lmms_output,
)

# PLM-VideoBench utilities
from .benchmarks.plm_videobench import (
    PLMResults,
    format_plm_results,
    get_plm_tasks,
    parse_plm_results,
)

__all__ = [
    # Main entry points
    "main",
    "EvaluationRunner",
    # Benchmark configs
    "BENCHMARK_GROUPS",
    "BENCHMARK_NAME_MAP",
    "VIDEO_BENCHMARKS",
    "PLM_VIDEOBENCH",
    "IMAGE_BENCHMARKS",
    "resolve_benchmark_names",
    "get_benchmark_info",
    "list_available_benchmarks",
    # Model configs
    "MODEL_REGISTRY",
    "SMOLVLM2_MODELS",
    "PERCEPTIONLM_MODELS",
    "ModelConfig",
    "ModelFamily",
    "get_model_config",
    "detect_model_family",
    "list_available_models",
    # Results
    "EvaluationResult",
    "parse_lmms_output",
    "load_evaluation_results",
    "compare_models",
    "generate_leaderboard",
    "format_results_table",
    "merge_results",
    # PLM-VideoBench
    "PLMResults",
    "parse_plm_results",
    "format_plm_results",
    "get_plm_tasks",
]
