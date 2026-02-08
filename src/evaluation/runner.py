"""lmms-eval orchestration for SmolVLM2 and PerceptionLM evaluation.

This module provides the EvaluationRunner class that handles running
lmms-eval benchmarks and collecting results.
"""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .benchmarks.benchmark_configs import get_benchmark_info
from .models.model_registry import get_model_config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRunner:
    """Orchestrates lmms-eval for VLM evaluation.

    This class manages running benchmarks through lmms-eval, either by
    invoking it as a subprocess or using its Python API directly.

    Args:
        model_path: HuggingFace model ID or path to local checkpoint.
        benchmarks: List of lmms-eval task names to run.
        output_dir: Directory to save results.
        batch_size: Batch size for evaluation.
        bf16: Use bfloat16 precision.
        num_gpus: Number of GPUs for distributed evaluation.
        max_frames: Maximum video frames.
        log_samples: Save individual sample predictions.
        use_subprocess: Run lmms-eval as subprocess vs Python API.
    """

    model_path: str
    benchmarks: List[str]
    output_dir: str = "./evaluation_results"
    batch_size: int = 16
    bf16: bool = True
    num_gpus: int = 1
    max_frames: int = 32
    log_samples: bool = True
    use_subprocess: bool = True

    # Internal state
    _results: Dict[str, Dict] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Get model configuration
        self._model_config = get_model_config(self.model_path)

        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self._model_config.name.replace(" ", "_")
        self.run_dir = self.output_path / f"{model_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Evaluation run directory: {self.run_dir}")

    def _build_model_args(self) -> str:
        """Build model arguments string for lmms-eval."""
        args = [f"pretrained={self.model_path}"]

        if self.bf16:
            args.append("dtype=bfloat16")

        args.append(f"max_frames_num={self.max_frames}")

        return ",".join(args)

    def _build_command(self, task: str) -> List[str]:
        """Build lmms-eval command for a specific task.

        Args:
            task: The lmms-eval task name.

        Returns:
            Command as list of strings.
        """
        model_args = self._build_model_args()
        task_output = self.run_dir / task

        cmd = [
            sys.executable,
            "-m",
            "lmms_eval",
            "--model",
            "vlm",
            "--model_args",
            model_args,
            "--tasks",
            task,
            "--batch_size",
            str(self.batch_size),
            "--output_path",
            str(task_output),
        ]

        if self.log_samples:
            cmd.append("--log_samples")

        return cmd

    def _run_subprocess(self, task: str) -> Dict:
        """Run lmms-eval as subprocess for a task.

        Args:
            task: The lmms-eval task name.

        Returns:
            Dictionary with task results or error info.
        """
        cmd = self._build_command(task)

        # Use accelerate for multi-GPU
        if self.num_gpus > 1:
            accelerate_cmd = [
                "accelerate",
                "launch",
                f"--num_processes={self.num_gpus}",
            ]
            cmd = accelerate_cmd + cmd

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout per task
            )

            if result.returncode != 0:
                logger.error(f"Task {task} failed: {result.stderr}")
                return {
                    "task": task,
                    "status": "error",
                    "error": result.stderr,
                    "stdout": result.stdout,
                }

            # Parse results from output directory
            return self._parse_task_results(task)

        except subprocess.TimeoutExpired:
            logger.error(f"Task {task} timed out")
            return {
                "task": task,
                "status": "timeout",
                "error": "Task exceeded 2 hour timeout",
            }
        except Exception as e:
            logger.error(f"Task {task} failed with exception: {e}")
            return {
                "task": task,
                "status": "error",
                "error": str(e),
            }

    def _run_python_api(self, task: str) -> Dict:
        """Run lmms-eval using Python API for a task.

        Args:
            task: The lmms-eval task name.

        Returns:
            Dictionary with task results or error info.
        """
        try:
            from lmms_eval import evaluator
            from lmms_eval.tasks import TaskManager

            # Initialize task manager
            task_manager = TaskManager()

            # Run evaluation
            results = evaluator.simple_evaluate(
                model="vlm",
                model_args=self._build_model_args(),
                tasks=[task],
                batch_size=self.batch_size,
                log_samples=self.log_samples,
                task_manager=task_manager,
            )

            # Save results
            task_output = self.run_dir / task
            task_output.mkdir(parents=True, exist_ok=True)

            with open(task_output / "results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

            return {
                "task": task,
                "status": "success",
                "results": results.get("results", {}).get(task, {}),
            }

        except ImportError as e:
            logger.error(f"lmms_eval not available: {e}")
            return {
                "task": task,
                "status": "error",
                "error": f"lmms_eval not installed: {e}",
            }
        except Exception as e:
            logger.error(f"Task {task} failed: {e}")
            return {
                "task": task,
                "status": "error",
                "error": str(e),
            }

    def _parse_task_results(self, task: str) -> Dict:
        """Parse lmms-eval output files for a task.

        Args:
            task: The task name.

        Returns:
            Dictionary with parsed results.
        """
        task_dir = self.run_dir / task

        # Look for results file (lmms-eval creates various output files)
        possible_files = [
            task_dir / "results.json",
            task_dir / f"{task}_results.json",
        ]

        for result_file in possible_files:
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        data = json.load(f)

                    # Extract metrics from lmms-eval format
                    results = data.get("results", {}).get(task, data.get("results", {}))

                    return {
                        "task": task,
                        "status": "success",
                        "results": results,
                    }
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse {result_file}: {e}")

        # Also check for results in parent directory
        for json_file in task_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if "results" in data:
                    results = data.get("results", {}).get(task, data.get("results", {}))
                    return {
                        "task": task,
                        "status": "success",
                        "results": results,
                    }
            except (json.JSONDecodeError, Exception):
                continue

        return {
            "task": task,
            "status": "error",
            "error": "Results file not found or could not be parsed",
        }

    def run_task(self, task: str) -> Dict:
        """Run evaluation for a single task.

        Args:
            task: The lmms-eval task name.

        Returns:
            Dictionary with task results.
        """
        logger.info(f"Evaluating on {task}...")

        benchmark_info = get_benchmark_info(task)
        logger.info(f"  Benchmark: {benchmark_info.get('name', task)}")
        logger.info(f"  Modality: {benchmark_info.get('modality', 'unknown')}")

        if self.use_subprocess:
            result = self._run_subprocess(task)
        else:
            result = self._run_python_api(task)

        self._results[task] = result
        return result

    def run(self) -> Dict[str, Dict]:
        """Run all benchmarks.

        Returns:
            Dictionary mapping task names to results.
        """
        logger.info(f"Starting evaluation of {self._model_config.name}")
        logger.info(f"Benchmarks: {', '.join(self.benchmarks)}")
        logger.info(f"Output directory: {self.run_dir}")

        for task in self.benchmarks:
            self.run_task(task)

        # Save aggregated results
        self._save_aggregated_results()

        return self._results

    def _save_aggregated_results(self):
        """Save combined results to a single file."""
        aggregated = {
            "model": {
                "path": self.model_path,
                "name": self._model_config.name,
                "family": self._model_config.family.value,
                "parameters": self._model_config.parameters,
            },
            "config": {
                "batch_size": self.batch_size,
                "bf16": self.bf16,
                "max_frames": self.max_frames,
                "num_gpus": self.num_gpus,
            },
            "timestamp": datetime.now().isoformat(),
            "results": self._results,
        }

        output_file = self.run_dir / "aggregated_results.json"
        with open(output_file, "w") as f:
            json.dump(aggregated, f, indent=2, default=str)

        logger.info(f"Aggregated results saved to {output_file}")

    def print_summary(self, results: Optional[Dict[str, Dict]] = None):
        """Print formatted results summary.

        Args:
            results: Results dictionary (uses internal results if not provided).
        """
        results = results or self._results

        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS: {self._model_config.name}")
        print("=" * 70)
        print(f"Model: {self.model_path}")
        print(f"Parameters: {self._model_config.parameters}")
        print("-" * 70)

        # Group by status
        successful = []
        failed = []

        for task, result in results.items():
            if result.get("status") == "success":
                successful.append((task, result))
            else:
                failed.append((task, result))

        # Print successful results
        if successful:
            print("\nSuccessful Evaluations:")
            print("-" * 70)

            for task, result in successful:
                info = get_benchmark_info(task)
                task_results = result.get("results", {})

                # Extract primary metric
                if isinstance(task_results, dict):
                    # Find accuracy or first metric
                    primary_value = (
                        task_results.get("accuracy")
                        or task_results.get("acc")
                        or task_results.get("exact_match")
                        or next(iter(task_results.values()), "N/A")
                    )

                    if isinstance(primary_value, (int, float)):
                        primary_value = f"{primary_value:.2%}"
                else:
                    primary_value = str(task_results)

                print(f"  {info.get('name', task):30} {primary_value}")

        # Print failed results
        if failed:
            print("\nFailed Evaluations:")
            print("-" * 70)

            for task, result in failed:
                info = get_benchmark_info(task)
                error = result.get("error", "Unknown error")[:50]
                print(f"  {info.get('name', task):30} ERROR: {error}")

        print("=" * 70)
        print(f"Results saved to: {self.run_dir}")
        print()

    def get_results(self) -> Dict[str, Dict]:
        """Get the current results.

        Returns:
            Dictionary mapping task names to results.
        """
        return self._results.copy()
