"""Parse and aggregate evaluation results.

This module provides utilities for loading, parsing, and comparing
evaluation results from lmms-eval runs.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..benchmarks.benchmark_configs import get_benchmark_info


@dataclass
class EvaluationResult:
    """Container for evaluation results from a single model run.

    Attributes:
        model_path: HuggingFace model ID or local checkpoint path.
        model_name: Human-readable model name.
        model_family: Model family (smolvlm2 or perceptionlm).
        parameters: Model parameter count (e.g., "256M", "3B").
        benchmarks: Dictionary mapping task names to their metrics.
        config: Evaluation configuration used.
        timestamp: When the evaluation was run.
    """

    model_path: str
    model_name: str = ""
    model_family: str = ""
    parameters: str = ""
    benchmarks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def add_result(self, task: str, metrics: Dict[str, Any]):
        """Add results for a task.

        Args:
            task: The lmms-eval task name.
            metrics: Dictionary of metric names to values.
        """
        self.benchmarks[task] = metrics

    def get_metric(
        self, task: str, metric: str, default: Any = None
    ) -> Any:
        """Get a specific metric value.

        Args:
            task: The lmms-eval task name.
            metric: The metric name (e.g., "accuracy", "bleu").
            default: Value to return if not found.

        Returns:
            The metric value or default.
        """
        return self.benchmarks.get(task, {}).get(metric, default)

    def get_primary_metric(self, task: str) -> Optional[float]:
        """Get the primary metric for a task.

        The primary metric is typically "accuracy" for most benchmarks.

        Args:
            task: The lmms-eval task name.

        Returns:
            The primary metric value or None.
        """
        metrics = self.benchmarks.get(task, {})

        # Try common metric names in order of preference
        for metric_name in ["accuracy", "acc", "exact_match", "bleu", "rouge"]:
            if metric_name in metrics:
                return metrics[metric_name]

        # Return first numeric value if no standard metric found
        for value in metrics.values():
            if isinstance(value, (int, float)):
                return value

        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary format.

        Returns:
            Dictionary representation of the results.
        """
        return {
            "model": {
                "path": self.model_path,
                "name": self.model_name,
                "family": self.model_family,
                "parameters": self.parameters,
            },
            "config": self.config,
            "timestamp": self.timestamp,
            "results": self.benchmarks,
        }

    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis.

        Returns:
            pandas DataFrame with task, metric, value columns.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for DataFrame conversion. "
                "Install with: pip install pandas"
            )

        rows = []
        for task, metrics in self.benchmarks.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    rows.append({
                        "model": self.model_name or self.model_path,
                        "task": task,
                        "metric": metric_name,
                        "value": value,
                    })

        return pd.DataFrame(rows)

    def save(self, path: Union[str, Path]):
        """Save results to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationResult":
        """Load results from JSON file.

        Args:
            path: Path to results JSON file.

        Returns:
            EvaluationResult object.
        """
        with open(path) as f:
            data = json.load(f)

        model_data = data.get("model", {})

        result = cls(
            model_path=model_data.get("path", ""),
            model_name=model_data.get("name", ""),
            model_family=model_data.get("family", ""),
            parameters=model_data.get("parameters", ""),
            config=data.get("config", {}),
            timestamp=data.get("timestamp", ""),
        )
        result.benchmarks = data.get("results", {})

        return result


def parse_lmms_output(output_dir: Union[str, Path]) -> Dict[str, Dict]:
    """Parse lmms-eval output directory.

    Scans the output directory for task result files and extracts metrics.

    Args:
        output_dir: Directory containing lmms-eval outputs.

    Returns:
        Dictionary mapping task names to their metrics.
    """
    output_dir = Path(output_dir)
    results: Dict[str, Dict] = {}

    # Check for aggregated results first
    aggregated_file = output_dir / "aggregated_results.json"
    if aggregated_file.exists():
        try:
            with open(aggregated_file) as f:
                data = json.load(f)

            # Extract results from aggregated format
            for task, task_data in data.get("results", {}).items():
                if isinstance(task_data, dict):
                    if "results" in task_data:
                        results[task] = task_data["results"]
                    elif task_data.get("status") == "success":
                        results[task] = task_data.get("results", {})

            if results:
                return results
        except (json.JSONDecodeError, Exception):
            pass

    # Scan subdirectories for individual task results
    for item in output_dir.iterdir():
        if item.is_dir():
            task_name = item.name

            # Look for results.json in task directory
            for json_file in item.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    # Extract metrics from lmms-eval format
                    if "results" in data:
                        task_results = data["results"]
                        if task_name in task_results:
                            results[task_name] = task_results[task_name]
                        else:
                            # Use first result if task name not found
                            results[task_name] = next(iter(task_results.values()), {})
                        break
                except (json.JSONDecodeError, Exception):
                    continue

    return results


def load_evaluation_results(path: Union[str, Path]) -> EvaluationResult:
    """Load evaluation results from file or directory.

    Args:
        path: Path to results JSON file or lmms-eval output directory.

    Returns:
        EvaluationResult object.
    """
    path = Path(path)

    if path.is_file():
        return EvaluationResult.load(path)

    # It's a directory - parse lmms-eval output
    benchmarks = parse_lmms_output(path)

    # Try to load aggregated results for model info
    aggregated_file = path / "aggregated_results.json"
    if aggregated_file.exists():
        try:
            with open(aggregated_file) as f:
                data = json.load(f)

            model_data = data.get("model", {})
            result = EvaluationResult(
                model_path=model_data.get("path", str(path)),
                model_name=model_data.get("name", ""),
                model_family=model_data.get("family", ""),
                parameters=model_data.get("parameters", ""),
                config=data.get("config", {}),
                timestamp=data.get("timestamp", ""),
            )
            result.benchmarks = benchmarks
            return result
        except Exception:
            pass

    # Create basic result with parsed benchmarks
    result = EvaluationResult(model_path=str(path))
    result.benchmarks = benchmarks
    return result


def compare_models(
    results: List[EvaluationResult],
    tasks: Optional[List[str]] = None,
    metric: str = "accuracy",
):
    """Compare multiple model evaluations.

    Args:
        results: List of EvaluationResult objects.
        tasks: Optional filter for specific tasks.
        metric: Metric to compare (default: "accuracy").

    Returns:
        pandas DataFrame with model comparison (pivot table).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for model comparison. "
            "Install with: pip install pandas"
        )

    rows = []

    for result in results:
        model_name = result.model_name or result.parameters or result.model_path

        for task, metrics in result.benchmarks.items():
            if tasks and task not in tasks:
                continue

            # Get the specified metric or primary metric
            value = metrics.get(metric)
            if value is None:
                value = result.get_primary_metric(task)

            if value is not None:
                rows.append({
                    "model": model_name,
                    "task": task,
                    "value": value,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Pivot for easier comparison
    pivot = df.pivot_table(
        index="task",
        columns="model",
        values="value",
        aggfunc="first",
    )

    return pivot


def generate_leaderboard(
    results: List[EvaluationResult],
    primary_metric: str = "accuracy",
    tasks: Optional[List[str]] = None,
):
    """Generate a leaderboard ranking models by average performance.

    Args:
        results: List of EvaluationResult objects.
        primary_metric: Metric to rank by (default: "accuracy").
        tasks: Optional filter for specific tasks.

    Returns:
        pandas DataFrame with leaderboard (sorted by avg score).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for leaderboard generation. "
            "Install with: pip install pandas"
        )

    rows = []

    for result in results:
        model_name = result.model_name or result.parameters or result.model_path
        scores = []
        task_count = 0

        for task, metrics in result.benchmarks.items():
            if tasks and task not in tasks:
                continue

            value = metrics.get(primary_metric)
            if value is None:
                value = result.get_primary_metric(task)

            if isinstance(value, (int, float)):
                scores.append(value)
                task_count += 1

        if scores:
            rows.append({
                "model": model_name,
                "family": result.model_family,
                "parameters": result.parameters,
                f"avg_{primary_metric}": sum(scores) / len(scores),
                "num_tasks": task_count,
                "total_score": sum(scores),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(f"avg_{primary_metric}", ascending=False)


def format_results_table(
    results: EvaluationResult,
    include_description: bool = True,
) -> str:
    """Format results as a readable table string.

    Args:
        results: EvaluationResult object.
        include_description: Include benchmark descriptions.

    Returns:
        Formatted table string.
    """
    lines = [
        "=" * 70,
        f"Evaluation Results: {results.model_name or results.model_path}",
        "=" * 70,
    ]

    if results.parameters:
        lines.append(f"Parameters: {results.parameters}")
    if results.model_family:
        lines.append(f"Family: {results.model_family}")

    lines.append("-" * 70)

    # Group by modality
    video_tasks = []
    image_tasks = []
    other_tasks = []

    for task in results.benchmarks.keys():
        info = get_benchmark_info(task)
        modality = info.get("modality", "unknown")

        if modality == "video":
            video_tasks.append(task)
        elif modality == "image":
            image_tasks.append(task)
        else:
            other_tasks.append(task)

    # Print each group
    for group_name, task_list in [
        ("Video Benchmarks", video_tasks),
        ("Image Benchmarks", image_tasks),
        ("Other Benchmarks", other_tasks),
    ]:
        if not task_list:
            continue

        lines.append(f"\n{group_name}:")
        lines.append("-" * 50)

        for task in sorted(task_list):
            info = get_benchmark_info(task)
            display_name = info.get("name", task)
            metrics = results.benchmarks[task]

            # Get primary metric
            primary = results.get_primary_metric(task)
            if primary is not None:
                if isinstance(primary, float) and primary <= 1.0:
                    metric_str = f"{primary:.2%}"
                else:
                    metric_str = f"{primary:.4f}"
            else:
                metric_str = "N/A"

            line = f"  {display_name:35} {metric_str:>10}"

            if include_description and info.get("description"):
                line += f"  ({info['description'][:30]}...)"

            lines.append(line)

    lines.append("=" * 70)

    return "\n".join(lines)


def merge_results(
    *results: EvaluationResult,
    model_name: Optional[str] = None,
) -> EvaluationResult:
    """Merge multiple evaluation results into one.

    Useful when running benchmarks in separate batches.

    Args:
        *results: EvaluationResult objects to merge.
        model_name: Optional name for merged result.

    Returns:
        Merged EvaluationResult.
    """
    if not results:
        raise ValueError("At least one result is required")

    # Use first result as base
    merged = EvaluationResult(
        model_path=results[0].model_path,
        model_name=model_name or results[0].model_name,
        model_family=results[0].model_family,
        parameters=results[0].parameters,
        config=results[0].config,
        timestamp=results[0].timestamp,
    )

    # Merge benchmarks from all results
    for result in results:
        for task, metrics in result.benchmarks.items():
            if task not in merged.benchmarks:
                merged.benchmarks[task] = metrics
            else:
                # Merge metrics, preferring newer values
                merged.benchmarks[task].update(metrics)

    return merged
