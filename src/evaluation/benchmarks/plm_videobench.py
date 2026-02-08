"""PLM-VideoBench utilities for PerceptionLM evaluation.

PLM-VideoBench is a benchmark suite from Meta's PerceptionLM paper that
evaluates fine-grained video understanding capabilities:
- Fine-Grained QA (FGQA): Multiple-choice question answering
- Smart Glasses QA (SGQA): Open-ended video QA
- Region Captioning (RCap): Region-based video captioning
- Region Temporal Localization (RTLoc): Spatio-temporal grounding
- Region Dense Captioning (RDCap): Dense video captioning with regions

PLM-VideoBench is available in lmms-eval as of April 2025.

References:
    - Paper: https://arxiv.org/abs/2504.13180
    - GitHub: https://github.com/facebookresearch/perception_models
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


# PLM-VideoBench task definitions
@dataclass
class PLMTask:
    """Definition of a PLM-VideoBench task."""

    name: str
    lmms_eval_name: str
    description: str
    task_type: str  # "mcq", "openqa", "caption", "grounding"
    metrics: List[str]


# Task definitions
PLM_FGQA = PLMTask(
    name="Fine-Grained QA",
    lmms_eval_name="plm_fgqa",
    description="Fine-grained multiple-choice question answering about video content",
    task_type="mcq",
    metrics=["accuracy"],
)

PLM_SGQA = PLMTask(
    name="Smart Glasses QA",
    lmms_eval_name="plm_sgqa",
    description="Open-ended video question answering from egocentric perspective",
    task_type="openqa",
    metrics=["bleu", "rouge", "cider"],
)

PLM_RCAP = PLMTask(
    name="Region Captioning",
    lmms_eval_name="plm_rcap",
    description="Generate captions for specific regions in video",
    task_type="caption",
    metrics=["bleu", "rouge", "cider"],
)

PLM_RTLOC = PLMTask(
    name="Region Temporal Localization",
    lmms_eval_name="plm_rtloc",
    description="Localize when and where activities occur in video",
    task_type="grounding",
    metrics=["iou", "temporal_accuracy", "spatial_accuracy"],
)

PLM_RDCAP = PLMTask(
    name="Region Dense Captioning",
    lmms_eval_name="plm_rdcap",
    description="Generate dense captions with spatial-temporal grounding",
    task_type="caption",
    metrics=["bleu", "rouge", "cider", "soda_c"],
)

# Task collection
PLM_TASKS: Dict[str, PLMTask] = {
    "fgqa": PLM_FGQA,
    "sgqa": PLM_SGQA,
    "rcap": PLM_RCAP,
    "rtloc": PLM_RTLOC,
    "rdcap": PLM_RDCAP,
}

# Mapping from short names to lmms-eval task names
PLM_TASK_NAMES: Dict[str, str] = {
    task.lmms_eval_name: task.lmms_eval_name for task in PLM_TASKS.values()
}


def get_plm_tasks() -> List[str]:
    """Get all PLM-VideoBench lmms-eval task names.

    Returns:
        List of lmms-eval task names for PLM-VideoBench.
    """
    return [task.lmms_eval_name for task in PLM_TASKS.values()]


def get_plm_task_info(task_name: str) -> Optional[PLMTask]:
    """Get task information for a PLM-VideoBench task.

    Args:
        task_name: Short name (e.g., "fgqa") or lmms-eval name (e.g., "plm_fgqa").

    Returns:
        PLMTask object or None if not found.
    """
    # Check short name
    if task_name in PLM_TASKS:
        return PLM_TASKS[task_name]

    # Check lmms-eval name
    for task in PLM_TASKS.values():
        if task.lmms_eval_name == task_name:
            return task

    return None


@dataclass
class PLMResults:
    """Container for PLM-VideoBench evaluation results."""

    task_results: Dict[str, Dict]
    model_name: str = ""

    def get_accuracy_tasks(self) -> Dict[str, float]:
        """Get accuracy for MCQ tasks.

        Returns:
            Dictionary mapping task name to accuracy.
        """
        results = {}
        for task_name, metrics in self.task_results.items():
            if "accuracy" in metrics:
                results[task_name] = metrics["accuracy"]
        return results

    def get_generation_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get generation metrics (BLEU, ROUGE, CIDEr) for open-ended tasks.

        Returns:
            Dictionary mapping task name to metrics dictionary.
        """
        results = {}
        gen_metrics = ["bleu", "rouge", "cider"]

        for task_name, metrics in self.task_results.items():
            task_gen = {}
            for metric in gen_metrics:
                if metric in metrics:
                    task_gen[metric] = metrics[metric]
            if task_gen:
                results[task_name] = task_gen

        return results

    def get_grounding_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get grounding metrics (IoU, temporal/spatial accuracy).

        Returns:
            Dictionary mapping task name to metrics dictionary.
        """
        results = {}
        grounding_metrics = ["iou", "temporal_accuracy", "spatial_accuracy"]

        for task_name, metrics in self.task_results.items():
            task_grounding = {}
            for metric in grounding_metrics:
                if metric in metrics:
                    task_grounding[metric] = metrics[metric]
            if task_grounding:
                results[task_name] = task_grounding

        return results

    def compute_overall_score(self) -> float:
        """Compute overall PLM-VideoBench score.

        The overall score is computed as the average of:
        - FGQA accuracy
        - Average of generation metrics (BLEU, ROUGE, CIDEr) for SGQA
        - Average of generation metrics for RCap
        - Average of grounding metrics for RTLoc
        - Average of all metrics for RDCap

        Returns:
            Overall score as a float between 0 and 1.
        """
        scores = []

        # FGQA accuracy
        if "plm_fgqa" in self.task_results:
            acc = self.task_results["plm_fgqa"].get("accuracy", 0)
            scores.append(acc)

        # SGQA generation
        if "plm_sgqa" in self.task_results:
            metrics = self.task_results["plm_sgqa"]
            gen_scores = [
                metrics.get("bleu", 0),
                metrics.get("rouge", 0),
                metrics.get("cider", 0) / 10,  # Normalize CIDEr
            ]
            if any(gen_scores):
                scores.append(sum(gen_scores) / len([s for s in gen_scores if s > 0]))

        # RCap generation
        if "plm_rcap" in self.task_results:
            metrics = self.task_results["plm_rcap"]
            gen_scores = [
                metrics.get("bleu", 0),
                metrics.get("rouge", 0),
                metrics.get("cider", 0) / 10,
            ]
            if any(gen_scores):
                scores.append(sum(gen_scores) / len([s for s in gen_scores if s > 0]))

        # RTLoc grounding
        if "plm_rtloc" in self.task_results:
            metrics = self.task_results["plm_rtloc"]
            grounding_scores = [
                metrics.get("iou", 0),
                metrics.get("temporal_accuracy", 0),
            ]
            if any(grounding_scores):
                scores.append(
                    sum(grounding_scores) / len([s for s in grounding_scores if s > 0])
                )

        # RDCap (all metrics)
        if "plm_rdcap" in self.task_results:
            metrics = self.task_results["plm_rdcap"]
            all_scores = list(metrics.values())
            if all_scores:
                # Normalize CIDEr if present
                normalized = []
                for k, v in metrics.items():
                    if k == "cider":
                        normalized.append(v / 10)
                    else:
                        normalized.append(v)
                scores.append(sum(normalized) / len(normalized))

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def to_dict(self) -> Dict:
        """Convert results to dictionary format.

        Returns:
            Dictionary with all results and computed scores.
        """
        return {
            "model": self.model_name,
            "task_results": self.task_results,
            "accuracy_tasks": self.get_accuracy_tasks(),
            "generation_metrics": self.get_generation_metrics(),
            "grounding_metrics": self.get_grounding_metrics(),
            "overall_score": self.compute_overall_score(),
        }


def parse_plm_results(
    raw_results: Dict[str, Dict],
    model_name: str = "",
) -> PLMResults:
    """Parse lmms-eval results into PLMResults container.

    Args:
        raw_results: Raw results from lmms-eval (task_name -> metrics).
        model_name: Name of the evaluated model.

    Returns:
        PLMResults object with parsed results.

    Example:
        >>> raw = {
        ...     "plm_fgqa": {"accuracy": 0.75},
        ...     "plm_sgqa": {"bleu": 0.3, "rouge": 0.4, "cider": 1.2},
        ... }
        >>> results = parse_plm_results(raw, "PerceptionLM-3B")
        >>> results.compute_overall_score()
        0.55
    """
    # Filter to only PLM tasks
    plm_results = {}

    for task_name, metrics in raw_results.items():
        if task_name.startswith("plm_"):
            # Handle nested results structure from lmms-eval
            if isinstance(metrics, dict):
                if "results" in metrics:
                    plm_results[task_name] = metrics["results"]
                else:
                    plm_results[task_name] = metrics
            else:
                plm_results[task_name] = {"value": metrics}

    return PLMResults(task_results=plm_results, model_name=model_name)


def format_plm_results(results: PLMResults) -> str:
    """Format PLM-VideoBench results as a string.

    Args:
        results: PLMResults object.

    Returns:
        Formatted string for display.
    """
    lines = [
        "=" * 60,
        "PLM-VideoBench Results",
        "=" * 60,
        f"Model: {results.model_name or 'Unknown'}",
        "-" * 60,
    ]

    # Accuracy tasks (FGQA)
    accuracy_tasks = results.get_accuracy_tasks()
    if accuracy_tasks:
        lines.append("\nAccuracy-based Tasks:")
        for task_name, acc in accuracy_tasks.items():
            task_info = get_plm_task_info(task_name)
            name = task_info.name if task_info else task_name
            lines.append(f"  {name}: {acc:.2%}")

    # Generation tasks (SGQA, RCap, RDCap)
    gen_metrics = results.get_generation_metrics()
    if gen_metrics:
        lines.append("\nGeneration-based Tasks:")
        for task_name, metrics in gen_metrics.items():
            task_info = get_plm_task_info(task_name)
            name = task_info.name if task_info else task_name
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            lines.append(f"  {name}: {metrics_str}")

    # Grounding tasks (RTLoc)
    grounding_metrics = results.get_grounding_metrics()
    if grounding_metrics:
        lines.append("\nGrounding-based Tasks:")
        for task_name, metrics in grounding_metrics.items():
            task_info = get_plm_task_info(task_name)
            name = task_info.name if task_info else task_name
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            lines.append(f"  {name}: {metrics_str}")

    lines.append("-" * 60)
    lines.append(f"Overall Score: {results.compute_overall_score():.2%}")
    lines.append("=" * 60)

    return "\n".join(lines)
