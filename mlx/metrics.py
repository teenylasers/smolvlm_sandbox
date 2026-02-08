"""Evaluation metrics for local MLX evaluation.

This module provides metrics computation for MCQ-style video benchmarks.

Usage:
    from mlx.metrics import extract_answer, compute_accuracy, EvaluationResults

    # Extract answer from model response
    answer = extract_answer("The answer is B because...")  # Returns "B"

    # Compute accuracy
    predictions = ["A", "B", "C", "D", "A"]
    labels = ["A", "B", "A", "D", "A"]
    accuracy = compute_accuracy(predictions, labels)  # 0.8
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_answer(response: str) -> str:
    """Extract the answer letter from a model response.

    Attempts multiple strategies to find the answer:
    1. Direct single letter
    2. "Answer: X" or "answer is X" patterns
    3. First capital letter A-D
    4. Letter followed by period or parenthesis

    Args:
        response: Model's generated response.

    Returns:
        Single letter (A, B, C, D) or empty string if not found.
    """
    if not response:
        return ""

    response = response.strip()

    # Strategy 1: Response is just a single letter
    if len(response) == 1 and response.upper() in "ABCD":
        return response.upper()

    # Strategy 2: Common answer patterns
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Da-d])",
        r"^([A-Da-d])[\.\)\:]",  # Starts with "A." or "A)" or "A:"
        r"\b([A-Da-d])\s*(?:is correct|is the (?:correct|right|best))",
        r"(?:correct|right|best)\s*(?:answer|option|choice)\s*(?:is|:)?\s*([A-Da-d])",
        r"^([A-Da-d])\b",  # Starts with just the letter
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Strategy 3: Find first standalone A-D
    match = re.search(r"\b([A-Da-d])\b", response)
    if match:
        return match.group(1).upper()

    # Strategy 4: First letter if it's A-D
    first_char = response[0].upper()
    if first_char in "ABCD":
        return first_char

    return ""


def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
    """Compute accuracy from predictions and labels.

    Args:
        predictions: List of predicted answers.
        labels: List of correct answers.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    if not predictions or not labels:
        return 0.0

    if len(predictions) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions, {len(labels)} labels"
        )

    correct = sum(
        1 for pred, label in zip(predictions, labels) if pred.upper() == label.upper()
    )
    return correct / len(predictions)


@dataclass
class SampleResult:
    """Result for a single evaluation sample."""

    sample_id: str
    prediction: str
    label: str
    correct: bool
    response: str  # Full model response
    question: str
    video_path: str


@dataclass
class BenchmarkResult:
    """Results for a single benchmark."""

    benchmark: str
    accuracy: float
    num_samples: int
    num_correct: int
    samples: List[SampleResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "benchmark": self.benchmark,
            "accuracy": self.accuracy,
            "num_samples": self.num_samples,
            "num_correct": self.num_correct,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "prediction": s.prediction,
                    "label": s.label,
                    "correct": s.correct,
                }
                for s in self.samples
            ],
        }


@dataclass
class EvaluationResults:
    """Complete evaluation results across all benchmarks."""

    model_size: str
    model_id: str
    benchmarks: Dict[str, BenchmarkResult] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)

    def add_benchmark(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.benchmarks[result.benchmark] = result

    @property
    def overall_accuracy(self) -> float:
        """Compute overall accuracy across all benchmarks."""
        if not self.benchmarks:
            return 0.0

        total_correct = sum(b.num_correct for b in self.benchmarks.values())
        total_samples = sum(b.num_samples for b in self.benchmarks.values())

        return total_correct / total_samples if total_samples > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model": {
                "size": self.model_size,
                "id": self.model_id,
            },
            "timestamp": self.timestamp,
            "config": self.config,
            "overall_accuracy": self.overall_accuracy,
            "benchmarks": {
                name: result.to_dict() for name, result in self.benchmarks.items()
            },
        }

    def save(self, path: Path):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Results saved to: {path}")

    @classmethod
    def load(cls, path: Path) -> "EvaluationResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)

        results = cls(
            model_size=data["model"]["size"],
            model_id=data["model"]["id"],
            timestamp=data.get("timestamp", ""),
            config=data.get("config", {}),
        )

        for name, bench_data in data.get("benchmarks", {}).items():
            samples = [
                SampleResult(
                    sample_id=s["sample_id"],
                    prediction=s["prediction"],
                    label=s["label"],
                    correct=s["correct"],
                    response="",
                    question="",
                    video_path="",
                )
                for s in bench_data.get("samples", [])
            ]

            results.benchmarks[name] = BenchmarkResult(
                benchmark=bench_data["benchmark"],
                accuracy=bench_data["accuracy"],
                num_samples=bench_data["num_samples"],
                num_correct=bench_data["num_correct"],
                samples=samples,
            )

        return results

    def print_summary(self):
        """Print a formatted summary of results."""
        print("\n" + "=" * 60)
        print(f"Evaluation Results: {self.model_size}")
        print("=" * 60)
        print(f"Model: {self.model_id}")
        print(f"Timestamp: {self.timestamp}")
        print("-" * 60)

        for name, result in sorted(self.benchmarks.items()):
            print(
                f"  {name:20} {result.accuracy:6.2%} "
                f"({result.num_correct}/{result.num_samples})"
            )

        print("-" * 60)
        print(f"  {'Overall':20} {self.overall_accuracy:6.2%}")
        print("=" * 60)


def evaluate_benchmark(
    benchmark_name: str,
    samples: List["BenchmarkSample"],
    responses: List[str],
) -> BenchmarkResult:
    """Evaluate responses against a benchmark.

    Args:
        benchmark_name: Name of the benchmark.
        samples: List of BenchmarkSample objects.
        responses: List of model responses.

    Returns:
        BenchmarkResult with computed metrics.
    """
    if len(samples) != len(responses):
        raise ValueError(
            f"Length mismatch: {len(samples)} samples, {len(responses)} responses"
        )

    sample_results = []
    num_correct = 0

    for sample, response in zip(samples, responses):
        prediction = extract_answer(response)
        correct = prediction.upper() == sample.correct_answer.upper()

        if correct:
            num_correct += 1

        sample_results.append(
            SampleResult(
                sample_id=sample.sample_id,
                prediction=prediction,
                label=sample.correct_answer,
                correct=correct,
                response=response,
                question=sample.question,
                video_path=sample.video_path,
            )
        )

    accuracy = num_correct / len(samples) if samples else 0.0

    return BenchmarkResult(
        benchmark=benchmark_name,
        accuracy=accuracy,
        num_samples=len(samples),
        num_correct=num_correct,
        samples=sample_results,
    )
