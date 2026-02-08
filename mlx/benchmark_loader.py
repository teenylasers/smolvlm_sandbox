"""Benchmark data loading for local MLX evaluation.

This module loads video benchmark datasets from HuggingFace with support for
random sampling to enable quick local evaluation on Apple Silicon.

Supported benchmarks:
- Video-MME: Comprehensive video understanding
- MVBench: Multi-modal video benchmark
- MLVU: Multi-task long video understanding
- TempCompass: Temporal reasoning

Usage:
    from mlx.benchmark_loader import load_benchmark_samples

    samples = load_benchmark_samples(
        benchmark="video-mme",
        num_samples=100,
        seed=42,
    )
"""

import hashlib
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.request import urlretrieve

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("datasets library required. Install with: pip install datasets")


@dataclass
class BenchmarkSample:
    """A single benchmark sample for evaluation."""

    sample_id: str
    video_path: str  # Local path to downloaded video
    question: str
    options: List[str]  # For MCQ: ["A. ...", "B. ...", "C. ...", "D. ..."]
    correct_answer: str  # The correct option letter (A, B, C, D)
    metadata: Dict = None

    def get_prompt(self) -> str:
        """Format the question and options as a prompt."""
        options_text = "\n".join(self.options)
        return f"{self.question}\n\n{options_text}\n\nAnswer with the letter of the correct option."


# Benchmark configurations
BENCHMARK_CONFIGS = {
    "video-mme": {
        "hf_path": "lmms-lab/Video-MME",
        "split": "test",
        "video_field": "video",
        "question_field": "question",
        "options_field": "options",
        "answer_field": "answer",
    },
    "mvbench": {
        "hf_path": "OpenGVLab/MVBench",
        "split": "test",
        "video_field": "video",
        "question_field": "question",
        "options_field": "candidates",
        "answer_field": "answer",
    },
    "mlvu": {
        "hf_path": "MLVU/MLVU",
        "split": "test",
        "video_field": "video",
        "question_field": "question",
        "options_field": "options",
        "answer_field": "answer",
    },
    "tempcompass": {
        "hf_path": "lmms-lab/TempCompass",
        "split": "test",
        "video_field": "video",
        "question_field": "question",
        "options_field": "options",
        "answer_field": "answer",
    },
}

# Benchmark group aliases
BENCHMARK_GROUPS = {
    "video": ["video-mme", "mvbench", "mlvu", "tempcompass"],
    "all": ["video-mme", "mvbench", "mlvu", "tempcompass"],
}


def resolve_benchmarks(benchmark_str: str) -> List[str]:
    """Resolve benchmark string to list of benchmark names.

    Args:
        benchmark_str: Comma-separated benchmarks or group name.

    Returns:
        List of benchmark names.
    """
    benchmarks = []
    for name in benchmark_str.split(","):
        name = name.strip().lower()
        if name in BENCHMARK_GROUPS:
            benchmarks.extend(BENCHMARK_GROUPS[name])
        elif name in BENCHMARK_CONFIGS:
            benchmarks.append(name)
        else:
            raise ValueError(f"Unknown benchmark: {name}")
    return list(set(benchmarks))


def _get_cache_dir() -> Path:
    """Get the cache directory for downloaded videos."""
    cache_dir = Path.home() / ".cache" / "smolvlm_eval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_video(video_url: str, cache_dir: Path) -> str:
    """Download a video and cache it locally.

    Args:
        video_url: URL to the video.
        cache_dir: Directory to cache videos.

    Returns:
        Local path to the downloaded video.
    """
    # Create a hash-based filename for caching
    url_hash = hashlib.md5(video_url.encode()).hexdigest()[:16]
    extension = Path(video_url).suffix or ".mp4"
    local_path = cache_dir / f"{url_hash}{extension}"

    if not local_path.exists():
        print(f"  Downloading: {video_url[:80]}...")
        try:
            urlretrieve(video_url, local_path)
        except Exception as e:
            print(f"  Failed to download: {e}")
            return None

    return str(local_path)


def _parse_options(options_data, benchmark: str) -> Tuple[List[str], str]:
    """Parse options from benchmark-specific format.

    Args:
        options_data: Raw options data from dataset.
        benchmark: Benchmark name.

    Returns:
        Tuple of (formatted_options, correct_answer_letter).
    """
    if isinstance(options_data, list):
        # Most benchmarks: list of option texts
        formatted = []
        for i, opt in enumerate(options_data):
            letter = chr(ord("A") + i)
            formatted.append(f"{letter}. {opt}")
        return formatted, None  # Answer parsed separately
    elif isinstance(options_data, dict):
        # Some benchmarks use dict format
        formatted = []
        for letter in ["A", "B", "C", "D"]:
            if letter in options_data:
                formatted.append(f"{letter}. {options_data[letter]}")
        return formatted, None
    else:
        # Fallback
        return [str(options_data)], None


def _parse_answer(answer_data) -> str:
    """Parse the correct answer to a single letter.

    Args:
        answer_data: Raw answer data from dataset.

    Returns:
        Single letter (A, B, C, D).
    """
    if isinstance(answer_data, str):
        # Could be just the letter or full text
        answer_data = answer_data.strip().upper()
        if len(answer_data) == 1 and answer_data in "ABCD":
            return answer_data
        # Try to extract first letter
        if answer_data and answer_data[0] in "ABCD":
            return answer_data[0]
    elif isinstance(answer_data, int):
        # Index-based answer
        return chr(ord("A") + answer_data)

    # Default fallback
    return "A"


class BenchmarkLoader:
    """Loads and manages benchmark data for evaluation."""

    def __init__(
        self,
        benchmark: str,
        num_samples: int = 100,
        seed: int = 42,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize benchmark loader.

        Args:
            benchmark: Benchmark name (e.g., "video-mme").
            num_samples: Number of samples to load.
            seed: Random seed for reproducible sampling.
            cache_dir: Directory for caching videos.
        """
        if benchmark not in BENCHMARK_CONFIGS:
            raise ValueError(
                f"Unknown benchmark: {benchmark}. "
                f"Available: {list(BENCHMARK_CONFIGS.keys())}"
            )

        self.benchmark = benchmark
        self.config = BENCHMARK_CONFIGS[benchmark]
        self.num_samples = num_samples
        self.seed = seed
        self.cache_dir = cache_dir or _get_cache_dir()
        self._dataset = None
        self._samples = None

    def _load_dataset(self):
        """Load the dataset from HuggingFace."""
        if self._dataset is not None:
            return

        print(f"Loading {self.benchmark} dataset from {self.config['hf_path']}...")
        try:
            self._dataset = load_dataset(
                self.config["hf_path"],
                split=self.config["split"],
                streaming=True,  # Stream to avoid loading full dataset
            )
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            print("Attempting alternative loading...")
            # Try without streaming
            self._dataset = load_dataset(
                self.config["hf_path"],
                split=self.config["split"],
            )

    def _sample_indices(self, total_size: int) -> List[int]:
        """Get random sample indices.

        Args:
            total_size: Total number of items in dataset.

        Returns:
            List of sampled indices.
        """
        random.seed(self.seed)
        n = min(self.num_samples, total_size)
        return random.sample(range(total_size), n)

    def load_samples(self) -> List[BenchmarkSample]:
        """Load and return benchmark samples.

        Returns:
            List of BenchmarkSample objects.
        """
        if self._samples is not None:
            return self._samples

        self._load_dataset()

        samples = []
        seen_indices = set()
        target_count = self.num_samples

        # For streaming datasets, we need to iterate and sample
        random.seed(self.seed)

        print(f"Sampling {target_count} items from {self.benchmark}...")

        # Collect samples from the dataset
        all_items = []
        try:
            # Try to iterate (works for both streaming and regular datasets)
            for i, item in enumerate(self._dataset):
                all_items.append((i, item))
                # Stop early if we have enough for sampling
                if len(all_items) >= target_count * 10:
                    break
        except Exception:
            # Fallback for non-iterable datasets
            all_items = [(i, self._dataset[i]) for i in range(len(self._dataset))]

        # Sample from collected items
        if len(all_items) > target_count:
            sampled_items = random.sample(all_items, target_count)
        else:
            sampled_items = all_items

        # Process sampled items
        for idx, item in sampled_items:
            try:
                sample = self._process_item(idx, item)
                if sample is not None:
                    samples.append(sample)
            except Exception as e:
                print(f"  Error processing item {idx}: {e}")
                continue

        self._samples = samples
        print(f"Loaded {len(samples)} samples from {self.benchmark}")
        return samples

    def _process_item(self, idx: int, item: Dict) -> Optional[BenchmarkSample]:
        """Process a single dataset item into a BenchmarkSample.

        Args:
            idx: Item index.
            item: Raw dataset item.

        Returns:
            BenchmarkSample or None if processing fails.
        """
        # Get video
        video_data = item.get(self.config["video_field"])
        if video_data is None:
            return None

        # Handle different video formats
        if isinstance(video_data, str):
            # URL or path
            if video_data.startswith(("http://", "https://")):
                video_path = _download_video(video_data, self.cache_dir)
            else:
                video_path = video_data
        elif isinstance(video_data, dict):
            # May have 'path' or 'bytes' key
            if "path" in video_data:
                video_path = video_data["path"]
            elif "bytes" in video_data:
                # Save bytes to cache
                video_hash = hashlib.md5(str(idx).encode()).hexdigest()[:16]
                video_path = self.cache_dir / f"{self.benchmark}_{video_hash}.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_data["bytes"])
                video_path = str(video_path)
            else:
                return None
        else:
            return None

        if video_path is None:
            return None

        # Get question
        question = item.get(self.config["question_field"], "")
        if not question:
            return None

        # Get options
        options_data = item.get(self.config["options_field"], [])
        options, _ = _parse_options(options_data, self.benchmark)

        # Get answer
        answer_data = item.get(self.config["answer_field"])
        correct_answer = _parse_answer(answer_data)

        return BenchmarkSample(
            sample_id=f"{self.benchmark}_{idx}",
            video_path=video_path,
            question=question,
            options=options,
            correct_answer=correct_answer,
            metadata={"benchmark": self.benchmark, "index": idx},
        )

    def __len__(self) -> int:
        """Return number of loaded samples."""
        if self._samples is None:
            self.load_samples()
        return len(self._samples)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        """Iterate over samples."""
        if self._samples is None:
            self.load_samples()
        return iter(self._samples)


def load_benchmark_samples(
    benchmark: str,
    num_samples: int = 100,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> List[BenchmarkSample]:
    """Load samples from a benchmark.

    Args:
        benchmark: Benchmark name.
        num_samples: Number of samples to load.
        seed: Random seed for reproducibility.
        cache_dir: Directory for caching videos.

    Returns:
        List of BenchmarkSample objects.
    """
    loader = BenchmarkLoader(
        benchmark=benchmark,
        num_samples=num_samples,
        seed=seed,
        cache_dir=cache_dir,
    )
    return loader.load_samples()


def list_benchmarks() -> str:
    """List available benchmarks.

    Returns:
        Formatted string listing benchmarks.
    """
    lines = ["Available Benchmarks:", ""]
    for name, config in BENCHMARK_CONFIGS.items():
        lines.append(f"  {name}: {config['hf_path']}")
    lines.append("")
    lines.append("Groups:")
    for group, benchmarks in BENCHMARK_GROUPS.items():
        lines.append(f"  {group}: {', '.join(benchmarks)}")
    return "\n".join(lines)
