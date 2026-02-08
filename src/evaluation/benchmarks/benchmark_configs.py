"""Benchmark configuration and groupings for SmolVLM2 and PerceptionLM evaluation.

This module provides mappings between user-friendly benchmark names and lmms-eval
task names, along with benchmark groupings for convenient evaluation.
"""

from typing import Dict, List, Optional


# Benchmark name mappings (user-friendly -> lmms-eval task name)
BENCHMARK_NAME_MAP: Dict[str, str] = {
    # Video benchmarks (Priority 1 - SmolVLM2)
    "video-mme": "videomme",
    "video_mme": "videomme",
    "videomme": "videomme",
    "mlvu": "mlvu",
    "mvbench": "mvbench",
    "worldsense": "worldsense",
    "tempcompass": "tempcompass",
    "temp-compass": "tempcompass",
    # PLM-VideoBench (Priority 2 - PerceptionLM)
    "plm-videobench": "plm_videobench",
    "plm-fgqa": "plm_fgqa",
    "plm_fgqa": "plm_fgqa",
    "plm-sgqa": "plm_sgqa",
    "plm_sgqa": "plm_sgqa",
    "plm-rcap": "plm_rcap",
    "plm_rcap": "plm_rcap",
    "plm-rtloc": "plm_rtloc",
    "plm_rtloc": "plm_rtloc",
    "plm-rdcap": "plm_rdcap",
    "plm_rdcap": "plm_rdcap",
    # Image/Document benchmarks (Priority 3 - SmolVLM2)
    "textvqa": "textvqa",
    "text-vqa": "textvqa",
    "docvqa": "docvqa",
    "doc-vqa": "docvqa",
    "chartqa": "chartqa",
    "chart-qa": "chartqa",
    "mmmu": "mmmu_val",
    "mathvista": "mathvista_testmini",
    "math-vista": "mathvista_testmini",
    "ocrbench": "ocrbench",
    "ocr-bench": "ocrbench",
    "ai2d": "ai2d",
    "scienceqa": "scienceqa_img",
    "science-qa": "scienceqa_img",
    "mmstar": "mmstar",
}

# Video benchmarks from SmolVLM2 paper
VIDEO_BENCHMARKS: List[str] = [
    "videomme",
    "mlvu",
    "mvbench",
    "worldsense",
    "tempcompass",
]

# PLM-VideoBench from PerceptionLM paper
PLM_VIDEOBENCH: List[str] = [
    "plm_fgqa",      # Fine-Grained Question Answering (MCQ)
    "plm_sgqa",      # Smart Glasses QA (Open-ended)
    "plm_rcap",      # Region Captioning
    "plm_rtloc",     # Region Temporal Localization
    "plm_rdcap",     # Region Dense Video Captioning
]

# Image/Document benchmarks from SmolVLM2 paper
IMAGE_BENCHMARKS: List[str] = [
    "textvqa",
    "docvqa",
    "chartqa",
    "mmmu_val",
    "mathvista_testmini",
    "ocrbench",
    "ai2d",
    "scienceqa_img",
    "mmstar",
]

# Benchmark groups for convenient selection
BENCHMARK_GROUPS: Dict[str, List[str]] = {
    # Video-focused groups
    "video": VIDEO_BENCHMARKS,
    "video-priority": VIDEO_BENCHMARKS,
    "video-smolvlm": VIDEO_BENCHMARKS,
    # PLM groups
    "plm": PLM_VIDEOBENCH,
    "plm-videobench": PLM_VIDEOBENCH,
    "perceptionlm": PLM_VIDEOBENCH,
    # Image groups
    "image": IMAGE_BENCHMARKS,
    "document": ["textvqa", "docvqa", "chartqa", "ocrbench"],
    "reasoning": ["mmmu_val", "mathvista_testmini", "ai2d", "scienceqa_img"],
    # Combined groups
    "all-video": VIDEO_BENCHMARKS + PLM_VIDEOBENCH,
    "all": VIDEO_BENCHMARKS + PLM_VIDEOBENCH + IMAGE_BENCHMARKS,
    "smolvlm": VIDEO_BENCHMARKS + IMAGE_BENCHMARKS,
}

# Benchmark metadata for display and filtering
BENCHMARK_INFO: Dict[str, Dict] = {
    # Video benchmarks
    "videomme": {
        "name": "Video-MME",
        "description": "Comprehensive video understanding (900 videos, 254 hours)",
        "modality": "video",
        "metric": "accuracy",
        "paper": "https://arxiv.org/abs/2405.21075",
    },
    "mlvu": {
        "name": "MLVU",
        "description": "Multi-task Long Video Understanding",
        "modality": "video",
        "metric": "accuracy",
    },
    "mvbench": {
        "name": "MVBench",
        "description": "Multi-modal Video Benchmark for temporal reasoning",
        "modality": "video",
        "metric": "accuracy",
    },
    "worldsense": {
        "name": "WorldSense",
        "description": "World knowledge and commonsense in videos",
        "modality": "video",
        "metric": "accuracy",
    },
    "tempcompass": {
        "name": "TempCompass",
        "description": "Temporal reasoning and understanding in videos",
        "modality": "video",
        "metric": "accuracy",
    },
    # PLM-VideoBench
    "plm_fgqa": {
        "name": "PLM Fine-Grained QA",
        "description": "Fine-grained multiple-choice question answering",
        "modality": "video",
        "metric": "accuracy",
        "paper": "https://arxiv.org/abs/2504.13180",
    },
    "plm_sgqa": {
        "name": "PLM Smart Glasses QA",
        "description": "Open-ended video question answering",
        "modality": "video",
        "metric": ["bleu", "rouge", "cider"],
    },
    "plm_rcap": {
        "name": "PLM Region Captioning",
        "description": "Region-based video captioning",
        "modality": "video",
        "metric": ["bleu", "rouge", "cider"],
    },
    "plm_rtloc": {
        "name": "PLM Region Temporal Localization",
        "description": "Spatio-temporal grounding in videos",
        "modality": "video",
        "metric": ["iou", "temporal_accuracy"],
    },
    "plm_rdcap": {
        "name": "PLM Region Dense Captioning",
        "description": "Dense video captioning with region grounding",
        "modality": "video",
        "metric": ["bleu", "rouge", "cider", "dense_accuracy"],
    },
    # Image/Document benchmarks
    "textvqa": {
        "name": "TextVQA",
        "description": "Visual question answering about text in images",
        "modality": "image",
        "metric": "accuracy",
    },
    "docvqa": {
        "name": "DocVQA",
        "description": "Document visual question answering",
        "modality": "image",
        "metric": "anls",
    },
    "chartqa": {
        "name": "ChartQA",
        "description": "Chart understanding and question answering",
        "modality": "image",
        "metric": "accuracy",
    },
    "mmmu_val": {
        "name": "MMMU",
        "description": "Massive Multi-discipline Multimodal Understanding",
        "modality": "image",
        "metric": "accuracy",
    },
    "mathvista_testmini": {
        "name": "MathVista",
        "description": "Mathematical reasoning in visual contexts",
        "modality": "image",
        "metric": "accuracy",
    },
    "ocrbench": {
        "name": "OCRBench",
        "description": "OCR and text extraction evaluation",
        "modality": "image",
        "metric": "accuracy",
    },
    "ai2d": {
        "name": "AI2D",
        "description": "Diagram understanding and reasoning",
        "modality": "image",
        "metric": "accuracy",
    },
    "scienceqa_img": {
        "name": "ScienceQA",
        "description": "Science question answering with images",
        "modality": "image",
        "metric": "accuracy",
    },
    "mmstar": {
        "name": "MMStar",
        "description": "Multi-modal reasoning benchmark",
        "modality": "image",
        "metric": "accuracy",
    },
}


def resolve_benchmark_names(benchmark_str: str) -> List[str]:
    """Resolve user benchmark string to lmms-eval task names.

    Handles comma-separated lists of benchmark names or group names.
    Names are normalized (lowercased, trimmed) before lookup.

    Args:
        benchmark_str: Comma-separated benchmarks or group names.
            Examples: "video", "video-mme,mlvu", "all", "plm-videobench"

    Returns:
        Sorted list of unique lmms-eval task names.

    Examples:
        >>> resolve_benchmark_names("video")
        ['mlvu', 'mvbench', 'tempcompass', 'videomme', 'worldsense']

        >>> resolve_benchmark_names("video-mme,mlvu")
        ['mlvu', 'videomme']

        >>> resolve_benchmark_names("plm")
        ['plm_fgqa', 'plm_rcap', 'plm_rdcap', 'plm_rtloc', 'plm_sgqa']
    """
    requested = [b.strip().lower() for b in benchmark_str.split(",")]
    resolved: set[str] = set()

    for name in requested:
        if not name:
            continue

        # Check if it's a group name
        if name in BENCHMARK_GROUPS:
            resolved.update(BENCHMARK_GROUPS[name])
        # Check if it's a mapped benchmark name
        elif name in BENCHMARK_NAME_MAP:
            resolved.add(BENCHMARK_NAME_MAP[name])
        # Try as-is (may be a direct lmms-eval task name)
        else:
            resolved.add(name)

    return sorted(list(resolved))


def get_benchmark_info(task_name: str) -> Dict:
    """Get metadata for a benchmark task.

    Args:
        task_name: The lmms-eval task name.

    Returns:
        Dictionary with benchmark metadata (name, description, modality, metric).
        Returns minimal info if task is unknown.
    """
    return BENCHMARK_INFO.get(
        task_name,
        {
            "name": task_name,
            "description": "Unknown benchmark",
            "modality": "unknown",
            "metric": "unknown",
        },
    )


def get_benchmarks_by_modality(modality: str) -> List[str]:
    """Get all benchmark task names for a given modality.

    Args:
        modality: One of "video", "image", or "all".

    Returns:
        List of lmms-eval task names for the specified modality.
    """
    if modality == "all":
        return VIDEO_BENCHMARKS + PLM_VIDEOBENCH + IMAGE_BENCHMARKS
    elif modality == "video":
        return VIDEO_BENCHMARKS + PLM_VIDEOBENCH
    elif modality == "image":
        return IMAGE_BENCHMARKS
    else:
        return []


def list_available_benchmarks() -> str:
    """Generate a formatted string listing all available benchmarks.

    Returns:
        Formatted string with benchmark groups and individual benchmarks.
    """
    lines = ["Available Benchmarks:", ""]

    lines.append("Groups:")
    for group_name, benchmarks in sorted(BENCHMARK_GROUPS.items()):
        lines.append(f"  {group_name}: {', '.join(benchmarks[:3])}...")

    lines.append("")
    lines.append("Video Benchmarks:")
    for task in VIDEO_BENCHMARKS:
        info = BENCHMARK_INFO.get(task, {})
        lines.append(f"  {task}: {info.get('name', task)}")

    lines.append("")
    lines.append("PLM-VideoBench:")
    for task in PLM_VIDEOBENCH:
        info = BENCHMARK_INFO.get(task, {})
        lines.append(f"  {task}: {info.get('name', task)}")

    lines.append("")
    lines.append("Image/Document Benchmarks:")
    for task in IMAGE_BENCHMARKS:
        info = BENCHMARK_INFO.get(task, {})
        lines.append(f"  {task}: {info.get('name', task)}")

    return "\n".join(lines)
