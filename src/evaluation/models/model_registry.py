"""Model registry for SmolVLM2 and PerceptionLM models.

This module provides model configurations and utilities for detecting
and loading different vision-language models for evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union


class ModelFamily(Enum):
    """Supported model families."""

    SMOLVLM2 = "smolvlm2"
    PERCEPTIONLM = "perceptionlm"
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    """Configuration for a vision-language model."""

    # Model identification
    name: str
    family: ModelFamily
    hf_path: str
    parameters: str  # e.g., "256M", "1B", "2.2B"

    # Model settings
    dtype: str = "bfloat16"
    max_frames: int = 32
    max_image_patches: int = 36
    attn_implementation: Optional[str] = None  # auto-detect

    # Processor settings
    use_fast_processor: bool = True

    # Video settings
    video_load_backend: str = "decord"  # decord, torchvision, av

    # Additional kwargs for model loading
    model_kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Auto-detect attention implementation based on availability
        if self.attn_implementation is None:
            try:
                import flash_attn  # noqa: F401

                self.attn_implementation = "flash_attention_2"
            except ImportError:
                self.attn_implementation = "eager"


# SmolVLM2 model configurations
SMOLVLM2_256M = ModelConfig(
    name="SmolVLM2-256M",
    family=ModelFamily.SMOLVLM2,
    hf_path="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    parameters="256M",
    max_frames=32,
)

SMOLVLM2_500M = ModelConfig(
    name="SmolVLM2-500M",
    family=ModelFamily.SMOLVLM2,
    hf_path="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    parameters="500M",
    max_frames=32,
)

SMOLVLM2_2_2B = ModelConfig(
    name="SmolVLM2-2.2B",
    family=ModelFamily.SMOLVLM2,
    hf_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    parameters="2.2B",
    max_frames=32,
)

# PerceptionLM model configurations
PERCEPTIONLM_1B = ModelConfig(
    name="PerceptionLM-1B",
    family=ModelFamily.PERCEPTIONLM,
    hf_path="facebook/Perception-LM-1B",
    parameters="1B",
    max_frames=32,
    video_load_backend="decord",
)

PERCEPTIONLM_3B = ModelConfig(
    name="PerceptionLM-3B",
    family=ModelFamily.PERCEPTIONLM,
    hf_path="facebook/Perception-LM-3B",
    parameters="3B",
    max_frames=32,
    video_load_backend="decord",
)

PERCEPTIONLM_8B = ModelConfig(
    name="PerceptionLM-8B",
    family=ModelFamily.PERCEPTIONLM,
    hf_path="facebook/Perception-LM-8B",
    parameters="8B",
    max_frames=32,
    video_load_backend="decord",
)

# Model collections
SMOLVLM2_MODELS: Dict[str, ModelConfig] = {
    "256m": SMOLVLM2_256M,
    "500m": SMOLVLM2_500M,
    "2.2b": SMOLVLM2_2_2B,
    "2b": SMOLVLM2_2_2B,  # alias
}

PERCEPTIONLM_MODELS: Dict[str, ModelConfig] = {
    "1b": PERCEPTIONLM_1B,
    "3b": PERCEPTIONLM_3B,
    "8b": PERCEPTIONLM_8B,
}

# Combined registry indexed by HuggingFace path
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # SmolVLM2
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct": SMOLVLM2_256M,
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct": SMOLVLM2_500M,
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": SMOLVLM2_2_2B,
    # PerceptionLM
    "facebook/Perception-LM-1B": PERCEPTIONLM_1B,
    "facebook/Perception-LM-3B": PERCEPTIONLM_3B,
    "facebook/Perception-LM-8B": PERCEPTIONLM_8B,
}

# Path patterns for model family detection
_SMOLVLM2_PATTERNS = ["smolvlm", "SmolVLM"]
_PERCEPTIONLM_PATTERNS = ["perception", "Perception", "PLM", "plm"]


def detect_model_family(model_path: str) -> ModelFamily:
    """Detect the model family from a model path or HuggingFace ID.

    Args:
        model_path: Path to local checkpoint or HuggingFace model ID.

    Returns:
        The detected ModelFamily enum value.

    Examples:
        >>> detect_model_family("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        ModelFamily.SMOLVLM2

        >>> detect_model_family("facebook/Perception-LM-3B")
        ModelFamily.PERCEPTIONLM

        >>> detect_model_family("./checkpoints/my_model")
        ModelFamily.UNKNOWN
    """
    path_str = str(model_path).lower()

    # Check for SmolVLM2 patterns
    for pattern in _SMOLVLM2_PATTERNS:
        if pattern.lower() in path_str:
            return ModelFamily.SMOLVLM2

    # Check for PerceptionLM patterns
    for pattern in _PERCEPTIONLM_PATTERNS:
        if pattern.lower() in path_str:
            return ModelFamily.PERCEPTIONLM

    return ModelFamily.UNKNOWN


def get_model_config(
    model_path: str,
    model_size: Optional[str] = None,
) -> ModelConfig:
    """Get model configuration for a given model path.

    First checks if the path is a known HuggingFace model ID. If not,
    attempts to detect the model family and create an appropriate config.

    Args:
        model_path: Path to local checkpoint or HuggingFace model ID.
        model_size: Optional model size hint (e.g., "256m", "3b") for
            local checkpoints where size cannot be auto-detected.

    Returns:
        ModelConfig for the specified model.

    Examples:
        >>> config = get_model_config("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        >>> config.name
        'SmolVLM2-2.2B'

        >>> config = get_model_config("./checkpoints/vision_stage", model_size="256m")
        >>> config.family
        ModelFamily.SMOLVLM2
    """
    # Check if it's a known HuggingFace model
    if model_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_path]

    # Detect model family
    family = detect_model_family(model_path)

    # Try to infer size from path if not provided
    if model_size is None:
        model_size = _infer_model_size(model_path)

    # Create config based on family and size
    if family == ModelFamily.SMOLVLM2:
        base_config = SMOLVLM2_MODELS.get(
            model_size.lower() if model_size else "256m",
            SMOLVLM2_256M,
        )
    elif family == ModelFamily.PERCEPTIONLM:
        base_config = PERCEPTIONLM_MODELS.get(
            model_size.lower() if model_size else "1b",
            PERCEPTIONLM_1B,
        )
    else:
        # Default to SmolVLM2-like config for unknown models
        base_config = SMOLVLM2_256M

    # Return config with updated path
    return ModelConfig(
        name=f"Custom-{Path(model_path).name}",
        family=family,
        hf_path=model_path,
        parameters=model_size or "unknown",
        dtype=base_config.dtype,
        max_frames=base_config.max_frames,
        max_image_patches=base_config.max_image_patches,
        attn_implementation=base_config.attn_implementation,
        use_fast_processor=base_config.use_fast_processor,
        video_load_backend=base_config.video_load_backend,
    )


def _infer_model_size(model_path: str) -> Optional[str]:
    """Attempt to infer model size from path.

    Args:
        model_path: Model path or ID.

    Returns:
        Inferred size string (e.g., "256m", "2.2b") or None.
    """
    path_lower = model_path.lower()

    # Check for common size patterns
    size_patterns = [
        ("256m", "256m"),
        ("500m", "500m"),
        ("2.2b", "2.2b"),
        ("2b", "2.2b"),
        ("1b", "1b"),
        ("3b", "3b"),
        ("8b", "8b"),
    ]

    for pattern, size in size_patterns:
        if pattern in path_lower:
            return size

    return None


def list_available_models() -> str:
    """Generate a formatted string listing all available models.

    Returns:
        Formatted string with model information.
    """
    lines = ["Available Models:", ""]

    lines.append("SmolVLM2 (HuggingFace):")
    for size, config in SMOLVLM2_MODELS.items():
        if size in ["256m", "500m", "2.2b"]:  # Skip aliases
            lines.append(f"  {config.name}: {config.hf_path}")

    lines.append("")
    lines.append("PerceptionLM (Meta/Facebook):")
    for size, config in PERCEPTIONLM_MODELS.items():
        lines.append(f"  {config.name}: {config.hf_path}")

    return "\n".join(lines)


def get_all_model_paths() -> List[str]:
    """Get all known HuggingFace model paths.

    Returns:
        List of HuggingFace model IDs.
    """
    return list(MODEL_REGISTRY.keys())
