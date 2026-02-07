"""SmolVLM2 Model Configuration.

Defines configuration classes for SmolVLM2 256M and 500M variants.
Architecture follows Idefics3 with modifications for efficiency.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SmolVLMConfig:
    """Base configuration for SmolVLM2 models.

    Architecture:
    - Vision Encoder: SigLIP base-patch16-512 (93M params)
    - Text Decoder: SmolLM2 (135M or 360M)
    - Connector: 3x3 Pixel Shuffle + MLP
    """

    # Model identifiers
    model_name: str = "smolvlm2"
    model_size: str = "256m"  # "256m" or "500m"

    # Vision encoder (SigLIP)
    vision_encoder_name: str = "google/siglip-base-patch16-512"
    image_size: int = 384  # Divisible by 3 for pixel shuffle
    patch_size: int = 16
    vision_hidden_size: int = 768  # SigLIP base hidden dim

    # Pixel shuffle connector
    pixel_shuffle_ratio: int = 3  # 3x3 = 9x compression
    connector_hidden_size: int = 768
    connector_num_layers: int = 2

    # Text decoder (SmolLM2)
    text_decoder_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    text_hidden_size: int = 576  # SmolLM2-135M hidden dim
    text_vocab_size: int = 49152

    # Extended context for VLM
    max_position_embeddings: int = 16384
    rope_theta: float = 273000.0  # Extended from 10000 for 16k context

    # Visual tokens
    num_visual_tokens_per_patch: int = 81  # (384/16/3)^2 after pixel shuffle
    max_image_patches: int = 36  # Max sub-images for large images

    # Training settings
    freeze_vision_encoder: bool = True
    unfreeze_vision_after_steps: int = 10000

    # Precision
    torch_dtype: str = "bfloat16"

    # Special tokens
    image_token: str = "<image>"
    video_token: str = "<video>"
    image_start_token: str = "<image_start>"
    image_end_token: str = "<image_end>"

    def __post_init__(self):
        """Validate configuration."""
        assert self.image_size % self.pixel_shuffle_ratio == 0, \
            f"image_size ({self.image_size}) must be divisible by pixel_shuffle_ratio ({self.pixel_shuffle_ratio})"
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"


@dataclass
class SmolVLM256MConfig(SmolVLMConfig):
    """Configuration for SmolVLM2-256M.

    Uses SmolLM2-135M as the text decoder.
    Total params: ~256M (93M vision + 135M text + ~28M connector)
    """
    model_size: str = "256m"
    text_decoder_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    text_hidden_size: int = 576  # SmolLM2-135M

    # Smaller connector for 256M
    connector_hidden_size: int = 576
    connector_num_layers: int = 2


@dataclass
class SmolVLM500MConfig(SmolVLMConfig):
    """Configuration for SmolVLM2-500M.

    Uses SmolLM2-360M as the text decoder.
    Total params: ~500M (93M vision + 360M text + ~47M connector)
    """
    model_size: str = "500m"
    text_decoder_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    text_hidden_size: int = 960  # SmolLM2-360M

    # Larger connector for 500M
    connector_hidden_size: int = 960
    connector_num_layers: int = 2


def get_config(model_size: str = "256m") -> SmolVLMConfig:
    """Get configuration by model size.

    Args:
        model_size: Either "256m" or "500m"

    Returns:
        SmolVLMConfig instance for the specified size
    """
    configs = {
        "256m": SmolVLM256MConfig,
        "500m": SmolVLM500MConfig,
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return configs[model_size]()


# Model card info for reference
MODEL_CARDS = {
    "256m": {
        "hf_repo": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "params": "256M",
        "gpu_ram_inference": "1.38GB",
        "gpu_ram_training": "~4GB (LoRA)",
    },
    "500m": {
        "hf_repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "params": "500M",
        "gpu_ram_inference": "1.8GB",
        "gpu_ram_training": "~6GB (LoRA)",
    },
}
