"""Model wrappers for lmms-eval integration."""

from .model_registry import (
    MODEL_REGISTRY,
    SMOLVLM2_MODELS,
    PERCEPTIONLM_MODELS,
    ModelConfig,
    ModelFamily,
    get_model_config,
    detect_model_family,
    list_available_models,
)

# VLM wrapper is optional (requires lmms-eval)
try:
    from .vlm_wrapper import VLMWrapper, get_vlm_wrapper
    _VLM_AVAILABLE = True
except ImportError:
    VLMWrapper = None
    get_vlm_wrapper = None
    _VLM_AVAILABLE = False

__all__ = [
    "MODEL_REGISTRY",
    "SMOLVLM2_MODELS",
    "PERCEPTIONLM_MODELS",
    "ModelConfig",
    "ModelFamily",
    "get_model_config",
    "detect_model_family",
    "list_available_models",
    "VLMWrapper",
    "get_vlm_wrapper",
]
