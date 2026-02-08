"""Unified vision-language model wrapper for lmms-eval.

This module provides a single lmms-eval compatible wrapper that supports
both SmolVLM2 and PerceptionLM model families. Both families use the same
transformers interface (AutoModelForImageTextToText + AutoProcessor),
enabling unified handling.

Usage with lmms-eval:
    python -m lmms_eval \
        --model vlm \
        --model_args pretrained=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
        --tasks videomme \
        --batch_size 1
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

from .model_registry import ModelConfig, ModelFamily, get_model_config

logger = logging.getLogger(__name__)

# Try to import lmms_eval components
try:
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import register_model

    LMMS_EVAL_AVAILABLE = True
except ImportError:
    LMMS_EVAL_AVAILABLE = False
    # Create dummy classes for type hints when lmms_eval is not installed
    Instance = object
    lmms = object

    def register_model(name):
        def decorator(cls):
            return cls

        return decorator


def _create_vlm_wrapper():
    """Factory function to create the VLM wrapper class.

    This is wrapped in a function to handle the case where lmms_eval
    is not installed, allowing the module to be imported without errors.
    """
    if not LMMS_EVAL_AVAILABLE:
        raise ImportError(
            "lmms_eval is required for the VLM wrapper. "
            "Install with: pip install lmms-eval"
        )

    @register_model("vlm")
    class VLMWrapper(lmms):
        """Unified wrapper for SmolVLM2 and PerceptionLM models.

        This wrapper provides a consistent interface for evaluating both
        model families through lmms-eval. It auto-detects the model family
        and applies appropriate settings.

        Args:
            pretrained: HuggingFace model ID or path to local checkpoint.
            revision: Model revision/branch (default: "main").
            device: Device to load model on (default: "cuda").
            dtype: Model dtype (default: "bfloat16").
            batch_size: Batch size for generation (default: 1).
            attn_implementation: Attention implementation (auto-detected).
            device_map: Device mapping strategy (default: "auto").
            max_frames_num: Maximum video frames (default: 32).
            use_cache: Use KV cache during generation (default: True).
            trust_remote_code: Trust remote code (default: True).
        """

        def __init__(
            self,
            pretrained: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            revision: str = "main",
            device: str = "cuda",
            dtype: str = "bfloat16",
            batch_size: int = 1,
            attn_implementation: Optional[str] = None,
            device_map: str = "auto",
            max_frames_num: int = 32,
            use_cache: bool = True,
            trust_remote_code: bool = True,
            **kwargs,
        ):
            super().__init__()

            from transformers import AutoModelForImageTextToText, AutoProcessor

            # Get model configuration
            self._config: ModelConfig = get_model_config(pretrained)
            logger.info(
                f"Loading {self._config.name} ({self._config.family.value}) "
                f"from {pretrained}"
            )

            # Set up dtype
            self._dtype = getattr(torch, dtype)
            self._device = device
            self._batch_size = batch_size
            self.max_frames_num = max_frames_num

            # Determine attention implementation
            if attn_implementation is None:
                attn_implementation = self._config.attn_implementation
            logger.info(f"Using attention implementation: {attn_implementation}")

            # Load model
            model_kwargs = {
                "revision": revision,
                "torch_dtype": self._dtype,
                "device_map": device_map,
                "attn_implementation": attn_implementation,
                "trust_remote_code": trust_remote_code,
            }
            model_kwargs.update(kwargs)

            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained, **model_kwargs
            )
            self._model.eval()

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                use_fast=self._config.use_fast_processor,
            )

            self._use_cache = use_cache

            # Get actual device after loading (handles device_map="auto")
            self._device = next(self._model.parameters()).device

            logger.info(
                f"Model loaded successfully on {self._device} "
                f"with dtype {self._dtype}"
            )

        @property
        def batch_size(self) -> int:
            return self._batch_size

        @property
        def device(self) -> torch.device:
            return self._device

        @property
        def rank(self) -> int:
            return 0

        @property
        def world_size(self) -> int:
            return 1

        @property
        def config(self) -> ModelConfig:
            """Get the model configuration."""
            return self._config

        def tok_encode(self, string: str, **kwargs) -> List[int]:
            """Encode text to token IDs."""
            return self._processor.tokenizer.encode(string, **kwargs)

        def tok_decode(self, tokens: List[int], **kwargs) -> str:
            """Decode token IDs to text."""
            return self._processor.tokenizer.decode(tokens, **kwargs)

        def _extract_video_frames(
            self,
            video_path: str,
            max_frames: Optional[int] = None,
        ) -> List[Image.Image]:
            """Extract frames from a video file.

            Args:
                video_path: Path to video file.
                max_frames: Maximum number of frames to extract.

            Returns:
                List of PIL Images.
            """
            max_frames = max_frames or self.max_frames_num

            try:
                # Try decord first (fastest)
                import decord

                decord.bridge.set_bridge("torch")
                vr = decord.VideoReader(video_path)
                total_frames = len(vr)

                # Sample frames uniformly
                if total_frames <= max_frames:
                    indices = list(range(total_frames))
                else:
                    indices = [
                        int(i * total_frames / max_frames) for i in range(max_frames)
                    ]

                frames = vr.get_batch(indices).numpy()
                return [Image.fromarray(frame) for frame in frames]

            except ImportError:
                pass

            try:
                # Try torchvision
                import torchvision.io as io

                video, _, info = io.read_video(video_path, pts_unit="sec")
                total_frames = video.shape[0]

                if total_frames <= max_frames:
                    indices = list(range(total_frames))
                else:
                    indices = [
                        int(i * total_frames / max_frames) for i in range(max_frames)
                    ]

                frames = video[indices].numpy()
                return [Image.fromarray(frame) for frame in frames]

            except (ImportError, Exception):
                pass

            try:
                # Fallback to av (PyAV)
                import av

                container = av.open(video_path)
                frames = []

                for frame in container.decode(video=0):
                    frames.append(frame.to_image())
                    if len(frames) >= max_frames * 2:
                        break

                container.close()

                # Sample if too many frames
                if len(frames) > max_frames:
                    indices = [
                        int(i * len(frames) / max_frames) for i in range(max_frames)
                    ]
                    frames = [frames[i] for i in indices]

                return frames

            except ImportError:
                raise ImportError(
                    "No video backend available. Install one of: "
                    "decord, torchvision, av (PyAV)"
                )

        def _build_messages(
            self,
            context: str,
            visuals: List[Union[Image.Image, str]],
        ) -> List[dict]:
            """Build chat messages from context and visuals.

            Args:
                context: Text prompt.
                visuals: List of images (PIL) or video paths (str).

            Returns:
                List of message dictionaries for apply_chat_template.
            """
            content = []

            # Process visuals
            if visuals:
                for visual in visuals:
                    if isinstance(visual, str):
                        # Check if it's a video path
                        if any(
                            visual.lower().endswith(ext)
                            for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
                        ):
                            content.append({"type": "video", "video": visual})
                        else:
                            # Image path
                            content.append({"type": "image", "image": visual})
                    elif isinstance(visual, Image.Image):
                        content.append({"type": "image", "image": visual})
                    else:
                        # Assume it's an image-like object
                        content.append({"type": "image", "image": visual})

            # Add text prompt
            content.append({"type": "text", "text": context})

            return [{"role": "user", "content": content}]

        def generate_until(
            self,
            requests: List[Instance],
        ) -> List[str]:
            """Generate responses for a batch of requests.

            This is the main generation method called by lmms-eval.

            Args:
                requests: List of Instance objects containing prompts and visuals.

            Returns:
                List of generated text responses.
            """
            results = []

            for request in requests:
                # Extract request components
                # lmms-eval passes (context, gen_kwargs) as arguments
                context = request.arguments[0]
                gen_kwargs = (
                    request.arguments[1] if len(request.arguments) > 1 else {}
                )

                # Get visuals from request
                visuals = getattr(request, "visuals", []) or []

                # Build messages
                messages = self._build_messages(context, visuals)

                # Prepare processor kwargs
                processor_kwargs = {
                    "add_generation_prompt": True,
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                }

                # Add video-specific kwargs if processing video
                has_video = any(
                    isinstance(v, str)
                    and any(
                        v.lower().endswith(ext)
                        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
                    )
                    for v in visuals
                )
                if has_video:
                    processor_kwargs["num_frames"] = self.max_frames_num
                    # PerceptionLM-specific
                    if self._config.family == ModelFamily.PERCEPTIONLM:
                        processor_kwargs["video_load_backend"] = (
                            self._config.video_load_backend
                        )

                # Process inputs
                inputs = self._processor.apply_chat_template(
                    messages, **processor_kwargs
                )
                inputs = inputs.to(self._device)

                # Convert to correct dtype
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)

                # Generation parameters
                max_new_tokens = gen_kwargs.get("max_new_tokens", 512)
                temperature = gen_kwargs.get("temperature", 0.0)
                do_sample = temperature > 0
                top_p = gen_kwargs.get("top_p", 1.0)

                # Generate
                with torch.inference_mode():
                    generation_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "use_cache": self._use_cache,
                        "do_sample": do_sample,
                    }

                    if do_sample:
                        generation_kwargs["temperature"] = temperature
                        generation_kwargs["top_p"] = top_p

                    output_ids = self._model.generate(**inputs, **generation_kwargs)

                # Decode - remove input tokens
                input_len = inputs["input_ids"].shape[1]
                generated_ids = output_ids[0, input_len:]
                response = self._processor.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                results.append(response.strip())

            return results

        def loglikelihood(
            self, requests: List[Instance]
        ) -> List[Tuple[float, bool]]:
            """Compute log-likelihood (not implemented for VLMs).

            Most VLM evaluations use generation-based metrics.
            This method raises NotImplementedError.
            """
            raise NotImplementedError(
                "VLM evaluation uses generate_until for generation-based tasks. "
                "Log-likelihood computation is not supported."
            )

        def generate_until_multi_round(
            self,
            requests: List[Instance],
        ) -> List[str]:
            """Multi-round generation (delegates to single-round).

            For benchmarks requiring multi-turn conversations.
            Currently delegates to standard generation.
            """
            return self.generate_until(requests)

    return VLMWrapper


# Create the wrapper class if lmms_eval is available
if LMMS_EVAL_AVAILABLE:
    VLMWrapper = _create_vlm_wrapper()
else:
    VLMWrapper = None


def get_vlm_wrapper():
    """Get the VLM wrapper class.

    Returns:
        The VLMWrapper class.

    Raises:
        ImportError: If lmms_eval is not installed.
    """
    if VLMWrapper is None:
        raise ImportError(
            "lmms_eval is required for the VLM wrapper. "
            "Install with: pip install lmms-eval"
        )
    return VLMWrapper
