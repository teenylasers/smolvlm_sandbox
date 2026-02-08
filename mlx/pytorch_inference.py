"""PyTorch MPS inference for PerceptionLM on Apple Silicon.

This module provides PyTorch-based inference for PerceptionLM models on Apple Silicon
using the MPS (Metal Performance Shaders) backend.

Usage:
    from mlx.pytorch_inference import PyTorchModel

    model = PyTorchModel(model_id="facebook/Perception-LM-1B")
    response = model.generate_response(
        video_path="video.mp4",
        prompt="What is happening in this video?",
    )
"""

import gc
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# PerceptionLM model configurations
PYTORCH_MODELS = {
    "plm-1b": "facebook/Perception-LM-1B",
    "plm-3b": "facebook/Perception-LM-3B",
}


def check_mps_available() -> bool:
    """Check if MPS backend is available."""
    return torch.backends.mps.is_available()


def get_device() -> str:
    """Get the best available device for inference."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PyTorchModel:
    """PyTorch-based model for local inference on Apple Silicon."""

    def __init__(
        self,
        model_size: str = "plm-1b",
        max_tokens: int = 128,
        temperature: float = 0.0,
        max_frames: int = 16,
    ):
        """Initialize the PyTorch model.

        Args:
            model_size: Model size ("plm-1b" or "plm-3b").
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            max_frames: Maximum video frames to process.
        """
        if model_size not in PYTORCH_MODELS:
            raise ValueError(
                f"Invalid model size: {model_size}. "
                f"Available: {list(PYTORCH_MODELS.keys())}"
            )

        self.model_size = model_size
        self.model_id = PYTORCH_MODELS[model_size]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_frames = max_frames
        self.device = get_device()

        self._model = None
        self._processor = None

    def load(self):
        """Load the model into memory."""
        if self._model is not None:
            return

        print(f"Loading model: {self.model_id} on {self.device}...")

        # Set environment variable to help with MPS compatibility
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            device_map=self.device,
            # Use eager attention for MPS compatibility
            attn_implementation="eager",
        )

        print(f"Model loaded successfully on {self.device}!")

    def unload(self):
        """Unload the model to free memory."""
        self._model = None
        self._processor = None
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()

    def _load_video_frames(self, video_path: str) -> List:
        """Load and sample frames from a video file.

        Args:
            video_path: Path to video file.

        Returns:
            List of PIL Images.
        """
        from PIL import Image

        # Try decord first (faster), fall back to av
        try:
            import decord
            decord.bridge.set_bridge("native")

            vr = decord.VideoReader(video_path)
            total_frames = len(vr)

            # Sample frames uniformly
            num_frames = min(self.max_frames, total_frames)
            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

            frames = vr.get_batch(indices).asnumpy()
            return [Image.fromarray(frame) for frame in frames]

        except ImportError:
            pass

        # Fallback to av
        try:
            import av

            container = av.open(video_path)
            stream = container.streams.video[0]
            total_frames = stream.frames or 1000  # Estimate if unknown

            # Sample frames uniformly
            num_frames = min(self.max_frames, total_frames)

            frames = []
            frame_indices = set(
                int(i * total_frames / num_frames) for i in range(num_frames)
            )

            for i, frame in enumerate(container.decode(video=0)):
                if i in frame_indices:
                    frames.append(frame.to_image())
                if len(frames) >= num_frames:
                    break

            container.close()
            return frames

        except ImportError:
            raise ImportError(
                "No video decoder available. Install decord or av:\n"
                "  pip install decord  # or\n"
                "  pip install av"
            )

    def generate_response(
        self,
        video_path: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response for a video + prompt.

        Args:
            video_path: Path to video file.
            prompt: Text prompt/question.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            Generated text response.
        """
        if self._model is None:
            self.load()

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        try:
            # Load video frames
            frames = self._load_video_frames(video_path)

            if not frames:
                print(f"Warning: No frames loaded from {video_path}")
                return ""

            # Build conversation format for PerceptionLM
            # PerceptionLM expects video frames as a list
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Process inputs
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                if temperature == 0:
                    # Greedy decoding
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                    )
                else:
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                    )

            # Decode response
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = self._processor.decode(generated_ids, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def batch_generate(
        self,
        samples: List[Tuple[str, str]],
        show_progress: bool = True,
    ) -> List[str]:
        """Generate responses for multiple video+prompt pairs.

        Args:
            samples: List of (video_path, prompt) tuples.
            show_progress: Show progress indicator.

        Returns:
            List of generated responses.
        """
        responses = []

        for i, (video_path, prompt) in enumerate(samples):
            if show_progress:
                print(f"  Processing {i+1}/{len(samples)}...", end="\r")

            response = self.generate_response(video_path, prompt)
            responses.append(response)

            # Clear memory periodically
            if (i + 1) % 5 == 0:
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()

        if show_progress:
            print()  # New line after progress

        return responses


def create_model(
    model_size: str = "plm-1b",
    max_tokens: int = 128,
    temperature: float = 0.0,
    max_frames: int = 16,
) -> PyTorchModel:
    """Create and load a PyTorch model.

    Args:
        model_size: Model size ("plm-1b" or "plm-3b").
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        max_frames: Maximum video frames.

    Returns:
        Loaded PyTorchModel instance.
    """
    model = PyTorchModel(
        model_size=model_size,
        max_tokens=max_tokens,
        temperature=temperature,
        max_frames=max_frames,
    )
    model.load()
    return model


def list_models() -> str:
    """List available PyTorch models.

    Returns:
        Formatted string listing models.
    """
    lines = ["Available PyTorch Models (MPS):"]
    lines.append(f"  Device: {get_device()}")
    lines.append("")
    for size, model_id in PYTORCH_MODELS.items():
        lines.append(f"  {size}: {model_id}")
    return "\n".join(lines)
