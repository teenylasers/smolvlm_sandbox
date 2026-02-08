"""MLX model inference for local evaluation.

This module provides MLX-based inference for SmolVLM2 models on Apple Silicon,
optimized for video understanding tasks.

Usage:
    from mlx.mlx_inference import MLXModel

    model = MLXModel(model_size="256m")
    response = model.generate_video_response(
        video_path="video.mp4",
        prompt="What is happening in this video?",
    )
"""

import gc
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# MLX model IDs
MLX_MODELS = {
    "256m": "mlx-community/SmolVLM2-256M-Video-Instruct-mlx",
    "500m": "mlx-community/SmolVLM2-500M-Video-Instruct-mlx",
    "2.2b": "mlx-community/SmolVLM2-2.2B-Instruct-mlx",
}


def check_mlx_vlm_installed() -> bool:
    """Check if mlx-vlm is installed."""
    try:
        import mlx_vlm
        return True
    except ImportError:
        return False


def install_mlx_vlm():
    """Install mlx-vlm with SmolVLM2 support."""
    print("Installing mlx-vlm with SmolVLM2 support...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/pcuenca/mlx-vlm.git@smolvlm",
        ],
        check=True,
    )
    print("Installation complete!")


class MLXModel:
    """MLX-based SmolVLM2 model for local inference."""

    def __init__(
        self,
        model_size: str = "256m",
        max_tokens: int = 128,
        temperature: float = 0.0,
        max_frames: int = 16,
    ):
        """Initialize the MLX model.

        Args:
            model_size: Model size ("256m" or "500m").
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            max_frames: Maximum video frames to process.
        """
        if model_size not in MLX_MODELS:
            raise ValueError(
                f"Invalid model size: {model_size}. "
                f"Available: {list(MLX_MODELS.keys())}"
            )

        if not check_mlx_vlm_installed():
            raise ImportError(
                "mlx-vlm not installed. Install with:\n"
                "  pip install git+https://github.com/pcuenca/mlx-vlm.git@smolvlm"
            )

        self.model_size = model_size
        self.model_id = MLX_MODELS[model_size]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_frames = max_frames

        self._model = None
        self._processor = None
        self._config = None

    def load(self):
        """Load the model into memory."""
        if self._model is not None:
            return

        print(f"Loading model: {self.model_id}...")

        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self._model, self._processor = load(self.model_id)
        self._config = load_config(self.model_id)

        print(f"Model loaded successfully!")

    def unload(self):
        """Unload the model to free memory."""
        self._model = None
        self._processor = None
        self._config = None
        gc.collect()

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

        # Use subprocess-based video generation (more reliable for video)
        # as the Python API for video is less stable
        return self._generate_via_subprocess(
            video_path, prompt, max_tokens, temperature
        )

    def _generate_via_subprocess(
        self,
        video_path: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate response using subprocess (more reliable for video).

        Args:
            video_path: Path to video file.
            prompt: Text prompt.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        cmd = [
            sys.executable,
            "-m",
            "mlx_vlm.smolvlm_video_generate",
            "--model",
            self.model_id,
            "--video",
            video_path,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
            "--temp",
            str(temperature),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout per sample
            )

            if result.returncode == 0:
                # Parse output - mlx_vlm prints the response
                output = result.stdout.strip()
                # Remove any loading/progress messages
                lines = output.split("\n")
                # Find the actual response (usually the last non-empty line)
                for line in reversed(lines):
                    if line.strip() and not line.startswith(("Loading", "Fetching", "=")):
                        return line.strip()
                return output
            else:
                print(f"Error: {result.stderr}")
                return ""

        except subprocess.TimeoutExpired:
            print("Timeout: Response generation took too long")
            return ""
        except Exception as e:
            print(f"Error during generation: {e}")
            return ""

    def _generate_via_python_api(
        self,
        video_path: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate response using Python API directly.

        Note: This is less reliable for video but provided as an alternative.

        Args:
            video_path: Path to video file.
            prompt: Text prompt.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        try:
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template

            # Format the prompt
            formatted_prompt = apply_chat_template(
                self._processor,
                self._config,
                prompt,
                num_images=0,  # Video, not images
            )

            # Load video frames
            # Note: This depends on mlx_vlm's internal video processing
            output = generate(
                self._model,
                self._processor,
                formatted_prompt,
                video_path,
                max_tokens=max_tokens,
                temp=temperature if temperature > 0 else None,
                verbose=False,
            )

            return output.strip()

        except Exception as e:
            print(f"Python API error: {e}")
            # Fallback to subprocess
            return self._generate_via_subprocess(
                video_path, prompt, max_tokens, temperature
            )

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
            if (i + 1) % 10 == 0:
                gc.collect()

        if show_progress:
            print()  # New line after progress

        return responses


def create_model(
    model_size: str = "256m",
    max_tokens: int = 128,
    temperature: float = 0.0,
    max_frames: int = 16,
) -> MLXModel:
    """Create and load an MLX model.

    Args:
        model_size: Model size ("256m" or "500m").
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        max_frames: Maximum video frames.

    Returns:
        Loaded MLXModel instance.
    """
    model = MLXModel(
        model_size=model_size,
        max_tokens=max_tokens,
        temperature=temperature,
        max_frames=max_frames,
    )
    model.load()
    return model


def list_models() -> str:
    """List available MLX models.

    Returns:
        Formatted string listing models.
    """
    lines = ["Available MLX Models:", ""]
    for size, model_id in MLX_MODELS.items():
        lines.append(f"  {size}: {model_id}")
    return "\n".join(lines)
