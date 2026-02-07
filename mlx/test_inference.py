#!/usr/bin/env python3
"""Test SmolVLM2 inference on MLX (Apple Silicon).

This script validates that SmolVLM2 works correctly on your M4 MacBook
before running full training on the cluster.

Usage:
    # Test image inference
    python mlx/test_inference.py --image path/to/image.jpg --prompt "Describe this image"

    # Test video inference
    python mlx/test_inference.py --video path/to/video.mp4 --prompt "What's happening?"

    # Use specific model size
    python mlx/test_inference.py --model-size 500m --image test.jpg
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Model mappings
MLX_MODELS = {
    "256m": "mlx-community/SmolVLM2-256M-Video-Instruct-mlx",
    "500m": "mlx-community/SmolVLM2-500M-Video-Instruct-mlx",
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
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/pcuenca/mlx-vlm.git@smolvlm"
    ], check=True)
    print("Installation complete!")


def test_image_inference(
    model_id: str,
    image_path: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
):
    """Test image inference with SmolVLM2.

    Args:
        model_id: HuggingFace model ID
        image_path: Path to image file
        prompt: Text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    print(f"\n{'='*60}")
    print(f"Testing Image Inference")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "mlx_vlm.generate",
        "--model", model_id,
        "--image", image_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Output:")
        print(result.stdout)
    else:
        print("Error:")
        print(result.stderr)
        return False

    return True


def test_video_inference(
    model_id: str,
    video_path: str,
    prompt: str,
    system_prompt: str = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
):
    """Test video inference with SmolVLM2.

    Args:
        model_id: HuggingFace model ID
        video_path: Path to video file
        prompt: Text prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    print(f"\n{'='*60}")
    print(f"Testing Video Inference")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Video: {video_path}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "mlx_vlm.smolvlm_video_generate",
        "--model", model_id,
        "--video", video_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature),
    ]

    if system_prompt:
        cmd.extend(["--system", system_prompt])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Output:")
        print(result.stdout)
    else:
        print("Error:")
        print(result.stderr)
        return False

    return True


def test_python_api(model_id: str, image_path: str, prompt: str):
    """Test inference using Python API directly.

    This tests that the MLX model can be loaded and used programmatically.
    """
    print(f"\n{'='*60}")
    print(f"Testing Python API")
    print(f"{'='*60}\n")

    try:
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        from PIL import Image

        print("Loading model...")
        model, processor = load(model_id)
        config = load_config(model_id)

        print("Loading image...")
        image = Image.open(image_path)

        print("Preparing prompt...")
        formatted_prompt = apply_chat_template(
            processor,
            config,
            prompt,
            num_images=1,
        )

        print("Generating response...")
        output = generate(
            model,
            processor,
            formatted_prompt,
            image,
            max_tokens=256,
            verbose=False,
        )

        print(f"\nResponse: {output}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SmolVLM2 inference on MLX")
    parser.add_argument(
        "--model-size",
        choices=["256m", "500m"],
        default="256m",
        help="Model size to test",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file for testing",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file for testing",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Prompt for the model",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt (for video)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install mlx-vlm if not present",
    )
    parser.add_argument(
        "--use-python-api",
        action="store_true",
        help="Use Python API instead of CLI",
    )

    args = parser.parse_args()

    # Check/install mlx-vlm
    if not check_mlx_vlm_installed():
        if args.install:
            install_mlx_vlm()
        else:
            print("mlx-vlm not installed. Run with --install or:")
            print("  pip install git+https://github.com/pcuenca/mlx-vlm.git@smolvlm")
            sys.exit(1)

    model_id = MLX_MODELS[args.model_size]

    if args.image:
        if not Path(args.image).exists():
            print(f"Image not found: {args.image}")
            sys.exit(1)

        if args.use_python_api:
            success = test_python_api(model_id, args.image, args.prompt)
        else:
            success = test_image_inference(
                model_id,
                args.image,
                args.prompt,
                args.max_tokens,
            )

    elif args.video:
        if not Path(args.video).exists():
            print(f"Video not found: {args.video}")
            sys.exit(1)

        success = test_video_inference(
            model_id,
            args.video,
            args.prompt,
            args.system_prompt,
            args.max_tokens,
        )

    else:
        # Run quick sanity check with a test URL
        print("No image/video provided. Running quick test with sample image...")
        test_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
        success = test_image_inference(
            model_id,
            test_url,
            "What is in this image?",
            max_tokens=100,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
