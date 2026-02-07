#!/usr/bin/env python3
"""Test data loading pipeline locally on MLX.

Validates that the data pipeline works correctly before running
full training on the cluster. Uses small samples of each dataset.

Usage:
    # Test all dataset loaders
    python mlx/test_data_loading.py

    # Test specific dataset
    python mlx/test_data_loading.py --dataset the_cauldron

    # Download small sample for testing
    python mlx/test_data_loading.py --download --num-samples 100
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_dataset_download(dataset_name: str, num_samples: int = 100) -> bool:
    """Test downloading a small sample from a dataset.

    Args:
        dataset_name: Name of dataset to test
        num_samples: Number of samples to download

    Returns:
        True if successful
    """
    from datasets import load_dataset

    DATASET_CONFIGS = {
        "the_cauldron": {
            "path": "HuggingFaceM4/the_cauldron",
            "split": "train",
            "streaming": True,
        },
        "docmatix": {
            "path": "HuggingFaceM4/Docmatix",
            "name": "images",
            "split": "train",
            "streaming": True,
        },
        "llava_onevision": {
            "path": "lmms-lab/LLaVA-OneVision-Data",
            "split": "train",
            "streaming": True,
        },
        "llava_video": {
            "path": "lmms-lab/LLaVA-Video-178K",
            "split": "train",
            "streaming": True,
        },
    }

    if dataset_name not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASET_CONFIGS.keys())}")
        return False

    config = DATASET_CONFIGS[dataset_name]
    print(f"\nTesting dataset: {dataset_name}")
    print(f"Config: {config}")

    try:
        # Load with streaming to avoid downloading full dataset
        ds = load_dataset(
            config["path"],
            name=config.get("name"),
            split=config["split"],
            streaming=config.get("streaming", True),
            trust_remote_code=True,
        )

        # Take samples
        samples = list(ds.take(num_samples))
        print(f"Successfully loaded {len(samples)} samples")

        # Show sample structure
        if samples:
            sample = samples[0]
            print(f"\nSample keys: {list(sample.keys())}")

            # Print sample info
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {type(value).__name__} (len={len(value)})")
                elif hasattr(value, 'shape'):
                    print(f"  {key}: {type(value).__name__} shape={value.shape}")
                elif isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value).__name__} (len={len(value)})")
                else:
                    print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_processing() -> bool:
    """Test image preprocessing with SigLIP processor."""
    print("\nTesting image processing...")

    try:
        from transformers import AutoProcessor
        from PIL import Image
        import numpy as np

        # Load SigLIP processor
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512")
        print(f"Loaded processor: {type(processor)}")

        # Create test image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        )

        # Process image
        inputs = processor(images=test_image, return_tensors="pt")
        print(f"Processed image shape: {inputs['pixel_values'].shape}")

        # Check expected shape
        expected_shape = (1, 3, 512, 512)  # SigLIP uses 512x512
        if inputs['pixel_values'].shape == expected_shape:
            print(f"Shape matches expected: {expected_shape}")
        else:
            print(f"Warning: Expected {expected_shape}, got {inputs['pixel_values'].shape}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_processing(video_path: Optional[str] = None) -> bool:
    """Test video frame extraction with decord."""
    print("\nTesting video processing...")

    try:
        import decord
        import numpy as np
        from PIL import Image

        if video_path and Path(video_path).exists():
            # Use provided video
            vr = decord.VideoReader(video_path)
            print(f"Video: {video_path}")
            print(f"Total frames: {len(vr)}")
            print(f"FPS: {vr.get_avg_fps()}")

            # Extract frames uniformly
            num_frames = min(32, len(vr))
            indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()

            print(f"Extracted {len(frames)} frames, shape: {frames.shape}")
        else:
            # Create synthetic test data
            print("No video provided, creating synthetic test data...")
            frames = np.random.randint(0, 255, (32, 384, 384, 3), dtype=np.uint8)
            print(f"Created {len(frames)} synthetic frames")

        # Convert to PIL images
        pil_frames = [Image.fromarray(f) for f in frames]
        print(f"Converted to {len(pil_frames)} PIL images")

        return True

    except ImportError as e:
        print(f"decord not installed: {e}")
        print("Install with: pip install decord")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenization() -> bool:
    """Test tokenization with SmolLM2 tokenizer."""
    print("\nTesting tokenization...")

    try:
        from transformers import AutoTokenizer

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        print(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
        print(f"Vocab size: {tokenizer.vocab_size}")

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<image>",
                "<video>",
                "<image_start>",
                "<image_end>",
            ]
        }
        num_added = tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens")

        # Test encoding
        test_text = "<image>What is shown in this image?</image>"
        tokens = tokenizer.encode(test_text)
        print(f"Test text: {test_text}")
        print(f"Token IDs: {tokens}")
        print(f"Decoded: {tokenizer.decode(tokens)}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collator() -> bool:
    """Test data collation for batching."""
    print("\nTesting data collator...")

    try:
        import torch
        from transformers import AutoTokenizer, AutoProcessor
        import numpy as np
        from PIL import Image

        # Load components
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512")

        # Create batch of samples
        batch = []
        for i in range(4):
            sample = {
                "input_ids": torch.randint(0, 1000, (128,)),
                "attention_mask": torch.ones(128),
                "labels": torch.randint(0, 1000, (128,)),
                "pixel_values": torch.randn(3, 512, 512),
            }
            batch.append(sample)

        # Manual collation
        collated = {
            "input_ids": torch.stack([s["input_ids"] for s in batch]),
            "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
            "labels": torch.stack([s["labels"] for s in batch]),
            "pixel_values": torch.stack([s["pixel_values"] for s in batch]),
        }

        print("Collated batch shapes:")
        for key, value in collated.items():
            print(f"  {key}: {value.shape}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test data loading pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["the_cauldron", "docmatix", "llava_onevision", "llava_video", "all"],
        default="all",
        help="Dataset to test",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to load for testing",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file for testing video processing",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download tests",
    )

    args = parser.parse_args()

    print("="*60)
    print("SmolVLM2 Data Loading Tests")
    print("="*60)

    results = {}

    # Test image processing
    results["image_processing"] = test_image_processing()

    # Test video processing
    results["video_processing"] = test_video_processing(args.video)

    # Test tokenization
    results["tokenization"] = test_tokenization()

    # Test data collator
    results["data_collator"] = test_data_collator()

    # Test dataset downloads
    if not args.skip_download:
        datasets_to_test = (
            ["the_cauldron", "docmatix", "llava_onevision", "llava_video"]
            if args.dataset == "all"
            else [args.dataset]
        )

        for ds_name in datasets_to_test:
            results[f"dataset_{ds_name}"] = test_dataset_download(
                ds_name, args.num_samples
            )

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
