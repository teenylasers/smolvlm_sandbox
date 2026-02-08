"""Mock data generators for SmolVLM2 tests.

Provides utilities for generating synthetic test data without
requiring network access or real model weights.
"""

from typing import Dict, List, Any, Optional
import torch
import numpy as np
from PIL import Image


def create_random_image(
    width: int = 384,
    height: int = 384,
    mode: str = "RGB",
) -> Image.Image:
    """Create a random PIL image.

    Args:
        width: Image width
        height: Image height
        mode: Image mode (RGB, L, etc.)

    Returns:
        Random PIL image
    """
    if mode == "RGB":
        data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    elif mode == "L":
        data = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    else:
        data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    return Image.fromarray(data, mode=mode)


def create_random_images(
    count: int = 5,
    width: int = 384,
    height: int = 384,
) -> List[Image.Image]:
    """Create multiple random images.

    Args:
        count: Number of images
        width: Image width
        height: Image height

    Returns:
        List of random PIL images
    """
    return [create_random_image(width, height) for _ in range(count)]


def create_cauldron_sample(
    image: Optional[Image.Image] = None,
    question: str = "What is in this image?",
    answer: str = "This is a test image.",
) -> Dict[str, Any]:
    """Create a sample in The Cauldron format.

    Args:
        image: PIL image (creates random if None)
        question: Question text
        answer: Answer text

    Returns:
        Sample dictionary in Cauldron format
    """
    if image is None:
        image = create_random_image()

    return {
        "image": image,
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def create_docmatix_sample(
    images: Optional[List[Image.Image]] = None,
    question: str = "What text is in this document?",
    answer: str = "The document contains test text.",
) -> Dict[str, Any]:
    """Create a sample in Docmatix format.

    Args:
        images: List of PIL images (creates random if None)
        question: Question text
        answer: Answer text

    Returns:
        Sample dictionary in Docmatix format
    """
    if images is None:
        images = [create_random_image()]

    return {
        "images": images,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def create_video_sample(
    num_frames: int = 16,
    frame_size: int = 384,
    question: str = "What happens in this video?",
    answer: str = "The video shows a test sequence.",
) -> Dict[str, Any]:
    """Create a mock video sample.

    Args:
        num_frames: Number of video frames
        frame_size: Frame dimensions
        question: Question text
        answer: Answer text

    Returns:
        Sample dictionary with video frames
    """
    frames = create_random_images(num_frames, frame_size, frame_size)

    return {
        "video": frames,
        "conversations": [
            {"role": "user", "content": f"<video>{question}"},
            {"role": "assistant", "content": answer},
        ],
    }


def create_processed_sample(
    seq_len: int = 128,
    has_image: bool = True,
    has_video: bool = False,
    num_frames: int = 16,
    vocab_size: int = 32000,
) -> Dict[str, torch.Tensor]:
    """Create a processed sample ready for collation.

    Args:
        seq_len: Sequence length
        has_image: Whether to include pixel_values
        has_video: Whether to include video_frames
        num_frames: Number of video frames
        vocab_size: Vocabulary size for token IDs

    Returns:
        Dictionary with processed tensors
    """
    sample = {
        "input_ids": torch.randint(0, vocab_size, (seq_len,)),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": torch.randint(0, vocab_size, (seq_len,)),
    }

    if has_image:
        sample["pixel_values"] = torch.randn(3, 384, 384)

    if has_video:
        sample["video_frames"] = torch.randn(num_frames, 3, 384, 384)

    return sample


def create_batch_samples(
    batch_size: int = 4,
    base_seq_len: int = 128,
    vary_length: bool = True,
    has_images: bool = True,
    has_video: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """Create a batch of processed samples.

    Args:
        batch_size: Number of samples
        base_seq_len: Base sequence length
        vary_length: Whether to vary sequence lengths
        has_images: Whether to include images
        has_video: Whether to include video

    Returns:
        List of processed sample dictionaries
    """
    samples = []
    for i in range(batch_size):
        seq_len = base_seq_len - (i * 10 if vary_length else 0)
        num_frames = 16 + (i * 4 if vary_length else 0)

        sample = create_processed_sample(
            seq_len=seq_len,
            has_image=has_images,
            has_video=has_video,
            num_frames=num_frames,
        )
        samples.append(sample)

    return samples


def create_mock_dataset_samples(
    format_type: str = "cauldron",
    count: int = 10,
) -> List[Dict[str, Any]]:
    """Create multiple mock dataset samples.

    Args:
        format_type: Sample format ("cauldron", "docmatix", "video")
        count: Number of samples

    Returns:
        List of sample dictionaries
    """
    samples = []
    for i in range(count):
        if format_type == "cauldron":
            sample = create_cauldron_sample(
                question=f"Question {i}?",
                answer=f"Answer {i}.",
            )
        elif format_type == "docmatix":
            sample = create_docmatix_sample(
                question=f"Document question {i}?",
                answer=f"Document answer {i}.",
            )
        elif format_type == "video":
            sample = create_video_sample(
                question=f"Video question {i}?",
                answer=f"Video answer {i}.",
            )
        else:
            raise ValueError(f"Unknown format type: {format_type}")

        samples.append(sample)

    return samples


class MockTokenizer:
    """Mock tokenizer for testing without model downloads."""

    def __init__(
        self,
        vocab_size: int = 32000,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

    def __call__(
        self,
        text: str,
        max_length: int = 2048,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tokenize text."""
        # Deterministic length based on text
        seq_len = min(len(str(text)) // 2 + 10, max_length)

        if padding == "max_length":
            seq_len = max_length

        input_ids = torch.randint(0, self.vocab_size, (seq_len,))
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        if return_tensors == "pt":
            return {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
            }
        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
        }

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        max_length = kwargs.get("max_length", 2048)
        return list(range(min(len(text) // 2, max_length)))


class MockImageProcessor:
    """Mock image processor for testing without model downloads."""

    def __init__(self, image_size: int = 384):
        self.image_size = image_size

    def __call__(
        self,
        images=None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> Dict[str, Any]:
        """Process images."""
        if images is None:
            return {"pixel_values": None}

        # Handle single image or list
        if isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = 1

        pixel_values = torch.randn(batch_size, 3, self.image_size, self.image_size)

        if batch_size == 1:
            pixel_values = pixel_values.squeeze(0)

        return {"pixel_values": pixel_values}
