"""Shared pytest fixtures for SmolVLM2 data loading tests.

Provides platform detection, device management, and mock data generation
for cross-platform testing on Linux+CUDA and Apple M4 Silicon.
"""

import platform
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np
from PIL import Image


# ============================================================================
# Platform Detection Functions
# ============================================================================

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


def has_cuda() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def has_mps() -> bool:
    """Check if MPS (Apple Metal) is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_available_device() -> str:
    """Get the best available device."""
    if has_cuda():
        return "cuda"
    elif has_mps():
        return "mps"
    return "cpu"


def get_dtype_for_platform() -> torch.dtype:
    """Get appropriate dtype for current platform."""
    if has_cuda():
        return torch.bfloat16
    elif has_mps():
        return torch.float16
    return torch.float32


def has_decord() -> bool:
    """Check if decord is available."""
    try:
        import decord
        return True
    except ImportError:
        return False


def has_av() -> bool:
    """Check if PyAV is available."""
    try:
        import av
        return True
    except ImportError:
        return False


def has_torchvision_video() -> bool:
    """Check if torchvision video support is available."""
    try:
        from torchvision.io import read_video
        return True
    except ImportError:
        return False


def get_video_backend() -> Optional[str]:
    """Get the appropriate video backend for this platform."""
    if is_apple_silicon():
        if has_torchvision_video():
            return "torchvision"
        elif has_av():
            return "av"
    else:
        if has_decord():
            return "decord"
        elif has_torchvision_video():
            return "torchvision"
        elif has_av():
            return "av"
    return None


# ============================================================================
# pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "cuda: marks tests as requiring CUDA")
    config.addinivalue_line("markers", "mps: marks tests as requiring MPS (Apple Metal)")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "download_data: marks tests that download data from HuggingFace")
    config.addinivalue_line("markers", "video: marks tests requiring video processing")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on platform and available hardware."""
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_mps = pytest.mark.skip(reason="MPS not available")
    skip_video = pytest.mark.skip(reason="No video backend available")

    for item in items:
        if "cuda" in item.keywords and not has_cuda():
            item.add_marker(skip_cuda)
        if "mps" in item.keywords and not has_mps():
            item.add_marker(skip_mps)
        if "video" in item.keywords and get_video_backend() is None:
            item.add_marker(skip_video)


# ============================================================================
# Platform Info Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def platform_info() -> Dict[str, Any]:
    """Provide platform information for tests."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "is_apple_silicon": is_apple_silicon(),
        "is_linux": is_linux(),
        "has_cuda": has_cuda(),
        "has_mps": has_mps(),
        "device": get_available_device(),
        "dtype": get_dtype_for_platform(),
        "video_backend": get_video_backend(),
        "has_decord": has_decord(),
        "has_av": has_av(),
        "has_torchvision_video": has_torchvision_video(),
    }


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get torch device for tests."""
    return torch.device(get_available_device())


@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    """Get torch dtype for tests."""
    return get_dtype_for_platform()


# ============================================================================
# Mock Tokenizer and Processor Fixtures
# ============================================================================

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing without downloading models."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.bos_token_id = 2
    tokenizer.vocab_size = 32000

    def mock_call(text, **kwargs):
        max_length = kwargs.get("max_length", 2048)
        padding = kwargs.get("padding", False)
        return_tensors = kwargs.get("return_tensors", None)

        # Deterministic length based on text
        seq_len = min(len(str(text)) // 2 + 10, max_length)

        if padding == "max_length":
            seq_len = max_length

        input_ids = torch.randint(0, 32000, (seq_len,))
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        if return_tensors == "pt":
            return {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
            }
        return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()}

    tokenizer.side_effect = mock_call
    tokenizer.return_value = mock_call("test", return_tensors="pt")

    return tokenizer


@pytest.fixture
def mock_processor():
    """Create a mock image processor for testing."""
    processor = MagicMock()

    def mock_call(images=None, **kwargs):
        if images is None:
            return {"pixel_values": None}

        # Handle single image or list of images
        if isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = 1

        # SigLIP typically outputs 384x384
        pixel_values = torch.randn(batch_size, 3, 384, 384)

        if batch_size == 1:
            pixel_values = pixel_values.squeeze(0)

        return {"pixel_values": pixel_values}

    processor.side_effect = mock_call
    processor.return_value = mock_call([Image.new("RGB", (100, 100))])

    return processor


# ============================================================================
# Mock Image and Video Data Fixtures
# ============================================================================

@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a sample PIL image for testing."""
    return Image.fromarray(
        np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    )


@pytest.fixture
def sample_pil_images() -> List[Image.Image]:
    """Create multiple sample PIL images for testing."""
    return [
        Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
        for _ in range(5)
    ]


@pytest.fixture
def sample_video_frames() -> List[Image.Image]:
    """Create sample video frames as PIL images."""
    return [
        Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
        for _ in range(32)
    ]


@pytest.fixture
def sample_pixel_values() -> torch.Tensor:
    """Create sample pixel values tensor."""
    return torch.randn(3, 384, 384)


@pytest.fixture
def sample_video_tensor() -> torch.Tensor:
    """Create sample video tensor (num_frames, C, H, W)."""
    return torch.randn(32, 3, 384, 384)


# ============================================================================
# Mock Sample and Batch Fixtures
# ============================================================================

@pytest.fixture
def mock_smolvlm_sample(sample_pil_image):
    """Create a mock SmolVLMSample for testing."""
    from src.data.dataset_loaders import SmolVLMSample

    return SmolVLMSample(
        images=[sample_pil_image],
        video_frames=[],
        conversations=[
            {"role": "user", "content": "<image>What is in this image?"},
            {"role": "assistant", "content": "This is a test image."},
        ],
        metadata={"source": "test", "id": "test_001"},
    )


@pytest.fixture
def mock_raw_cauldron_sample(sample_pil_image):
    """Create a mock raw sample in The Cauldron format."""
    return {
        "image": sample_pil_image,
        "conversations": [
            {"role": "user", "content": "Describe this image."},
            {"role": "assistant", "content": "This is a test description."},
        ],
    }


@pytest.fixture
def mock_raw_docmatix_sample(sample_pil_image):
    """Create a mock raw sample in Docmatix format."""
    return {
        "images": [sample_pil_image],
        "messages": [
            {"role": "user", "content": "What text is in this document?"},
            {"role": "assistant", "content": "The document contains test text."},
        ],
    }


@pytest.fixture
def mock_raw_video_sample(sample_video_frames):
    """Create a mock raw video sample."""
    return {
        "video": sample_video_frames,
        "conversations": [
            {"role": "user", "content": "<video>What happens in this video?"},
            {"role": "assistant", "content": "The video shows a test sequence."},
        ],
    }


@pytest.fixture
def mock_processed_batch():
    """Create a mock processed batch for collator testing."""
    batch_size = 4
    seq_len = 128

    features = []
    for i in range(batch_size):
        # Vary sequence lengths to test padding
        actual_len = seq_len - i * 10
        features.append({
            "input_ids": torch.randint(0, 32000, (actual_len,)),
            "attention_mask": torch.ones(actual_len, dtype=torch.long),
            "labels": torch.randint(0, 32000, (actual_len,)),
            "pixel_values": torch.randn(3, 384, 384),
        })

    return features


@pytest.fixture
def mock_video_batch():
    """Create a mock batch with video frames for video collator testing."""
    batch_size = 4
    seq_len = 256

    features = []
    for i in range(batch_size):
        num_frames = 16 + i * 4  # Vary number of frames
        actual_len = seq_len - i * 10
        features.append({
            "input_ids": torch.randint(0, 32000, (actual_len,)),
            "attention_mask": torch.ones(actual_len, dtype=torch.long),
            "labels": torch.randint(0, 32000, (actual_len,)),
            "pixel_values": torch.randn(3, 384, 384),
            "video_frames": torch.randn(num_frames, 3, 384, 384),
        })

    return features


# ============================================================================
# Mock IterableDataset Fixtures
# ============================================================================

@pytest.fixture
def mock_iterable_dataset():
    """Factory fixture for creating mock iterable datasets."""
    class MockIterableDataset:
        def __init__(self, samples: List[Dict], name: str = "mock"):
            self.samples = samples
            self.name = name
            self._iter_count = 0

        def __iter__(self):
            self._iter_count = 0
            for sample in self.samples:
                yield sample.copy()  # Copy to avoid mutation
                self._iter_count += 1

        def take(self, n: int):
            return self.samples[:n]

    return MockIterableDataset


@pytest.fixture
def mock_vision_datasets(mock_iterable_dataset, sample_pil_image):
    """Create mock vision stage datasets."""
    cauldron_samples = [
        {
            "image": sample_pil_image,
            "conversations": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}."},
            ],
        }
        for i in range(100)
    ]
    docmatix_samples = [
        {
            "images": [sample_pil_image],
            "messages": [
                {"role": "user", "content": f"Doc question {i}?"},
                {"role": "assistant", "content": f"Doc answer {i}."},
            ],
        }
        for i in range(100)
    ]

    return {
        "the_cauldron": mock_iterable_dataset(cauldron_samples, "the_cauldron"),
        "docmatix": mock_iterable_dataset(docmatix_samples, "docmatix"),
    }


# ============================================================================
# Temporary File and Video Fixtures
# ============================================================================

@pytest.fixture
def temp_video_file(tmp_path) -> Optional[Path]:
    """Create a temporary synthetic video file for testing.

    Uses torchvision or av depending on platform.
    Returns None if no video creation backend is available.
    """
    video_path = tmp_path / "test_video.mp4"

    try:
        if has_torchvision_video():
            import torchvision.io as vio
            # Create a simple video tensor: (T, H, W, C) uint8
            video_tensor = torch.randint(
                0, 255, (30, 64, 64, 3), dtype=torch.uint8
            )
            vio.write_video(str(video_path), video_tensor, fps=30)
            return video_path
        elif has_av():
            import av
            container = av.open(str(video_path), mode='w')
            stream = container.add_stream('h264', rate=30)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = 'yuv420p'

            for i in range(30):
                frame = av.VideoFrame.from_ndarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                    format='rgb24'
                )
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
            container.close()
            return video_path
    except Exception:
        return None

    return None


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "vision").mkdir(parents=True)
    (data_dir / "video").mkdir(parents=True)
    return data_dir
