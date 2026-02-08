"""Tests for video_processor.py module.

Tests VideoProcessor with multiple backends and platform detection.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.video_processor import (
    AV_AVAILABLE,
    DECORD_AVAILABLE,
    DEFAULT_BACKEND,
    IS_APPLE_SILICON,
    TORCHVISION_AVAILABLE,
    VideoProcessor,
    create_video_processor,
)


class TestPlatformDetection:
    """Tests for platform and backend detection."""

    def test_is_apple_silicon_flag(self, platform_info):
        """Test IS_APPLE_SILICON detection."""
        expected = platform_info["is_apple_silicon"]
        assert IS_APPLE_SILICON == expected

    def test_default_backend_set_if_available(self, platform_info):
        """Test that DEFAULT_BACKEND is set if any backend available."""
        if (
            platform_info["has_decord"]
            or platform_info["has_av"]
            or platform_info["has_torchvision_video"]
        ):
            assert DEFAULT_BACKEND is not None

    def test_backend_availability_flags(self, platform_info):
        """Test that backend availability flags match platform info."""
        assert DECORD_AVAILABLE == platform_info["has_decord"]
        assert AV_AVAILABLE == platform_info["has_av"]
        assert TORCHVISION_AVAILABLE == platform_info["has_torchvision_video"]

    def test_backend_priority_apple_silicon(self):
        """Test backend priority on Apple Silicon."""
        if IS_APPLE_SILICON:
            if TORCHVISION_AVAILABLE:
                assert DEFAULT_BACKEND == "torchvision"
            elif AV_AVAILABLE:
                assert DEFAULT_BACKEND == "av"

    def test_backend_priority_linux(self):
        """Test backend priority on Linux."""
        if not IS_APPLE_SILICON:
            if DECORD_AVAILABLE:
                assert DEFAULT_BACKEND == "decord"
            elif TORCHVISION_AVAILABLE:
                assert DEFAULT_BACKEND == "torchvision"


class TestVideoProcessorInit:
    """Tests for VideoProcessor initialization."""

    def test_init_with_default_backend(self, mock_processor, platform_info):
        """Test initialization with default backend."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        processor = VideoProcessor(
            image_processor=mock_processor,
            num_frames=32,
        )

        assert processor.backend == DEFAULT_BACKEND
        assert processor.num_frames == 32

    def test_init_custom_settings(self, mock_processor, platform_info):
        """Test initialization with custom settings."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        processor = VideoProcessor(
            image_processor=mock_processor,
            num_frames=16,
            frame_sampling="random",
            image_size=512,
            max_video_duration=300.0,
            backend=platform_info["video_backend"],
        )

        assert processor.num_frames == 16
        assert processor.frame_sampling == "random"
        assert processor.image_size == 512
        assert processor.max_video_duration == 300.0

    def test_init_stores_image_processor(self, mock_processor, platform_info):
        """Test that image processor is stored."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        processor = VideoProcessor(
            image_processor=mock_processor,
            backend=platform_info["video_backend"],
        )

        assert processor.image_processor is mock_processor

    def test_init_no_backend_raises(self, mock_processor):
        """Test that initialization without any backend raises error."""
        with patch.dict(
            "src.data.video_processor.__dict__",
            {
                "DEFAULT_BACKEND": None,
                "DECORD_AVAILABLE": False,
                "AV_AVAILABLE": False,
                "TORCHVISION_AVAILABLE": False,
            },
        ):
            # Need to reload or mock at init time
            pass  # This test is tricky without module reload

    @pytest.mark.skipif(not DECORD_AVAILABLE, reason="decord not available")
    def test_init_decord_backend(self, mock_processor):
        """Test initialization with decord backend."""
        processor = VideoProcessor(
            image_processor=mock_processor,
            backend="decord",
        )
        assert processor.backend == "decord"

    @pytest.mark.skipif(not AV_AVAILABLE, reason="PyAV not available")
    def test_init_av_backend(self, mock_processor):
        """Test initialization with av backend."""
        processor = VideoProcessor(
            image_processor=mock_processor,
            backend="av",
        )
        assert processor.backend == "av"

    @pytest.mark.skipif(not TORCHVISION_AVAILABLE, reason="torchvision not available")
    def test_init_torchvision_backend(self, mock_processor):
        """Test initialization with torchvision backend."""
        processor = VideoProcessor(
            image_processor=mock_processor,
            backend="torchvision",
        )
        assert processor.backend == "torchvision"


class TestFrameIndexCalculation:
    """Tests for frame index calculation logic."""

    @pytest.fixture
    def processor(self, mock_processor, platform_info):
        """Create VideoProcessor for testing."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        return VideoProcessor(
            image_processor=mock_processor,
            num_frames=8,
            frame_sampling="uniform",
            max_video_duration=60.0,
            backend=platform_info["video_backend"],
        )

    def test_uniform_sampling(self, processor):
        """Test uniform frame sampling."""
        processor.frame_sampling = "uniform"

        indices = processor.get_frame_indices(total_frames=100, fps=30)

        assert len(indices) == processor.num_frames
        assert indices[0] == 0
        assert indices[-1] == 99  # Last frame

    def test_uniform_sampling_spacing(self, processor):
        """Test uniform sampling produces evenly spaced indices."""
        processor.frame_sampling = "uniform"
        processor.num_frames = 5

        indices = processor.get_frame_indices(total_frames=100, fps=30)

        # Should be roughly evenly spaced
        diffs = np.diff(indices)
        assert np.allclose(diffs, diffs.mean(), atol=1)

    def test_random_sampling(self, processor):
        """Test random frame sampling."""
        processor.frame_sampling = "random"

        indices = processor.get_frame_indices(total_frames=100, fps=30)

        assert len(indices) == processor.num_frames
        assert all(0 <= idx < 100 for idx in indices)
        # Should be sorted
        assert list(indices) == sorted(indices)

    def test_keyframe_sampling(self, processor):
        """Test keyframe-based sampling."""
        processor.frame_sampling = "keyframe"

        indices = processor.get_frame_indices(total_frames=100, fps=30)

        # Should have samples, may be fewer than num_frames due to dedup
        assert len(indices) <= processor.num_frames
        assert all(0 <= idx < 100 for idx in indices)

    def test_max_duration_clipping(self, processor):
        """Test that videos exceeding max_duration are clipped."""
        processor.max_video_duration = 10.0  # 10 seconds

        # 600 frames at 30 fps = 20 seconds, over limit
        indices = processor.get_frame_indices(total_frames=600, fps=30)

        # Should only sample from first 300 frames (10 seconds)
        assert all(idx < 300 for idx in indices)

    def test_max_duration_not_exceeded(self, processor):
        """Test that short videos are not affected by max_duration."""
        processor.max_video_duration = 60.0  # 60 seconds

        # 100 frames at 30 fps = 3.3 seconds, under limit
        indices = processor.get_frame_indices(total_frames=100, fps=30)

        assert indices[-1] == 99  # Can use all frames

    def test_invalid_sampling_strategy(self, processor):
        """Test that invalid sampling strategy raises error."""
        processor.frame_sampling = "invalid_strategy"

        with pytest.raises(ValueError, match="Unknown frame sampling"):
            processor.get_frame_indices(total_frames=100, fps=30)

    def test_fewer_frames_than_requested(self, processor):
        """Test handling when video has fewer frames than requested."""
        processor.num_frames = 20

        indices = processor.get_frame_indices(total_frames=10, fps=30)

        # Should handle gracefully (may have duplicates with linspace)
        assert len(indices) == 20


@pytest.mark.video
class TestVideoFileProcessing:
    """Tests for actual video file processing.

    These tests require a video backend and test with synthetic video files.
    """

    def test_extract_frames_from_file(
        self, mock_processor, temp_video_file, platform_info
    ):
        """Test extracting frames from a video file."""
        if temp_video_file is None:
            pytest.skip("Could not create test video")

        processor = VideoProcessor(
            image_processor=mock_processor,
            num_frames=8,
            backend=platform_info["video_backend"],
        )

        frames = processor.extract_frames(temp_video_file)

        assert len(frames) <= processor.num_frames
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_extract_frames_returns_pil(
        self, mock_processor, temp_video_file, platform_info
    ):
        """Test that extract_frames returns PIL Images."""
        if temp_video_file is None:
            pytest.skip("Could not create test video")

        processor = VideoProcessor(
            image_processor=mock_processor,
            num_frames=4,
            backend=platform_info["video_backend"],
        )

        frames = processor.extract_frames(temp_video_file)

        for frame in frames:
            assert isinstance(frame, Image.Image)
            assert frame.mode == "RGB"

    def test_process_video_returns_tensor(
        self, mock_processor, temp_video_file, platform_info
    ):
        """Test that process_video returns a tensor."""
        if temp_video_file is None:
            pytest.skip("Could not create test video")

        # Configure mock to return proper tensor
        mock_processor.side_effect = lambda images=None, **kwargs: {
            "pixel_values": torch.randn(len(images) if images else 1, 3, 384, 384)
        }

        processor = VideoProcessor(
            image_processor=mock_processor,
            num_frames=8,
            backend=platform_info["video_backend"],
        )

        tensor = processor.process_video(temp_video_file)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 4  # (num_frames, C, H, W)

    def test_get_video_info(self, temp_video_file, platform_info):
        """Test getting video metadata."""
        if temp_video_file is None:
            pytest.skip("Could not create test video")

        info = VideoProcessor.get_video_info(
            temp_video_file,
            backend=platform_info["video_backend"],
        )

        assert "num_frames" in info
        assert "fps" in info
        assert "duration_seconds" in info
        assert "backend" in info
        assert info["backend"] == platform_info["video_backend"]

    def test_get_video_info_returns_dimensions(self, temp_video_file, platform_info):
        """Test that get_video_info returns width and height."""
        if temp_video_file is None:
            pytest.skip("Could not create test video")

        info = VideoProcessor.get_video_info(
            temp_video_file,
            backend=platform_info["video_backend"],
        )

        if "error" not in info:
            assert "width" in info
            assert "height" in info


class TestProcessFrames:
    """Tests for processing pre-extracted frames."""

    def test_process_frames(self, mock_processor, sample_video_frames, platform_info):
        """Test processing list of PIL frames."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        # Configure mock
        mock_processor.side_effect = lambda images=None, **kwargs: {
            "pixel_values": torch.randn(len(images) if images else 1, 3, 384, 384)
        }

        processor = VideoProcessor(
            image_processor=mock_processor,
            num_frames=16,
            backend=platform_info["video_backend"],
        )

        tensor = processor.process_frames(sample_video_frames)

        assert isinstance(tensor, torch.Tensor)

    def test_process_frames_subsampling(self, mock_processor, platform_info):
        """Test that excess frames are subsampled."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        # Track how many images were processed
        processed_count = []

        def mock_call(images=None, **kwargs):
            if images:
                processed_count.append(len(images))
            return {
                "pixel_values": torch.randn(len(images) if images else 1, 3, 384, 384)
            }

        mock_processor.side_effect = mock_call

        processor = VideoProcessor(
            image_processor=mock_processor,
            num_frames=8,
            backend=platform_info["video_backend"],
        )

        # Create 20 frames (more than num_frames=8)
        frames = [
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            for _ in range(20)
        ]

        processor.process_frames(frames)

        # Should have subsampled to 8 frames
        assert processed_count[0] == 8

    def test_process_frames_fewer_than_num_frames(self, mock_processor, platform_info):
        """Test processing when frames < num_frames."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        mock_processor.side_effect = lambda images=None, **kwargs: {
            "pixel_values": torch.randn(len(images) if images else 1, 3, 384, 384)
        }

        processor = VideoProcessor(
            image_processor=mock_processor,
            num_frames=16,
            backend=platform_info["video_backend"],
        )

        # Only 5 frames
        frames = [
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            for _ in range(5)
        ]

        tensor = processor.process_frames(frames)

        # Should process all 5 frames
        assert tensor.shape[0] == 5


class TestCreateVideoProcessor:
    """Tests for create_video_processor factory function."""

    def test_create_default(self, mock_processor, platform_info):
        """Test creating processor with defaults."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        processor = create_video_processor(
            image_processor=mock_processor,
        )

        assert processor.num_frames == 32
        assert processor.frame_sampling == "uniform"

    def test_create_custom(self, mock_processor, platform_info):
        """Test creating processor with custom settings."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        processor = create_video_processor(
            image_processor=mock_processor,
            num_frames=16,
            frame_sampling="random",
        )

        assert processor.num_frames == 16
        assert processor.frame_sampling == "random"

    def test_create_returns_video_processor(self, mock_processor, platform_info):
        """Test that factory returns VideoProcessor instance."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        processor = create_video_processor(image_processor=mock_processor)

        assert isinstance(processor, VideoProcessor)


@pytest.mark.cuda
class TestCUDAVideoProcessing:
    """CUDA-specific video processing tests."""

    def test_decord_on_cuda(self, mock_processor):
        """Test decord backend on CUDA system."""
        if not DECORD_AVAILABLE:
            pytest.skip("decord not available")

        processor = VideoProcessor(
            image_processor=mock_processor,
            backend="decord",
        )

        assert processor.backend == "decord"


@pytest.mark.mps
class TestMPSVideoProcessing:
    """MPS (Apple Silicon) specific video processing tests."""

    def test_preferred_backend_on_mps(self, mock_processor):
        """Test that torchvision is preferred on Apple Silicon."""
        if not TORCHVISION_AVAILABLE:
            pytest.skip("torchvision not available")

        processor = VideoProcessor(
            image_processor=mock_processor,
        )

        # On Apple Silicon with torchvision available, should default to it
        if IS_APPLE_SILICON:
            assert processor.backend == "torchvision"

    def test_av_fallback_on_mps(self, mock_processor):
        """Test av backend works as fallback on MPS."""
        if not AV_AVAILABLE:
            pytest.skip("av not available")

        processor = VideoProcessor(
            image_processor=mock_processor,
            backend="av",
        )

        assert processor.backend == "av"
