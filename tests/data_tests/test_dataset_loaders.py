"""Tests for dataset_loaders.py module.

Tests SmolVLMSample, VisionStageDataset, VideoStageDataset and
loader functions.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.data.dataset_loaders import (
    SmolVLMSample,
    VideoStageDataset,
    VisionStageDataset,
    create_vision_stage_dataloader,
)


class TestSmolVLMSample:
    """Tests for SmolVLMSample dataclass."""

    def test_init_empty(self):
        """Test creating empty sample."""
        sample = SmolVLMSample()
        assert sample.images == []
        assert sample.video_frames == []
        assert sample.conversations == []
        assert sample.metadata == {}

    def test_init_with_images(self, sample_pil_images):
        """Test creating sample with images."""
        sample = SmolVLMSample(images=sample_pil_images)
        assert len(sample.images) == len(sample_pil_images)
        assert sample.has_images is True
        assert sample.has_video is False
        assert sample.num_images == len(sample_pil_images)
        assert sample.num_frames == 0

    def test_init_with_video_frames(self, sample_video_frames):
        """Test creating sample with video frames."""
        sample = SmolVLMSample(video_frames=sample_video_frames)
        assert len(sample.video_frames) == len(sample_video_frames)
        assert sample.has_images is False
        assert sample.has_video is True
        assert sample.num_images == 0
        assert sample.num_frames == len(sample_video_frames)

    def test_init_with_conversations(self):
        """Test creating sample with conversations."""
        convs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        sample = SmolVLMSample(conversations=convs)
        assert sample.conversations == convs

    def test_init_with_metadata(self):
        """Test creating sample with metadata."""
        meta = {"source": "test", "id": 123}
        sample = SmolVLMSample(metadata=meta)
        assert sample.metadata == meta

    def test_full_sample(self, sample_pil_images, sample_video_frames):
        """Test creating full multimodal sample."""
        convs = [{"role": "user", "content": "Describe"}]
        meta = {"source": "test"}

        sample = SmolVLMSample(
            images=sample_pil_images,
            video_frames=sample_video_frames,
            conversations=convs,
            metadata=meta,
        )

        assert sample.has_images is True
        assert sample.has_video is True
        assert sample.num_images == len(sample_pil_images)
        assert sample.num_frames == len(sample_video_frames)

    def test_properties_with_none_lists(self):
        """Test properties handle None gracefully."""
        sample = SmolVLMSample(images=None, video_frames=None)
        assert sample.has_images is False
        assert sample.has_video is False
        assert sample.num_images == 0
        assert sample.num_frames == 0


class TestVisionStageDataset:
    """Tests for VisionStageDataset."""

    def test_init_default_weights(self, mock_processor, mock_tokenizer):
        """Test initialization with default weights."""
        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            dataset = VisionStageDataset(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
            )

            assert dataset.weights["the_cauldron"] == 0.35
            assert dataset.weights["docmatix"] == 0.41

    def test_init_custom_weights(self, mock_processor, mock_tokenizer):
        """Test initialization with custom weights."""
        custom_weights = {"the_cauldron": 0.5, "docmatix": 0.5}

        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            dataset = VisionStageDataset(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
                weights=custom_weights,
            )

            assert dataset.weights == custom_weights

    def test_init_parameters(self, mock_processor, mock_tokenizer):
        """Test initialization parameters are stored correctly."""
        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            dataset = VisionStageDataset(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
                max_length=1024,
                image_size=512,
                streaming=False,
            )

            assert dataset.max_length == 1024
            assert dataset.image_size == 512
            assert dataset.streaming is False

    def test_process_sample_with_image(
        self, mock_processor, mock_tokenizer, sample_pil_image
    ):
        """Test processing a sample with image."""
        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            # Configure mock processor to return proper tensor
            mock_processor.side_effect = lambda images=None, **kwargs: {
                "pixel_values": torch.randn(1, 3, 384, 384) if images else None
            }

            # Configure mock tokenizer
            mock_tokenizer.side_effect = lambda text, **kwargs: {
                "input_ids": torch.randint(0, 1000, (1, 128)),
                "attention_mask": torch.ones(1, 128),
            }

            dataset = VisionStageDataset(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
                max_length=128,
            )

            raw_sample = {
                "image": sample_pil_image,
                "conversations": [
                    {"role": "user", "content": "Describe this."},
                    {"role": "assistant", "content": "A test image."},
                ],
            }

            result = dataset._process_sample(raw_sample, "the_cauldron")

            assert result is not None
            assert "input_ids" in result
            assert "attention_mask" in result
            assert "pixel_values" in result
            assert "labels" in result

    def test_process_sample_without_image(self, mock_processor, mock_tokenizer):
        """Test processing a sample without image returns None."""
        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            dataset = VisionStageDataset(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
            )

            # Sample with no image
            sample = {"conversations": [{"role": "user", "content": "Hello"}]}
            result = dataset._process_sample(sample, "test")

            assert result is None

    def test_process_sample_bytes_image(self, mock_processor, mock_tokenizer):
        """Test processing a sample with bytes image."""
        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            # Configure mocks
            mock_processor.side_effect = lambda images=None, **kwargs: {
                "pixel_values": torch.randn(1, 3, 384, 384) if images else None
            }
            mock_tokenizer.side_effect = lambda text, **kwargs: {
                "input_ids": torch.randint(0, 1000, (1, 128)),
                "attention_mask": torch.ones(1, 128),
            }

            dataset = VisionStageDataset(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
                max_length=128,
            )

            # Create bytes image
            img = Image.new("RGB", (100, 100), color="red")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

            sample = {
                "image": img_bytes,
                "question": "What color?",
                "answer": "Red",
            }

            result = dataset._process_sample(sample, "test")
            assert result is not None

    def test_process_sample_docmatix_format(
        self, mock_processor, mock_tokenizer, sample_pil_image
    ):
        """Test processing a sample in Docmatix format (images list, messages)."""
        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            mock_processor.side_effect = lambda images=None, **kwargs: {
                "pixel_values": torch.randn(1, 3, 384, 384) if images else None
            }
            mock_tokenizer.side_effect = lambda text, **kwargs: {
                "input_ids": torch.randint(0, 1000, (1, 128)),
                "attention_mask": torch.ones(1, 128),
            }

            dataset = VisionStageDataset(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
                max_length=128,
            )

            sample = {
                "images": [sample_pil_image],
                "messages": [
                    {"role": "user", "content": "Read this doc."},
                    {"role": "assistant", "content": "Document content."},
                ],
            }

            result = dataset._process_sample(sample, "docmatix")
            assert result is not None

    def test_process_sample_handles_exception(self, mock_processor, mock_tokenizer):
        """Test that _process_sample handles exceptions gracefully."""
        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            # Make processor raise an exception
            mock_processor.side_effect = Exception("Processing failed")

            dataset = VisionStageDataset(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
            )

            sample = {
                "image": Image.new("RGB", (10, 10)),
                "conversations": [{"role": "user", "content": "Test"}],
            }

            result = dataset._process_sample(sample, "test")
            assert result is None


class TestVideoStageDataset:
    """Tests for VideoStageDataset."""

    def test_init(self, mock_processor, mock_tokenizer):
        """Test VideoStageDataset initialization."""
        video_processor = MagicMock()

        dataset = VideoStageDataset(
            processor=mock_processor,
            tokenizer=mock_tokenizer,
            video_processor=video_processor,
            max_length=4096,
            max_frames=32,
        )

        assert dataset.max_length == 4096
        assert dataset.max_frames == 32

    def test_init_default_weights(self, mock_processor, mock_tokenizer):
        """Test default weights are set correctly."""
        video_processor = MagicMock()

        dataset = VideoStageDataset(
            processor=mock_processor,
            tokenizer=mock_tokenizer,
            video_processor=video_processor,
        )

        assert dataset.weights["llava_onevision"] == 0.15
        assert dataset.weights["llava_video"] == 0.08
        assert dataset.weights["video_star"] == 0.05

    def test_iter_not_implemented(self, mock_processor, mock_tokenizer):
        """Test that iteration raises NotImplementedError."""
        video_processor = MagicMock()

        dataset = VideoStageDataset(
            processor=mock_processor,
            tokenizer=mock_tokenizer,
            video_processor=video_processor,
        )

        with pytest.raises(NotImplementedError):
            next(iter(dataset))


class TestDataLoaderCreation:
    """Tests for dataloader factory functions."""

    def test_create_vision_stage_dataloader(self, mock_processor, mock_tokenizer):
        """Test creating vision stage dataloader."""
        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            dataloader = create_vision_stage_dataloader(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
                batch_size=4,
                num_workers=0,
            )

            assert dataloader is not None
            assert dataloader.batch_size == 4

    def test_create_vision_stage_dataloader_with_kwargs(
        self, mock_processor, mock_tokenizer
    ):
        """Test creating dataloader with additional kwargs."""
        with (
            patch("src.data.dataset_loaders.load_the_cauldron") as mock_cauldron,
            patch("src.data.dataset_loaders.load_docmatix") as mock_docmatix,
        ):
            mock_cauldron.return_value = iter([])
            mock_docmatix.return_value = iter([])

            dataloader = create_vision_stage_dataloader(
                processor=mock_processor,
                tokenizer=mock_tokenizer,
                batch_size=8,
                num_workers=0,
                max_length=1024,
                streaming=True,
            )

            assert dataloader.batch_size == 8


@pytest.mark.download_data
class TestRealDatasetLoading:
    """Tests that download data from HuggingFace.

    These tests verify that the actual HuggingFace dataset APIs work.
    Skip with: pytest -m "not download_data"

    Note: If you get "torch_shm_manager: Permission denied", run:
        chmod +x ~/.pyenv/versions/*/lib/python*/site-packages/torch/bin/torch_shm_manager
    """

    def test_load_the_cauldron(self):
        """Test loading The Cauldron dataset from Huggingface."""
        from src.data.dataset_loaders import load_the_cauldron

        # The Cauldron requires a subset name (e.g., 'ai2d')
        ds = load_the_cauldron(split="train[:10]", streaming=False, subset="ai2d")

        assert len(ds) == 10
        for sample in ds:
            print(sample.keys())
            assert "image" in sample or "images" in sample

    def test_load_docmatix(self):
        """Test loading Docmatix dataset from Huggingface.

        Note: Dataset is too large for streaming=False. When streaming=True, the test
        passes but teardown throws an error '[Errno 9] Bad file descriptor'
        """

        from src.data.dataset_loaders import load_docmatix

        ds = load_docmatix(split="train", streaming=True)
        samples = ds.take(10)
        for sample in samples:
            print(sample.keys())
            assert sample is not None
