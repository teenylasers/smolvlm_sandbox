"""Tests for data_collator.py module.

Tests SmolVLMDataCollator and VideoDataCollator.
"""

import pytest
import torch

from src.data.data_collator import (
    SmolVLMDataCollator,
    VideoDataCollator,
    create_data_collator,
)


class TestSmolVLMDataCollator:
    """Tests for SmolVLMDataCollator."""

    @pytest.fixture
    def collator(self, mock_tokenizer, mock_processor):
        """Create collator for testing."""
        return SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=256,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )

    def test_init(self, mock_tokenizer, mock_processor):
        """Test collator initialization."""
        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=1024,
        )

        assert collator.max_length == 1024
        assert collator.label_pad_token_id == -100
        assert collator.pad_to_multiple_of == 8

    def test_init_default_values(self, mock_tokenizer, mock_processor):
        """Test collator default initialization values."""
        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        assert collator.max_length == 2048
        assert collator.padding == "max_length"
        assert collator.return_tensors == "pt"

    def test_collate_single_sample(self, collator):
        """Test collating a single sample."""
        features = [{
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50, dtype=torch.long),
            "labels": torch.randint(0, 1000, (50,)),
            "pixel_values": torch.randn(3, 384, 384),
        }]

        batch = collator(features)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "pixel_values" in batch
        assert batch["input_ids"].shape[0] == 1

    def test_collate_multiple_samples(self, collator, mock_processed_batch):
        """Test collating multiple samples."""
        batch = collator(mock_processed_batch)

        assert batch["input_ids"].shape[0] == len(mock_processed_batch)
        assert batch["attention_mask"].shape[0] == len(mock_processed_batch)
        assert batch["labels"].shape[0] == len(mock_processed_batch)

    def test_collate_variable_lengths(self, collator):
        """Test collating samples with variable sequence lengths."""
        features = [
            {
                "input_ids": torch.randint(0, 1000, (30,)),
                "attention_mask": torch.ones(30, dtype=torch.long),
            },
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
            },
            {
                "input_ids": torch.randint(0, 1000, (40,)),
                "attention_mask": torch.ones(40, dtype=torch.long),
            },
        ]

        batch = collator(features)

        # All should have same length after padding
        assert batch["input_ids"].shape[1] == batch["attention_mask"].shape[1]
        # Shortest sample should have been padded
        assert batch["input_ids"].shape[1] >= 50

    def test_padding_to_multiple(self, collator):
        """Test that sequences are padded to multiple of pad_to_multiple_of."""
        features = [{
            "input_ids": torch.randint(0, 1000, (35,)),
            "attention_mask": torch.ones(35, dtype=torch.long),
        }]

        batch = collator(features)

        # Should be padded to 40 (next multiple of 8)
        assert batch["input_ids"].shape[1] % 8 == 0
        assert batch["input_ids"].shape[1] >= 35

    def test_label_padding(self, collator):
        """Test that labels are padded with -100."""
        features = [
            {
                "input_ids": torch.randint(0, 1000, (20,)),
                "attention_mask": torch.ones(20, dtype=torch.long),
                "labels": torch.randint(0, 1000, (20,)),
            },
            {
                "input_ids": torch.randint(0, 1000, (30,)),
                "attention_mask": torch.ones(30, dtype=torch.long),
                "labels": torch.randint(0, 1000, (30,)),
            },
        ]

        batch = collator(features)

        # First sample's labels should have -100 padding at the end
        padded_len = batch["labels"].shape[1]
        # The positions beyond original length should be -100
        assert (batch["labels"][0, 20:padded_len] == -100).all()

    def test_truncation(self, mock_tokenizer, mock_processor):
        """Test truncation to max_length."""
        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=64,
        )

        features = [{
            "input_ids": torch.randint(0, 1000, (100,)),
            "attention_mask": torch.ones(100, dtype=torch.long),
        }]

        batch = collator(features)

        assert batch["input_ids"].shape[1] <= 64

    def test_collate_without_images(self, collator):
        """Test collating samples without pixel_values."""
        features = [
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
            },
            {
                "input_ids": torch.randint(0, 1000, (60,)),
                "attention_mask": torch.ones(60, dtype=torch.long),
            },
        ]

        batch = collator(features)

        assert "input_ids" in batch
        assert "pixel_values" not in batch

    def test_collate_with_all_images(self, collator):
        """Test collating when all samples have images."""
        features = [
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "pixel_values": torch.randn(3, 384, 384),
            },
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "pixel_values": torch.randn(3, 384, 384),
            },
        ]

        batch = collator(features)

        assert "pixel_values" in batch
        assert batch["pixel_values"].shape[0] == 2

    def test_collate_mixed_image_availability(self, collator):
        """Test collating when only some samples have images."""
        features = [
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "pixel_values": torch.randn(3, 384, 384),
            },
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "pixel_values": None,
            },
        ]

        batch = collator(features)

        # pixel_values should be removed if not all samples have them
        assert "pixel_values" not in batch

    def test_labels_from_input_ids(self, collator):
        """Test that labels are cloned from input_ids if not provided."""
        input_ids = torch.randint(0, 1000, (50,))
        features = [{
            "input_ids": input_ids,
            "attention_mask": torch.ones(50, dtype=torch.long),
        }]

        batch = collator(features)

        assert "labels" in batch
        # After padding, compare original portion
        assert torch.equal(batch["labels"][0, :50], input_ids)

    def test_pad_sequences_helper(self, collator):
        """Test _pad_sequences helper method."""
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6, 7, 8]),
            torch.tensor([9, 10]),
        ]

        padded = collator._pad_sequences(sequences, pad_value=0)

        # All should have same length
        assert padded.shape[0] == 3
        assert padded.shape[1] % 8 == 0  # Padded to multiple of 8

    def test_pad_images_3d(self, collator):
        """Test _pad_images with 3D tensors (C, H, W)."""
        images = [
            torch.randn(3, 384, 384),
            torch.randn(3, 256, 256),
        ]

        padded = collator._pad_images(images)

        assert padded.shape == (2, 3, 384, 384)

    def test_pad_images_4d(self, collator):
        """Test _pad_images with 4D tensors (N, C, H, W)."""
        images = [
            torch.randn(10, 3, 384, 384),
            torch.randn(5, 3, 256, 256),
        ]

        padded = collator._pad_images(images)

        assert padded.shape == (2, 10, 3, 384, 384)

    def test_pad_images_same_size(self, collator):
        """Test _pad_images when all images have same size."""
        images = [
            torch.randn(3, 384, 384),
            torch.randn(3, 384, 384),
            torch.randn(3, 384, 384),
        ]

        padded = collator._pad_images(images)

        assert padded.shape == (3, 3, 384, 384)


class TestVideoDataCollator:
    """Tests for VideoDataCollator."""

    @pytest.fixture
    def video_collator(self, mock_tokenizer, mock_processor):
        """Create video collator for testing."""
        return VideoDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=256,
            max_frames=32,
        )

    def test_init(self, mock_tokenizer, mock_processor):
        """Test video collator initialization."""
        collator = VideoDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_frames=16,
        )

        assert collator.max_frames == 16

    def test_inherits_from_base(self, video_collator):
        """Test that VideoDataCollator inherits from SmolVLMDataCollator."""
        assert isinstance(video_collator, SmolVLMDataCollator)

    def test_collate_with_video_frames(self, video_collator, mock_video_batch):
        """Test collating batch with video frames."""
        batch = video_collator(mock_video_batch)

        assert "video_frames" in batch
        assert "video_mask" in batch
        assert batch["video_frames"].shape[0] == len(mock_video_batch)

    def test_video_frame_padding(self, video_collator):
        """Test that video frames are padded to same length."""
        features = [
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "video_frames": torch.randn(10, 3, 384, 384),
            },
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "video_frames": torch.randn(20, 3, 384, 384),
            },
        ]

        batch = video_collator(features)

        # All samples should have same number of frames
        assert batch["video_frames"].shape[1] == 20
        # Video mask should indicate actual frames
        assert batch["video_mask"][0].sum() == 10  # First sample had 10 frames
        assert batch["video_mask"][1].sum() == 20  # Second sample had 20 frames

    def test_video_frame_truncation(self, video_collator):
        """Test that video frames exceeding max_frames are subsampled."""
        features = [{
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50, dtype=torch.long),
            "video_frames": torch.randn(64, 3, 384, 384),  # More than max_frames=32
        }]

        batch = video_collator(features)

        assert batch["video_frames"].shape[1] <= video_collator.max_frames

    def test_collate_mixed_video_availability(self, video_collator):
        """Test collating when only some samples have video."""
        features = [
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "video_frames": torch.randn(16, 3, 384, 384),
            },
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "video_frames": None,
            },
        ]

        batch = video_collator(features)

        # Should handle None video frames with zero tensors
        assert "video_frames" in batch
        assert batch["video_mask"][1].sum() == 0  # Second sample has no valid frames

    def test_video_mask_dtype(self, video_collator):
        """Test that video_mask has boolean dtype."""
        features = [{
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50, dtype=torch.long),
            "video_frames": torch.randn(16, 3, 384, 384),
        }]

        batch = video_collator(features)

        assert batch["video_mask"].dtype == torch.bool

    def test_video_collator_inherits_text_handling(self, video_collator):
        """Test that video collator properly handles text like base class."""
        features = [
            {
                "input_ids": torch.randint(0, 1000, (30,)),
                "attention_mask": torch.ones(30, dtype=torch.long),
                "labels": torch.randint(0, 1000, (30,)),
                "video_frames": torch.randn(8, 3, 384, 384),
            },
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "labels": torch.randint(0, 1000, (50,)),
                "video_frames": torch.randn(8, 3, 384, 384),
            },
        ]

        batch = video_collator(features)

        # Text should be properly padded
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[1] >= 50
        assert batch["labels"].shape == batch["input_ids"].shape


class TestCreateDataCollator:
    """Tests for create_data_collator factory function."""

    def test_create_vision_collator(self, mock_tokenizer, mock_processor):
        """Test creating vision collator."""
        collator = create_data_collator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            is_video=False,
        )

        assert isinstance(collator, SmolVLMDataCollator)
        assert not isinstance(collator, VideoDataCollator)

    def test_create_video_collator(self, mock_tokenizer, mock_processor):
        """Test creating video collator."""
        collator = create_data_collator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            is_video=True,
        )

        assert isinstance(collator, VideoDataCollator)

    def test_custom_max_length(self, mock_tokenizer, mock_processor):
        """Test creating collator with custom max_length."""
        collator = create_data_collator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=4096,
        )

        assert collator.max_length == 4096

    def test_passes_kwargs(self, mock_tokenizer, mock_processor):
        """Test that factory passes additional kwargs."""
        collator = create_data_collator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=1024,
            pad_to_multiple_of=16,
            is_video=True,
        )

        assert collator.max_length == 1024
        assert collator.pad_to_multiple_of == 16
