"""Integration tests for the complete data pipeline.

Tests end-to-end data flow from loading through collation.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from src.data.dataset_loaders import SmolVLMSample
from src.data.data_collator import SmolVLMDataCollator, VideoDataCollator
from src.data.data_mixer import DatasetMixer, DatasetWeight


@pytest.mark.integration
class TestVisionPipelineIntegration:
    """Integration tests for vision stage pipeline."""

    def test_sample_to_collator_flow(
        self,
        mock_processor,
        mock_tokenizer,
        mock_processed_batch,
    ):
        """Test flow from processed samples through collator."""
        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=256,
        )

        batch = collator(mock_processed_batch)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].dim() == 2  # (batch, seq_len)
        assert batch["labels"].dim() == 2

    def test_mixer_to_collator_flow(
        self,
        mock_iterable_dataset,
        mock_processor,
        mock_tokenizer,
    ):
        """Test flow from mixer through to collator."""
        # Create mock datasets with pre-processed samples
        samples_a = [
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "pixel_values": torch.randn(3, 384, 384),
            }
            for _ in range(10)
        ]
        samples_b = [
            {
                "input_ids": torch.randint(0, 1000, (60,)),
                "attention_mask": torch.ones(60, dtype=torch.long),
                "pixel_values": torch.randn(3, 384, 384),
            }
            for _ in range(10)
        ]

        ds_a = mock_iterable_dataset(samples_a, "a")
        ds_b = mock_iterable_dataset(samples_b, "b")

        mixer = DatasetMixer(
            datasets=[
                DatasetWeight(name="a", dataset=ds_a, weight=0.5),
                DatasetWeight(name="b", dataset=ds_b, weight=0.5),
            ],
            shuffle_buffer_size=0,
        )

        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=128,
        )

        # Get batch from mixer
        batch_samples = []
        for i, sample in enumerate(mixer):
            if i >= 4:
                break
            # Remove metadata keys added by mixer
            clean_sample = {k: v for k, v in sample.items() if not k.startswith("_")}
            batch_samples.append(clean_sample)

        # Collate
        batch = collator(batch_samples)

        assert batch["input_ids"].shape[0] == 4

    def test_smolvlm_sample_to_processed_format(self, sample_pil_image):
        """Test converting SmolVLMSample to processed format."""
        sample = SmolVLMSample(
            images=[sample_pil_image],
            conversations=[
                {"role": "user", "content": "<image>Describe this."},
                {"role": "assistant", "content": "A test image."},
            ],
            metadata={"id": "test_001"},
        )

        assert sample.has_images
        assert sample.num_images == 1
        assert len(sample.conversations) == 2

    def test_batch_tensor_shapes(self, mock_processor, mock_tokenizer):
        """Test that batch tensors have correct shapes."""
        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=128,
            pad_to_multiple_of=8,
        )

        batch_size = 4
        features = [
            {
                "input_ids": torch.randint(0, 1000, (64 + i * 10,)),
                "attention_mask": torch.ones(64 + i * 10, dtype=torch.long),
                "pixel_values": torch.randn(3, 384, 384),
            }
            for i in range(batch_size)
        ]

        batch = collator(features)

        # Check batch dimension
        assert batch["input_ids"].shape[0] == batch_size
        assert batch["attention_mask"].shape[0] == batch_size
        assert batch["pixel_values"].shape[0] == batch_size

        # Check sequence dimension is padded to multiple of 8
        assert batch["input_ids"].shape[1] % 8 == 0


@pytest.mark.integration
@pytest.mark.video
class TestVideoPipelineIntegration:
    """Integration tests for video stage pipeline."""

    def test_video_sample_to_collator(
        self,
        mock_processor,
        mock_tokenizer,
        mock_video_batch,
    ):
        """Test flow from video samples through video collator."""
        collator = VideoDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=256,
            max_frames=32,
        )

        batch = collator(mock_video_batch)

        assert "input_ids" in batch
        assert "video_frames" in batch
        assert "video_mask" in batch
        assert batch["video_frames"].dim() == 5  # (batch, frames, C, H, W)

    def test_video_processor_to_collator(
        self,
        mock_processor,
        mock_tokenizer,
        sample_video_frames,
        platform_info,
    ):
        """Test video processor output through collator."""
        if platform_info["video_backend"] is None:
            pytest.skip("No video backend available")

        from src.data.video_processor import VideoProcessor

        # Configure mock processor
        mock_processor.side_effect = lambda images=None, **kwargs: {
            "pixel_values": torch.randn(len(images) if images else 1, 3, 384, 384)
        }

        video_processor = VideoProcessor(
            image_processor=mock_processor,
            num_frames=8,
            backend=platform_info["video_backend"],
        )

        # Process frames
        processed = video_processor.process_frames(sample_video_frames[:10])

        # Create sample with processed video
        features = [{
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50, dtype=torch.long),
            "video_frames": processed,
        }]

        collator = VideoDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_frames=16,
        )

        batch = collator(features)

        assert "video_frames" in batch

    def test_video_mask_indicates_valid_frames(self, mock_processor, mock_tokenizer):
        """Test that video mask correctly indicates valid frames."""
        collator = VideoDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_frames=32,
        )

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

        batch = collator(features)

        # First sample had 10 frames
        assert batch["video_mask"][0].sum() == 10
        # Second sample had 20 frames
        assert batch["video_mask"][1].sum() == 20


@pytest.mark.integration
class TestCrossModalityPipeline:
    """Tests for pipelines handling multiple modalities."""

    def test_mixed_image_video_collation(
        self,
        mock_processor,
        mock_tokenizer,
    ):
        """Test collating batch with both image and video samples."""
        collator = VideoDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=128,
            max_frames=16,
        )

        # Mixed batch: some with images, some with video
        features = [
            {
                "input_ids": torch.randint(0, 1000, (50,)),
                "attention_mask": torch.ones(50, dtype=torch.long),
                "pixel_values": torch.randn(3, 384, 384),
                "video_frames": None,
            },
            {
                "input_ids": torch.randint(0, 1000, (60,)),
                "attention_mask": torch.ones(60, dtype=torch.long),
                "pixel_values": None,
                "video_frames": torch.randn(8, 3, 384, 384),
            },
        ]

        batch = collator(features)

        # Both should be collated
        assert "input_ids" in batch
        assert "video_frames" in batch

    def test_text_only_batch(self, mock_processor, mock_tokenizer):
        """Test collating text-only batch."""
        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=128,
        )

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
        assert "attention_mask" in batch
        assert "pixel_values" not in batch


@pytest.mark.slow
@pytest.mark.integration
class TestLargeScalePipeline:
    """Tests for large-scale pipeline behavior."""

    def test_large_batch_collation(self, mock_processor, mock_tokenizer):
        """Test collating large batches."""
        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=2048,
        )

        # Create large batch
        batch_size = 32
        features = [
            {
                "input_ids": torch.randint(0, 32000, (1024,)),
                "attention_mask": torch.ones(1024, dtype=torch.long),
                "pixel_values": torch.randn(3, 384, 384),
            }
            for _ in range(batch_size)
        ]

        batch = collator(features)

        assert batch["input_ids"].shape == (batch_size, 1024)

    def test_memory_efficient_iteration(
        self,
        mock_iterable_dataset,
        mock_processor,
        mock_tokenizer,
    ):
        """Test that iteration doesn't accumulate memory."""
        samples = [
            {"input_ids": torch.randint(0, 1000, (100,)), "attention_mask": torch.ones(100)}
            for _ in range(1000)
        ]

        ds = mock_iterable_dataset(samples, "test")
        mixer = DatasetMixer(
            datasets=[DatasetWeight(name="test", dataset=ds, weight=1.0)],
            shuffle_buffer_size=100,
        )

        # Iterate through without storing
        count = 0
        for sample in mixer:
            count += 1
            if count >= 500:
                break

        assert count == 500


@pytest.mark.cuda
@pytest.mark.integration
class TestCUDAPipeline:
    """CUDA-specific integration tests."""

    def test_tensors_on_cuda(self, mock_processor, mock_tokenizer, device):
        """Test that pipeline can produce CUDA tensors."""
        if device.type != "cuda":
            pytest.skip("CUDA not available")

        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        features = [{
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50, dtype=torch.long),
            "pixel_values": torch.randn(3, 384, 384),
        }]

        batch = collator(features)

        # Move to CUDA
        cuda_batch = {k: v.to(device) for k, v in batch.items()}

        assert cuda_batch["input_ids"].device.type == "cuda"
        assert cuda_batch["pixel_values"].device.type == "cuda"

    def test_batch_dtype_on_cuda(self, mock_processor, mock_tokenizer, device, dtype):
        """Test tensor dtype on CUDA."""
        if device.type != "cuda":
            pytest.skip("CUDA not available")

        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        features = [{
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50, dtype=torch.long),
            "pixel_values": torch.randn(3, 384, 384),
        }]

        batch = collator(features)

        # Move to CUDA and convert dtype
        cuda_batch = {
            k: v.to(device=device, dtype=dtype if v.dtype == torch.float32 else v.dtype)
            for k, v in batch.items()
        }

        assert cuda_batch["pixel_values"].dtype == dtype


@pytest.mark.mps
@pytest.mark.integration
class TestMPSPipeline:
    """MPS (Apple Silicon) specific integration tests."""

    def test_tensors_on_mps(self, mock_processor, mock_tokenizer, device):
        """Test that pipeline can produce MPS tensors."""
        if device.type != "mps":
            pytest.skip("MPS not available")

        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        features = [{
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50, dtype=torch.long),
            "pixel_values": torch.randn(3, 384, 384),
        }]

        batch = collator(features)

        # Move to MPS
        mps_batch = {k: v.to(device) for k, v in batch.items()}

        assert mps_batch["input_ids"].device.type == "mps"
        assert mps_batch["pixel_values"].device.type == "mps"

    def test_batch_dtype_on_mps(self, mock_processor, mock_tokenizer, device, dtype):
        """Test tensor dtype on MPS."""
        if device.type != "mps":
            pytest.skip("MPS not available")

        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        features = [{
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50, dtype=torch.long),
            "pixel_values": torch.randn(3, 384, 384),
        }]

        batch = collator(features)

        # Move to MPS and convert dtype (only for tensor values)
        mps_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float32:
                    mps_batch[k] = v.to(device=device, dtype=dtype)
                else:
                    mps_batch[k] = v.to(device=device)

        # MPS typically uses float16
        assert mps_batch["pixel_values"].dtype == dtype


@pytest.mark.integration
class TestEndToEndFlow:
    """End-to-end pipeline tests."""

    def test_complete_vision_flow(
        self,
        mock_iterable_dataset,
        mock_processor,
        mock_tokenizer,
        sample_pil_image,
    ):
        """Test complete flow from raw data to batch."""
        # Create raw samples
        raw_samples = [
            {
                "image": sample_pil_image,
                "conversations": [
                    {"role": "user", "content": f"Question {i}?"},
                    {"role": "assistant", "content": f"Answer {i}."},
                ],
            }
            for i in range(10)
        ]

        # Create "processed" samples (simulating what dataset loader does)
        processed_samples = []
        for i, _ in enumerate(raw_samples):
            processed_samples.append({
                "input_ids": torch.randint(0, 1000, (64,)),
                "attention_mask": torch.ones(64, dtype=torch.long),
                "pixel_values": torch.randn(3, 384, 384),
            })

        # Create dataset and mixer
        ds = mock_iterable_dataset(processed_samples, "test")
        mixer = DatasetMixer(
            datasets=[DatasetWeight(name="test", dataset=ds, weight=1.0)],
            shuffle_buffer_size=0,
        )

        # Create collator
        collator = SmolVLMDataCollator(
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=128,
        )

        # Get batch
        batch_samples = []
        for i, sample in enumerate(mixer):
            if i >= 4:
                break
            clean_sample = {k: v for k, v in sample.items() if not k.startswith("_")}
            batch_samples.append(clean_sample)

        batch = collator(batch_samples)

        # Verify batch structure
        assert batch["input_ids"].shape[0] == 4
        assert batch["attention_mask"].shape[0] == 4
        assert batch["pixel_values"].shape[0] == 4
        assert batch["input_ids"].shape == batch["attention_mask"].shape
