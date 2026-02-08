"""Tests for data_mixer.py module.

Tests DatasetMixer, BalancedMixer, and factory functions.
"""

from collections import Counter

import pytest

from src.data.data_mixer import (
    BalancedMixer,
    DatasetMixer,
    DatasetWeight,
    create_video_stage_mixer,
    create_vision_stage_mixer,
)


class TestDatasetWeight:
    """Tests for DatasetWeight dataclass."""

    def test_creation(self, mock_iterable_dataset):
        """Test creating DatasetWeight."""
        ds = mock_iterable_dataset([{"id": 1}], "test")

        weight = DatasetWeight(
            name="test_dataset",
            dataset=ds,
            weight=0.5,
            modality="image",
        )

        assert weight.name == "test_dataset"
        assert weight.weight == 0.5
        assert weight.modality == "image"

    def test_default_modality(self, mock_iterable_dataset):
        """Test default modality is 'image'."""
        ds = mock_iterable_dataset([{"id": 1}], "test")

        weight = DatasetWeight(
            name="test",
            dataset=ds,
            weight=1.0,
        )

        assert weight.modality == "image"

    def test_video_modality(self, mock_iterable_dataset):
        """Test setting video modality."""
        ds = mock_iterable_dataset([{"id": 1}], "test")

        weight = DatasetWeight(
            name="video_test",
            dataset=ds,
            weight=0.3,
            modality="video",
        )

        assert weight.modality == "video"


class TestDatasetMixer:
    """Tests for DatasetMixer."""

    @pytest.fixture
    def sample_datasets(self, mock_iterable_dataset):
        """Create sample datasets for mixing."""
        ds_a = mock_iterable_dataset(
            [{"id": i, "source": "A"} for i in range(100)], "dataset_a"
        )
        ds_b = mock_iterable_dataset(
            [{"id": i, "source": "B"} for i in range(100)], "dataset_b"
        )
        return ds_a, ds_b

    @pytest.fixture
    def mixer(self, sample_datasets):
        """Create a DatasetMixer with sample datasets."""
        ds_a, ds_b = sample_datasets

        datasets = [
            DatasetWeight(name="a", dataset=ds_a, weight=0.7),
            DatasetWeight(name="b", dataset=ds_b, weight=0.3),
        ]

        return DatasetMixer(datasets=datasets, seed=42, shuffle_buffer_size=0)

    def test_init(self, sample_datasets):
        """Test DatasetMixer initialization."""
        ds_a, ds_b = sample_datasets

        datasets = [
            DatasetWeight(name="a", dataset=ds_a, weight=0.6),
            DatasetWeight(name="b", dataset=ds_b, weight=0.4),
        ]

        mixer = DatasetMixer(datasets=datasets, seed=42)

        # Check normalized probabilities
        assert abs(mixer.probabilities[0] - 0.6) < 0.001
        assert abs(mixer.probabilities[1] - 0.4) < 0.001

    def test_weight_normalization(self, sample_datasets):
        """Test that weights are normalized to sum to 1."""
        ds_a, ds_b = sample_datasets

        datasets = [
            DatasetWeight(name="a", dataset=ds_a, weight=2.0),
            DatasetWeight(name="b", dataset=ds_b, weight=8.0),
        ]

        mixer = DatasetMixer(datasets=datasets)

        assert abs(sum(mixer.probabilities) - 1.0) < 0.001
        assert abs(mixer.probabilities[0] - 0.2) < 0.001
        assert abs(mixer.probabilities[1] - 0.8) < 0.001

    def test_iteration(self, mixer):
        """Test iterating over mixed dataset."""
        samples = list(mixer)

        assert len(samples) == 200  # 100 from each dataset

    def test_dataset_info_added(self, mixer):
        """Test that _dataset_name and _modality are added to samples."""
        sample = next(iter(mixer))

        assert "_dataset_name" in sample
        assert "_modality" in sample
        assert sample["_dataset_name"] in ["a", "b"]

    def test_all_samples_consumed(self, sample_datasets):
        """Test that all samples from all datasets are consumed."""
        ds_a, ds_b = sample_datasets

        datasets = [
            DatasetWeight(name="a", dataset=ds_a, weight=0.5),
            DatasetWeight(name="b", dataset=ds_b, weight=0.5),
        ]

        mixer = DatasetMixer(datasets=datasets, seed=42, shuffle_buffer_size=0)
        samples = list(mixer)

        # Should get all 200 samples
        assert len(samples) == 200

        # Count by source
        counts = Counter(s["_dataset_name"] for s in samples)
        assert counts["a"] == 100
        assert counts["b"] == 100

    def test_reproducibility(self, mock_iterable_dataset):
        """Test that same seed produces same sequence."""

        def create_mixer():
            ds_a = mock_iterable_dataset(
                [{"id": i, "source": "A"} for i in range(50)], "a"
            )
            ds_b = mock_iterable_dataset(
                [{"id": i, "source": "B"} for i in range(50)], "b"
            )

            datasets = [
                DatasetWeight(name="a", dataset=ds_a, weight=0.5),
                DatasetWeight(name="b", dataset=ds_b, weight=0.5),
            ]
            return DatasetMixer(datasets=datasets, seed=42, shuffle_buffer_size=0)

        mixer1 = create_mixer()
        mixer2 = create_mixer()

        samples1 = [s["_dataset_name"] for s in mixer1]
        samples2 = [s["_dataset_name"] for s in mixer2]

        assert samples1 == samples2

    def test_shuffle_buffer(self, mock_iterable_dataset):
        """Test that shuffle buffer affects ordering."""

        def create_mixer(buffer_size):
            ds_a = mock_iterable_dataset(
                [{"id": i, "source": "A"} for i in range(20)], "a"
            )
            ds_b = mock_iterable_dataset(
                [{"id": i, "source": "B"} for i in range(20)], "b"
            )

            datasets = [
                DatasetWeight(name="a", dataset=ds_a, weight=0.5),
                DatasetWeight(name="b", dataset=ds_b, weight=0.5),
            ]
            return DatasetMixer(
                datasets=datasets, seed=42, shuffle_buffer_size=buffer_size
            )

        mixer_no_buffer = create_mixer(0)
        mixer_with_buffer = create_mixer(50)

        samples_no = list(mixer_no_buffer)
        samples_with = list(mixer_with_buffer)

        # Same total samples
        assert len(samples_no) == len(samples_with) == 40

    def test_single_dataset(self, mock_iterable_dataset):
        """Test mixer with single dataset."""
        ds = mock_iterable_dataset([{"id": i} for i in range(50)], "only")

        datasets = [DatasetWeight(name="only", dataset=ds, weight=1.0)]
        mixer = DatasetMixer(datasets=datasets, shuffle_buffer_size=0)

        samples = list(mixer)
        assert len(samples) == 50
        assert all(s["_dataset_name"] == "only" for s in samples)

    def test_empty_dataset(self, mock_iterable_dataset):
        """Test mixer when one dataset is empty."""
        ds_a = mock_iterable_dataset([], "empty")
        ds_b = mock_iterable_dataset([{"id": i} for i in range(50)], "full")

        datasets = [
            DatasetWeight(name="empty", dataset=ds_a, weight=0.5),
            DatasetWeight(name="full", dataset=ds_b, weight=0.5),
        ]

        mixer = DatasetMixer(datasets=datasets, shuffle_buffer_size=0)
        samples = list(mixer)

        # Should only get samples from non-empty dataset
        assert len(samples) == 50
        assert all(s["_dataset_name"] == "full" for s in samples)


class TestBalancedMixer:
    """Tests for BalancedMixer."""

    @pytest.fixture
    def balanced_datasets(self, mock_iterable_dataset):
        """Create datasets with different modalities."""
        image_ds = mock_iterable_dataset(
            [{"id": i, "type": "image"} for i in range(50)], "image_ds"
        )
        video_ds = mock_iterable_dataset(
            [{"id": i, "type": "video"} for i in range(50)], "video_ds"
        )
        text_ds = mock_iterable_dataset(
            [{"id": i, "type": "text"} for i in range(50)], "text_ds"
        )

        return [
            DatasetWeight(
                name="images", dataset=image_ds, weight=0.5, modality="image"
            ),
            DatasetWeight(
                name="videos", dataset=video_ds, weight=0.3, modality="video"
            ),
            DatasetWeight(name="texts", dataset=text_ds, weight=0.2, modality="text"),
        ]

    def test_init(self, balanced_datasets):
        """Test BalancedMixer initialization."""
        mixer = BalancedMixer(datasets=balanced_datasets)

        assert "image" in mixer.modality_datasets
        assert "video" in mixer.modality_datasets
        assert "text" in mixer.modality_datasets

    def test_default_modality_weights(self, balanced_datasets):
        """Test default modality weights from paper."""
        mixer = BalancedMixer(datasets=balanced_datasets)

        assert mixer.modality_weights["image"] == 0.344
        assert mixer.modality_weights["video"] == 0.330
        assert mixer.modality_weights["text"] == 0.202
        assert mixer.modality_weights["multi-image"] == 0.123

    def test_custom_modality_weights(self, balanced_datasets):
        """Test custom modality weights."""
        custom_weights = {"image": 0.5, "video": 0.3, "text": 0.2}

        mixer = BalancedMixer(
            datasets=balanced_datasets,
            modality_weights=custom_weights,
        )

        assert mixer.modality_weights == custom_weights

    def test_iteration(self, balanced_datasets):
        """Test iterating over balanced mixer."""
        mixer = BalancedMixer(datasets=balanced_datasets, seed=42)

        samples = list(mixer)

        # Should get all samples from all modalities
        assert len(samples) == 150

    def test_modality_grouping(self, balanced_datasets):
        """Test that datasets are properly grouped by modality."""
        mixer = BalancedMixer(datasets=balanced_datasets)

        assert len(mixer.modality_datasets["image"]) == 1
        assert len(mixer.modality_datasets["video"]) == 1
        assert len(mixer.modality_datasets["text"]) == 1

    def test_multiple_datasets_per_modality(self, mock_iterable_dataset):
        """Test mixing with multiple datasets per modality."""
        datasets = [
            DatasetWeight(
                name="img1",
                dataset=mock_iterable_dataset([{"id": i} for i in range(25)], "img1"),
                weight=0.3,
                modality="image",
            ),
            DatasetWeight(
                name="img2",
                dataset=mock_iterable_dataset([{"id": i} for i in range(25)], "img2"),
                weight=0.3,
                modality="image",
            ),
            DatasetWeight(
                name="vid1",
                dataset=mock_iterable_dataset([{"id": i} for i in range(50)], "vid1"),
                weight=0.4,
                modality="video",
            ),
        ]

        mixer = BalancedMixer(datasets=datasets, seed=42)

        # Image modality should have 2 datasets
        assert len(mixer.modality_datasets["image"]) == 2
        assert len(mixer.modality_datasets["video"]) == 1

        samples = list(mixer)
        assert len(samples) == 100  # 25 + 25 + 50


class TestVisionStageMixer:
    """Tests for create_vision_stage_mixer factory."""

    def test_creation(self, mock_iterable_dataset):
        """Test creating vision stage mixer."""
        cauldron_ds = mock_iterable_dataset([{"id": i} for i in range(10)], "cauldron")
        docmatix_ds = mock_iterable_dataset([{"id": i} for i in range(10)], "docmatix")

        mixer = create_vision_stage_mixer(cauldron_ds, docmatix_ds, seed=42)

        assert isinstance(mixer, DatasetMixer)
        assert len(mixer.datasets) == 2

    def test_weights(self, mock_iterable_dataset):
        """Test that vision stage mixer has correct weights."""
        cauldron_ds = mock_iterable_dataset([{"id": i} for i in range(10)], "cauldron")
        docmatix_ds = mock_iterable_dataset([{"id": i} for i in range(10)], "docmatix")

        mixer = create_vision_stage_mixer(cauldron_ds, docmatix_ds)

        cauldron_weight = next(d for d in mixer.datasets if d.name == "the_cauldron")
        docmatix_weight = next(d for d in mixer.datasets if d.name == "docmatix")

        assert cauldron_weight.weight == 0.35
        assert docmatix_weight.weight == 0.41

    def test_modalities_are_image(self, mock_iterable_dataset):
        """Test that vision stage datasets have image modality."""
        cauldron_ds = mock_iterable_dataset([{"id": i} for i in range(10)], "cauldron")
        docmatix_ds = mock_iterable_dataset([{"id": i} for i in range(10)], "docmatix")

        mixer = create_vision_stage_mixer(cauldron_ds, docmatix_ds)

        for ds in mixer.datasets:
            assert ds.modality == "image"


class TestVideoStageMixer:
    """Tests for create_video_stage_mixer factory."""

    def test_creation(self, mock_iterable_dataset):
        """Test creating video stage mixer."""
        datasets = {
            "llava_onevision": mock_iterable_dataset(
                [{"id": i} for i in range(10)], "llava"
            ),
            "llava_video": mock_iterable_dataset(
                [{"id": i} for i in range(10)], "video"
            ),
        }

        mixer = create_video_stage_mixer(datasets, seed=42)

        assert isinstance(mixer, BalancedMixer)

    def test_modality_assignment(self, mock_iterable_dataset):
        """Test that datasets get correct modality assignments."""
        datasets = {
            "llava_onevision": mock_iterable_dataset(
                [{"id": i} for i in range(10)], "llava"
            ),
            "llava_video": mock_iterable_dataset(
                [{"id": i} for i in range(10)], "video"
            ),
            "m4_instruct": mock_iterable_dataset([{"id": i} for i in range(10)], "m4"),
        }

        mixer = create_video_stage_mixer(datasets)

        # Check modality assignments
        for ds in mixer.datasets:
            if ds.name == "llava_onevision":
                assert ds.modality == "image"
            elif ds.name == "llava_video":
                assert ds.modality == "video"
            elif ds.name == "m4_instruct":
                assert ds.modality == "multi-image"

    def test_weight_assignment(self, mock_iterable_dataset):
        """Test that datasets get correct weight assignments."""
        datasets = {
            "llava_onevision": mock_iterable_dataset(
                [{"id": i} for i in range(10)], "llava"
            ),
            "llava_video": mock_iterable_dataset(
                [{"id": i} for i in range(10)], "video"
            ),
            "video_star": mock_iterable_dataset([{"id": i} for i in range(10)], "star"),
        }

        mixer = create_video_stage_mixer(datasets)

        weight_map = {ds.name: ds.weight for ds in mixer.datasets}

        assert weight_map["llava_onevision"] == 0.15
        assert weight_map["llava_video"] == 0.08
        assert weight_map["video_star"] == 0.05

    def test_unknown_dataset_ignored(self, mock_iterable_dataset):
        """Test that unknown dataset names are ignored."""
        datasets = {
            "llava_onevision": mock_iterable_dataset(
                [{"id": i} for i in range(10)], "llava"
            ),
            "unknown_dataset": mock_iterable_dataset(
                [{"id": i} for i in range(10)], "unknown"
            ),
        }

        mixer = create_video_stage_mixer(datasets)

        # Only known dataset should be included
        assert len(mixer.datasets) == 1
        assert mixer.datasets[0].name == "llava_onevision"
