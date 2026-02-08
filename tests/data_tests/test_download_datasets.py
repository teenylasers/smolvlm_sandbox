"""Tests for download_datasets.py module.

Tests DatasetConfig and download utilities.
"""

import json
from unittest.mock import MagicMock, patch

from src.data.download_datasets import (
    VIDEO_STAGE_DATASETS,
    VISION_STAGE_DATASETS,
    DatasetConfig,
    download_all,
    download_dataset,
    get_all_datasets,
    preview_datasets,
)


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_minimal_config(self):
        """Test creating minimal config."""
        config = DatasetConfig(
            name="test",
            hf_path="test/dataset",
        )

        assert config.name == "test"
        assert config.hf_path == "test/dataset"
        assert config.split == "train"
        assert config.streaming is True
        assert config.modality == "image"
        assert config.stage == "vision"

    def test_full_config(self):
        """Test creating full config."""
        config = DatasetConfig(
            name="custom",
            hf_path="org/dataset",
            hf_name="subset",
            split="validation",
            streaming=False,
            modality="video",
            stage="video",
            mix_weight=0.5,
            max_samples=1000,
        )

        assert config.name == "custom"
        assert config.hf_name == "subset"
        assert config.split == "validation"
        assert config.streaming is False
        assert config.modality == "video"
        assert config.stage == "video"
        assert config.mix_weight == 0.5
        assert config.max_samples == 1000

    def test_default_values(self):
        """Test all default values."""
        config = DatasetConfig(name="test", hf_path="test/path")

        assert config.hf_name is None
        assert config.split == "train"
        assert config.streaming is True
        assert config.modality == "image"
        assert config.stage == "vision"
        assert config.mix_weight == 1.0
        assert config.max_samples is None


class TestPresetDatasets:
    """Tests for preset dataset configurations."""

    def test_vision_stage_datasets_exist(self):
        """Test that vision stage datasets are defined."""
        assert len(VISION_STAGE_DATASETS) >= 2

        names = [ds.name for ds in VISION_STAGE_DATASETS]
        assert "the_cauldron" in names
        assert "docmatix" in names

    def test_video_stage_datasets_exist(self):
        """Test that video stage datasets are defined."""
        assert len(VIDEO_STAGE_DATASETS) >= 5

        modalities = set(ds.modality for ds in VIDEO_STAGE_DATASETS)
        assert "image" in modalities
        assert "video" in modalities

    def test_vision_stage_weights(self):
        """Test vision stage dataset weights."""
        cauldron = next(ds for ds in VISION_STAGE_DATASETS if ds.name == "the_cauldron")
        docmatix = next(ds for ds in VISION_STAGE_DATASETS if ds.name == "docmatix")

        assert cauldron.mix_weight == 0.35
        assert docmatix.mix_weight == 0.41

    def test_all_datasets_have_required_fields(self):
        """Test that all preset datasets have required fields."""
        all_datasets = VISION_STAGE_DATASETS + VIDEO_STAGE_DATASETS

        for ds in all_datasets:
            assert ds.name is not None
            assert ds.hf_path is not None
            assert ds.modality in ["image", "video", "multi-image", "text"]
            assert ds.stage in ["vision", "video"]
            assert 0 < ds.mix_weight <= 1.0

    def test_vision_stage_all_vision(self):
        """Test that vision stage datasets have stage='vision'."""
        for ds in VISION_STAGE_DATASETS:
            assert ds.stage == "vision"

    def test_video_stage_all_video(self):
        """Test that video stage datasets have stage='video'."""
        for ds in VIDEO_STAGE_DATASETS:
            assert ds.stage == "video"

    def test_video_stage_has_multiple_modalities(self):
        """Test video stage has image, video, and multi-image."""
        modalities = {ds.modality for ds in VIDEO_STAGE_DATASETS}

        assert "image" in modalities
        assert "video" in modalities
        assert "multi-image" in modalities


class TestGetAllDatasets:
    """Tests for get_all_datasets function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = get_all_datasets()

        assert isinstance(result, dict)
        assert "vision" in result
        assert "video" in result

    def test_vision_datasets(self):
        """Test vision stage datasets."""
        result = get_all_datasets()

        assert len(result["vision"]) == len(VISION_STAGE_DATASETS)

    def test_video_datasets(self):
        """Test video stage datasets."""
        result = get_all_datasets()

        assert len(result["video"]) == len(VIDEO_STAGE_DATASETS)

    def test_returns_correct_types(self):
        """Test that returned values are DatasetConfig lists."""
        result = get_all_datasets()

        for stage, datasets in result.items():
            assert isinstance(datasets, list)
            for ds in datasets:
                assert isinstance(ds, DatasetConfig)


class TestPreviewDatasets:
    """Tests for preview_datasets function."""

    def test_preview_runs_without_error(self, capsys):
        """Test that preview runs and produces output."""
        preview_datasets()

        captured = capsys.readouterr()
        assert "SmolVLM2 Training Datasets" in captured.out
        assert "VISION STAGE" in captured.out
        assert "VIDEO STAGE" in captured.out

    def test_preview_shows_dataset_names(self, capsys):
        """Test that preview shows dataset names."""
        preview_datasets()

        captured = capsys.readouterr()
        assert "the_cauldron" in captured.out
        assert "docmatix" in captured.out

    def test_preview_shows_percentages(self, capsys):
        """Test that preview shows weight percentages."""
        preview_datasets()

        captured = capsys.readouterr()
        # Should contain percentage values
        assert "%" in captured.out


class TestDownloadDataset:
    """Tests for download_dataset function."""

    def test_download_streaming(self, temp_data_dir):
        """Test downloading in streaming mode (mock)."""
        config = DatasetConfig(
            name="test_dataset",
            hf_path="test/path",
            streaming=True,
        )

        with patch("datasets.load_dataset") as mock_load:
            # Mock the dataset
            mock_ds = MagicMock()
            mock_ds.take.return_value = [{"id": 1}, {"id": 2}]
            mock_load.return_value = mock_ds

            result = download_dataset(config, temp_data_dir, num_samples=2)

            assert result is True
            mock_load.assert_called_once()

    def test_download_non_streaming(self, temp_data_dir):
        """Test downloading in non-streaming mode (mock)."""
        config = DatasetConfig(
            name="test_dataset",
            hf_path="test/path",
            streaming=False,
        )

        with patch("datasets.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__len__ = MagicMock(return_value=100)
            mock_ds.select.return_value = mock_ds
            mock_ds.save_to_disk = MagicMock()
            mock_load.return_value = mock_ds

            result = download_dataset(config, temp_data_dir, num_samples=10)

            assert result is True

    def test_download_handles_error(self, temp_data_dir):
        """Test that download handles errors gracefully."""
        config = DatasetConfig(
            name="test_dataset",
            hf_path="nonexistent/path",
        )

        with patch("datasets.load_dataset") as mock_load:
            mock_load.side_effect = Exception("Network error")

            result = download_dataset(config, temp_data_dir)

            assert result is False

    def test_metadata_saved_streaming(self, temp_data_dir):
        """Test that metadata is saved for streaming datasets."""
        config = DatasetConfig(
            name="test_dataset",
            hf_path="test/path",
            modality="image",
            stage="vision",
            mix_weight=0.5,
            streaming=True,
        )

        with patch("datasets.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.take.return_value = [{"id": 1}]
            mock_load.return_value = mock_ds

            download_dataset(config, temp_data_dir, num_samples=1)

            # Check metadata file exists
            metadata_path = temp_data_dir / config.stage / config.name / "metadata.json"
            assert metadata_path.exists()

            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["name"] == "test_dataset"
            assert metadata["modality"] == "image"
            assert metadata["mix_weight"] == 0.5

    def test_download_calls_load_dataset_correctly(self, temp_data_dir):
        """Test that load_dataset is called with correct parameters."""
        config = DatasetConfig(
            name="test",
            hf_path="org/repo",
            hf_name="subset",
            split="train",
            streaming=True,
        )

        with patch("datasets.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.take.return_value = []
            mock_load.return_value = mock_ds

            download_dataset(config, temp_data_dir, num_samples=1)

            mock_load.assert_called_once_with(
                "org/repo",
                name="subset",
                split="train",
                streaming=True,
                num_proc=None,  # None for streaming
            )

    def test_download_creates_directory(self, temp_data_dir):
        """Test that download creates output directory."""
        config = DatasetConfig(
            name="new_dataset",
            hf_path="test/path",
            stage="vision",
            streaming=True,
        )

        with patch("datasets.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.take.return_value = [{"id": 1}]
            mock_load.return_value = mock_ds

            download_dataset(config, temp_data_dir, num_samples=1)

            expected_dir = temp_data_dir / "vision" / "new_dataset"
            assert expected_dir.exists()


class TestDownloadAll:
    """Tests for download_all function."""

    def test_download_all_vision(self, temp_data_dir):
        """Test downloading all vision stage datasets."""
        with patch("src.data.download_datasets.download_dataset") as mock_download:
            mock_download.return_value = True

            download_all(temp_data_dir, stage="vision", num_samples=1)

            # Should be called for each vision dataset
            assert mock_download.call_count == len(VISION_STAGE_DATASETS)

    def test_download_all_video(self, temp_data_dir):
        """Test downloading all video stage datasets."""
        with patch("src.data.download_datasets.download_dataset") as mock_download:
            mock_download.return_value = True

            download_all(temp_data_dir, stage="video", num_samples=1)

            assert mock_download.call_count == len(VIDEO_STAGE_DATASETS)

    def test_download_all_both_stages(self, temp_data_dir):
        """Test downloading all datasets from both stages."""
        with patch("src.data.download_datasets.download_dataset") as mock_download:
            mock_download.return_value = True

            download_all(temp_data_dir, num_samples=1)

            total = len(VISION_STAGE_DATASETS) + len(VIDEO_STAGE_DATASETS)
            assert mock_download.call_count == total

    def test_download_all_creates_output_dir(self, tmp_path):
        """Test that download_all creates output directory."""
        output_dir = tmp_path / "new_data_dir"

        with patch("src.data.download_datasets.download_dataset") as mock_download:
            mock_download.return_value = True

            download_all(output_dir, stage="vision", num_samples=1)

            assert output_dir.exists()

    def test_download_all_handles_failures(self, temp_data_dir, capsys):
        """Test that download_all handles failures gracefully."""
        with patch("src.data.download_datasets.download_dataset") as mock_download:
            # First succeeds, second fails
            mock_download.side_effect = [True, False]

            download_all(temp_data_dir, stage="vision", num_samples=1)

            captured = capsys.readouterr()
            assert "SUCCESS" in captured.out
            assert "FAILED" in captured.out

    def test_download_all_prints_summary(self, temp_data_dir, capsys):
        """Test that download_all prints summary."""
        with patch("src.data.download_datasets.download_dataset") as mock_download:
            mock_download.return_value = True

            download_all(temp_data_dir, stage="vision", num_samples=1)

            captured = capsys.readouterr()
            assert "Download Summary" in captured.out
