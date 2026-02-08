#!/usr/bin/env python3
"""Download datasets for SmolVLM2 training.

Downloads all required datasets for vision and video stage training.
Supports streaming mode for large datasets.

Usage:
    # Download all datasets
    python -m src.data.download_datasets --output-dir ./data

    # Download specific stage
    python -m src.data.download_datasets --stage vision --output-dir ./data

    # Preview datasets without downloading
    python -m src.data.download_datasets --preview
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    hf_path: str
    hf_name: Optional[str] = None
    split: str = "train"
    streaming: bool = True
    modality: str = "image"  # image, video, multi-image, text
    stage: str = "vision"  # vision or video
    mix_weight: float = 1.0
    max_samples: Optional[int] = None


# Vision Stage Datasets (Stage 1)
VISION_STAGE_DATASETS = [
    DatasetConfig(
        name="the_cauldron",
        hf_path="HuggingFaceM4/the_cauldron",
        modality="image",
        stage="vision",
        mix_weight=0.35,
    ),
    DatasetConfig(
        name="docmatix",
        hf_path="HuggingFaceM4/Docmatix",
        hf_name="images",
        modality="image",
        stage="vision",
        mix_weight=0.41,
    ),
]

# Video Stage Datasets (Stage 2) - 3.3M samples total
VIDEO_STAGE_DATASETS = [
    # Image datasets (34.4%)
    DatasetConfig(
        name="llava_onevision",
        hf_path="lmms-lab/LLaVA-OneVision-Data",
        modality="image",
        stage="video",
        mix_weight=0.15,
    ),
    DatasetConfig(
        name="mammoth_vl",
        hf_path="MAmmoTH-VL/MAmmoTH-VL-Instruct-12M",
        modality="image",
        stage="video",
        mix_weight=0.10,
    ),
    # Multi-image datasets (12.3%)
    DatasetConfig(
        name="m4_instruct",
        hf_path="lmms-lab/M4-Instruct-Data",
        modality="multi-image",
        stage="video",
        mix_weight=0.12,
    ),
    # Video datasets (33.0%)
    DatasetConfig(
        name="llava_video_178k",
        hf_path="lmms-lab/LLaVA-Video-178K",
        modality="video",
        stage="video",
        mix_weight=0.08,
    ),
    DatasetConfig(
        name="finevideo",
        hf_path="HuggingFaceFV/finevideo",
        modality="video",
        stage="video",
        mix_weight=0.05,
    ),
    DatasetConfig(
        name="video_star",
        hf_path="orrzohar/Video-STaR",
        modality="video",
        stage="video",
        mix_weight=0.05,
    ),
    DatasetConfig(
        name="vript",
        hf_path="Mutonix/Vript",
        modality="video",
        stage="video",
        mix_weight=0.05,
    ),
    DatasetConfig(
        name="vista_400k",
        hf_path="TIGER-Lab/VISTA-400K",
        modality="video",
        stage="video",
        mix_weight=0.05,
    ),
    DatasetConfig(
        name="moviechat",
        hf_path="Enxin/MovieChat-1K_train",
        modality="video",
        stage="video",
        mix_weight=0.03,
    ),
    DatasetConfig(
        name="sharegpt4video",
        hf_path="ShareGPT4Video/ShareGPT4Video",
        modality="video",
        stage="video",
        mix_weight=0.02,
    ),
    # Text datasets (20.2%)
    # These are typically included from the vision stage or LLM training
]


def get_all_datasets() -> Dict[str, List[DatasetConfig]]:
    """Get all datasets organized by stage."""
    return {
        "vision": VISION_STAGE_DATASETS,
        "video": VIDEO_STAGE_DATASETS,
    }


def preview_datasets():
    """Print information about all datasets."""
    print("\n" + "=" * 80)
    print("SmolVLM2 Training Datasets")
    print("=" * 80)

    all_datasets = get_all_datasets()

    for stage, datasets in all_datasets.items():
        print(f"\n{stage.upper()} STAGE DATASETS:")
        print("-" * 60)

        total_weight = sum(d.mix_weight for d in datasets)

        for ds in datasets:
            pct = (ds.mix_weight / total_weight * 100) if total_weight > 0 else 0
            print(f"  {ds.name:25} | {ds.modality:12} | {pct:5.1f}% | {ds.hf_path}")

        print(f"\n  Total datasets: {len(datasets)}")
        print(f"  Total weight: {total_weight:.2f}")


def download_dataset(
    config: DatasetConfig,
    output_dir: Path,
    num_samples: Optional[int] = None,
    num_proc: int = 4,
) -> bool:
    """Download a single dataset.

    Args:
        config: Dataset configuration
        output_dir: Directory to save dataset
        num_samples: Optional limit on samples to download
        num_proc: Number of processes for downloading

    Returns:
        True if successful
    """
    from datasets import load_dataset

    logger.info(f"Downloading {config.name} from {config.hf_path}")

    try:
        # Load dataset
        ds = load_dataset(
            config.hf_path,
            name=config.hf_name,
            split=config.split,
            streaming=config.streaming,
            num_proc=num_proc if not config.streaming else None,
        )

        # Take subset if requested
        if num_samples and config.streaming:
            ds = ds.take(num_samples)
            samples = list(ds)
            logger.info(f"Downloaded {len(samples)} samples from {config.name}")
        elif num_samples and not config.streaming:
            ds = ds.select(range(min(num_samples, len(ds))))
            logger.info(f"Selected {len(ds)} samples from {config.name}")

        # Save to disk
        save_path = output_dir / config.stage / config.name

        if config.streaming:
            # For streaming datasets, save samples incrementally
            save_path.mkdir(parents=True, exist_ok=True)

            # Save metadata
            import json

            metadata = {
                "name": config.name,
                "hf_path": config.hf_path,
                "modality": config.modality,
                "stage": config.stage,
                "mix_weight": config.mix_weight,
            }
            with open(save_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved metadata for {config.name}")
        else:
            # Save full dataset
            ds.save_to_disk(str(save_path))
            logger.info(f"Saved {config.name} to {save_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to download {config.name}: {e}")
        return False


def download_all(
    output_dir: Path,
    stage: Optional[str] = None,
    num_samples: Optional[int] = None,
    max_workers: int = 2,
):
    """Download all datasets.

    Args:
        output_dir: Directory to save datasets
        stage: Optional stage to download ("vision" or "video")
        num_samples: Optional limit per dataset
        max_workers: Number of parallel downloads
    """
    all_datasets = get_all_datasets()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by stage if specified
    if stage:
        datasets_to_download = all_datasets.get(stage, [])
    else:
        datasets_to_download = []
        for stage_datasets in all_datasets.values():
            datasets_to_download.extend(stage_datasets)

    logger.info(f"Downloading {len(datasets_to_download)} datasets to {output_dir}")

    # Download sequentially to avoid rate limits
    # Could parallelize with ThreadPoolExecutor if needed
    results = {}
    for config in datasets_to_download:
        success = download_dataset(config, output_dir, num_samples)
        results[config.name] = success

    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name:30} {status}")

    num_success = sum(results.values())
    print(f"\nDownloaded {num_success}/{len(results)} datasets")


def main():
    parser = argparse.ArgumentParser(description="Download SmolVLM2 training datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save datasets",
    )
    parser.add_argument(
        "--stage",
        choices=["vision", "video"],
        help="Download only datasets for specific stage",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Limit number of samples per dataset (for testing)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview datasets without downloading",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of parallel downloads",
    )

    args = parser.parse_args()

    if args.preview:
        preview_datasets()
    else:
        download_all(
            Path(args.output_dir),
            stage=args.stage,
            num_samples=args.num_samples,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
