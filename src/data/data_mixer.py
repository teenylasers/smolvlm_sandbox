"""Dataset mixer for SmolVLM2 training.

Implements weighted sampling across multiple datasets
according to the data mixing ratios from the paper.
"""

import torch
from torch.utils.data import IterableDataset
from typing import Dict, List, Optional, Iterator, Any
import random
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetWeight:
    """Configuration for a dataset in the mixture."""
    name: str
    dataset: IterableDataset
    weight: float
    modality: str = "image"  # image, video, multi-image, text


class DatasetMixer(IterableDataset):
    """Mix multiple datasets with configurable weights.

    Implements weighted sampling across datasets, ensuring each
    dataset contributes proportionally to the training.
    """

    def __init__(
        self,
        datasets: List[DatasetWeight],
        seed: int = 42,
        shuffle_buffer_size: int = 10000,
    ):
        """Initialize dataset mixer.

        Args:
            datasets: List of datasets with weights
            seed: Random seed for reproducibility
            shuffle_buffer_size: Size of shuffle buffer
        """
        self.datasets = datasets
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size

        # Normalize weights
        total_weight = sum(d.weight for d in datasets)
        self.probabilities = [d.weight / total_weight for d in datasets]

        logger.info(
            f"Initialized mixer with {len(datasets)} datasets. "
            f"Weights: {[f'{d.name}={d.weight:.2f}' for d in datasets]}"
        )

    def __iter__(self) -> Iterator[Any]:
        """Iterate over mixed samples.

        Uses weighted sampling to select which dataset to draw from.
        """
        random.seed(self.seed)

        # Create iterators for each dataset
        iterators = {d.name: iter(d.dataset) for d in self.datasets}
        dataset_names = [d.name for d in self.datasets]
        active_datasets = set(dataset_names)

        # Optional: shuffle buffer for better mixing
        buffer = []

        while active_datasets:
            # Update probabilities for remaining datasets
            active_probs = []
            active_names = []
            for d, prob in zip(self.datasets, self.probabilities):
                if d.name in active_datasets:
                    active_names.append(d.name)
                    active_probs.append(prob)

            if not active_probs:
                break

            # Renormalize
            total = sum(active_probs)
            active_probs = [p / total for p in active_probs]

            # Sample dataset
            selected = random.choices(active_names, weights=active_probs)[0]

            try:
                sample = next(iterators[selected])

                # Add dataset info to sample
                if isinstance(sample, dict):
                    sample["_dataset_name"] = selected
                    sample["_modality"] = next(
                        d.modality for d in self.datasets if d.name == selected
                    )

                # Add to buffer or yield directly
                if self.shuffle_buffer_size > 0:
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer_size:
                        random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []
                else:
                    yield sample

            except StopIteration:
                active_datasets.remove(selected)
                logger.info(f"Dataset {selected} exhausted")

        # Yield remaining buffer
        random.shuffle(buffer)
        for item in buffer:
            yield item


class BalancedMixer(IterableDataset):
    """Mix datasets while balancing modalities.

    Ensures even distribution across image, video, multi-image, and text.
    """

    def __init__(
        self,
        datasets: List[DatasetWeight],
        modality_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        """Initialize balanced mixer.

        Args:
            datasets: List of datasets with weights
            modality_weights: Override weights per modality
            seed: Random seed
        """
        self.datasets = datasets
        self.seed = seed

        # Default modality weights from paper (video stage)
        self.modality_weights = modality_weights or {
            "image": 0.344,
            "video": 0.330,
            "text": 0.202,
            "multi-image": 0.123,
        }

        # Group datasets by modality
        self.modality_datasets: Dict[str, List[DatasetWeight]] = {}
        for d in datasets:
            if d.modality not in self.modality_datasets:
                self.modality_datasets[d.modality] = []
            self.modality_datasets[d.modality].append(d)

        logger.info(
            f"Balanced mixer initialized. Modalities: {list(self.modality_datasets.keys())}"
        )

    def __iter__(self) -> Iterator[Any]:
        """Iterate with modality balancing."""
        random.seed(self.seed)

        # Create modality-level mixers
        modality_mixers = {}
        for modality, datasets in self.modality_datasets.items():
            modality_mixers[modality] = DatasetMixer(
                datasets=datasets,
                seed=self.seed,
                shuffle_buffer_size=1000,
            )

        # Create iterators
        modality_iters = {
            m: iter(mixer) for m, mixer in modality_mixers.items()
        }
        active_modalities = set(modality_iters.keys())

        while active_modalities:
            # Sample modality
            active_list = list(active_modalities)
            weights = [
                self.modality_weights.get(m, 0.1) for m in active_list
            ]
            total = sum(weights)
            weights = [w / total for w in weights]

            selected_modality = random.choices(active_list, weights=weights)[0]

            try:
                sample = next(modality_iters[selected_modality])
                yield sample

            except StopIteration:
                active_modalities.remove(selected_modality)
                logger.info(f"Modality {selected_modality} exhausted")


def create_vision_stage_mixer(
    the_cauldron_ds,
    docmatix_ds,
    seed: int = 42,
) -> DatasetMixer:
    """Create mixer for vision stage training.

    Args:
        the_cauldron_ds: The Cauldron dataset
        docmatix_ds: Docmatix dataset
        seed: Random seed

    Returns:
        DatasetMixer configured for vision stage
    """
    datasets = [
        DatasetWeight(
            name="the_cauldron",
            dataset=the_cauldron_ds,
            weight=0.35,
            modality="image",
        ),
        DatasetWeight(
            name="docmatix",
            dataset=docmatix_ds,
            weight=0.41,
            modality="image",
        ),
    ]

    return DatasetMixer(datasets=datasets, seed=seed)


def create_video_stage_mixer(
    datasets_dict: Dict[str, IterableDataset],
    seed: int = 42,
) -> BalancedMixer:
    """Create mixer for video stage training.

    Args:
        datasets_dict: Dictionary of dataset name -> dataset
        seed: Random seed

    Returns:
        BalancedMixer configured for video stage
    """
    # Weights and modalities from paper
    dataset_configs = {
        "llava_onevision": ("image", 0.15),
        "mammoth_vl": ("image", 0.10),
        "m4_instruct": ("multi-image", 0.12),
        "llava_video": ("video", 0.08),
        "finevideo": ("video", 0.05),
        "video_star": ("video", 0.05),
        "vript": ("video", 0.05),
        "vista_400k": ("video", 0.05),
        "moviechat": ("video", 0.03),
        "sharegpt4video": ("video", 0.02),
    }

    datasets = []
    for name, ds in datasets_dict.items():
        if name in dataset_configs:
            modality, weight = dataset_configs[name]
            datasets.append(DatasetWeight(
                name=name,
                dataset=ds,
                weight=weight,
                modality=modality,
            ))

    return BalancedMixer(datasets=datasets, seed=seed)
