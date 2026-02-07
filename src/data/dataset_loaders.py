"""Dataset loaders for SmolVLM2 training.

Implements loaders for each dataset used in vision and video stages.
Handles different data formats and converts to unified format.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
from PIL import Image
from typing import Dict, List, Optional, Any, Iterator
import logging
from pathlib import Path
import json
import io

logger = logging.getLogger(__name__)


class SmolVLMSample:
    """Unified sample format for SmolVLM2 training.

    Attributes:
        images: List of PIL images
        video_frames: List of video frames (PIL images)
        conversations: List of conversation turns
        metadata: Additional metadata
    """

    def __init__(
        self,
        images: Optional[List[Image.Image]] = None,
        video_frames: Optional[List[Image.Image]] = None,
        conversations: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.images = images or []
        self.video_frames = video_frames or []
        self.conversations = conversations or []
        self.metadata = metadata or {}

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0

    @property
    def has_video(self) -> bool:
        return len(self.video_frames) > 0

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)


def load_the_cauldron(
    split: str = "train",
    streaming: bool = True,
    subset: Optional[str] = None,
) -> IterableDataset:
    """Load The Cauldron dataset.

    Multi-task VQA dataset with various image understanding tasks.

    Args:
        split: Dataset split
        streaming: Whether to stream
        subset: Optional subset name

    Returns:
        Dataset iterator
    """
    logger.info("Loading The Cauldron dataset")

    ds = load_dataset(
        "HuggingFaceM4/the_cauldron",
        name=subset,
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    return ds


def load_docmatix(
    split: str = "train",
    streaming: bool = True,
) -> IterableDataset:
    """Load Docmatix dataset.

    Document understanding dataset with OCR and document QA.

    Args:
        split: Dataset split
        streaming: Whether to stream

    Returns:
        Dataset iterator
    """
    logger.info("Loading Docmatix dataset")

    ds = load_dataset(
        "HuggingFaceM4/Docmatix",
        name="images",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    return ds


def load_llava_onevision(
    split: str = "train",
    streaming: bool = True,
) -> IterableDataset:
    """Load LLaVA-OneVision dataset.

    High-quality visual instruction tuning data.
    """
    logger.info("Loading LLaVA-OneVision dataset")

    ds = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    return ds


def load_llava_video(
    split: str = "train",
    streaming: bool = True,
) -> IterableDataset:
    """Load LLaVA-Video-178K dataset.

    Video understanding dataset with diverse video QA tasks.
    """
    logger.info("Loading LLaVA-Video-178K dataset")

    ds = load_dataset(
        "lmms-lab/LLaVA-Video-178K",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    return ds


def load_video_star(
    split: str = "train",
    streaming: bool = True,
) -> IterableDataset:
    """Load Video-STAR dataset.

    Video reasoning with step-by-step annotations.
    """
    logger.info("Loading Video-STAR dataset")

    ds = load_dataset(
        "orrzohar/Video-STaR",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    return ds


class VisionStageDataset(IterableDataset):
    """Dataset for Vision Stage training.

    Combines The Cauldron and Docmatix with proper mixing.
    """

    def __init__(
        self,
        processor,
        tokenizer,
        max_length: int = 2048,
        image_size: int = 384,
        streaming: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize vision stage dataset.

        Args:
            processor: Image processor
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
            image_size: Target image size
            streaming: Whether to stream datasets
            weights: Optional mixing weights
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.streaming = streaming

        # Default weights from paper: 41% doc, 35% cauldron
        self.weights = weights or {
            "the_cauldron": 0.35,
            "docmatix": 0.41,
        }

        # Load datasets
        self.datasets = {}
        if "the_cauldron" in self.weights:
            self.datasets["the_cauldron"] = load_the_cauldron(streaming=streaming)
        if "docmatix" in self.weights:
            self.datasets["docmatix"] = load_docmatix(streaming=streaming)

        logger.info(f"Loaded {len(self.datasets)} datasets for vision stage")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over mixed samples."""
        import random

        # Create iterators
        iterators = {name: iter(ds) for name, ds in self.datasets.items()}

        # Weighted sampling
        dataset_names = list(self.weights.keys())
        weights = [self.weights[name] for name in dataset_names]

        while iterators:
            # Sample dataset according to weights
            available = [name for name in dataset_names if name in iterators]
            if not available:
                break

            available_weights = [self.weights[name] for name in available]
            total = sum(available_weights)
            probs = [w / total for w in available_weights]

            dataset_name = random.choices(available, weights=probs)[0]

            try:
                raw_sample = next(iterators[dataset_name])
                processed = self._process_sample(raw_sample, dataset_name)
                if processed is not None:
                    yield processed
            except StopIteration:
                del iterators[dataset_name]
                logger.info(f"Exhausted {dataset_name}")

    def _process_sample(
        self,
        sample: Dict,
        dataset_name: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Process a raw sample into training format.

        Args:
            sample: Raw sample from dataset
            dataset_name: Name of source dataset

        Returns:
            Processed sample or None if invalid
        """
        try:
            # Extract image
            image = None
            if "image" in sample:
                image = sample["image"]
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image))
            elif "images" in sample and sample["images"]:
                image = sample["images"][0]
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image))

            if image is None:
                return None

            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Extract conversation
            conversations = []
            if "conversations" in sample:
                conversations = sample["conversations"]
            elif "messages" in sample:
                conversations = sample["messages"]
            elif "question" in sample and "answer" in sample:
                conversations = [
                    {"role": "user", "content": sample["question"]},
                    {"role": "assistant", "content": sample["answer"]},
                ]

            if not conversations:
                return None

            # Format text with image tokens
            text_parts = []
            for turn in conversations:
                role = turn.get("role", turn.get("from", "user"))
                content = turn.get("content", turn.get("value", ""))

                # Add image token for user turn
                if role in ["user", "human"] and "<image>" not in content:
                    content = "<image>" + content

                text_parts.append(f"{role}: {content}")

            text = "\n".join(text_parts)

            # Process image
            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)

            # Tokenize text
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "pixel_values": pixel_values,
                "labels": encoding["input_ids"].squeeze(0).clone(),
            }

        except Exception as e:
            logger.warning(f"Failed to process sample from {dataset_name}: {e}")
            return None


class VideoStageDataset(IterableDataset):
    """Dataset for Video Stage training.

    Combines multiple video datasets with proper mixing.
    Data split: Image 34.4%, Video 33.0%, Text 20.2%, Multi-image 12.3%
    """

    def __init__(
        self,
        processor,
        tokenizer,
        video_processor,
        max_length: int = 4096,
        max_frames: int = 32,
        streaming: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize video stage dataset.

        Args:
            processor: Image processor
            tokenizer: Text tokenizer
            video_processor: Video frame processor
            max_length: Maximum sequence length
            max_frames: Maximum video frames
            streaming: Whether to stream
            weights: Optional mixing weights
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.max_length = max_length
        self.max_frames = max_frames
        self.streaming = streaming

        # Default weights based on paper
        self.weights = weights or {
            "llava_onevision": 0.15,
            "llava_video": 0.08,
            "video_star": 0.05,
        }

        # Load datasets (subset for efficiency)
        self.datasets = {}

        logger.info(f"Video stage configured with {len(self.weights)} datasets")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over mixed samples."""
        # Similar to VisionStageDataset but handles video
        # Implementation would follow same pattern
        raise NotImplementedError("Video stage iteration not yet implemented")


def create_vision_stage_dataloader(
    processor,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Create dataloader for vision stage.

    Args:
        processor: Image processor
        tokenizer: Tokenizer
        batch_size: Batch size
        num_workers: Data loading workers

    Returns:
        DataLoader instance
    """
    dataset = VisionStageDataset(
        processor=processor,
        tokenizer=tokenizer,
        **kwargs,
    )

    # For IterableDataset, don't shuffle
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
