"""Data collator for SmolVLM2 training.

Handles batching of multi-modal samples with variable-length
sequences and multiple images/video frames.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class SmolVLMDataCollator:
    """Collate function for SmolVLM2 training.

    Handles:
    - Variable length text sequences (with padding)
    - Variable number of images per sample
    - Video frames
    - Attention masks
    - Labels for causal LM training
    """

    tokenizer: Any
    processor: Any
    max_length: int = 2048
    padding: str = "max_length"
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Collate batch of features.

        Args:
            features: List of processed samples

        Returns:
            Batched tensors
        """
        batch_size = len(features)

        # Initialize batch containers
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "pixel_values": [],
        }

        # Track which samples have images
        has_images = []

        for feature in features:
            # Text tokens
            if "input_ids" in feature:
                batch["input_ids"].append(feature["input_ids"])
            if "attention_mask" in feature:
                batch["attention_mask"].append(feature["attention_mask"])
            if "labels" in feature:
                batch["labels"].append(feature["labels"])
            elif "input_ids" in feature:
                # Use input_ids as labels for causal LM
                batch["labels"].append(feature["input_ids"].clone())

            # Images
            if "pixel_values" in feature and feature["pixel_values"] is not None:
                batch["pixel_values"].append(feature["pixel_values"])
                has_images.append(True)
            else:
                has_images.append(False)

        # Pad text sequences
        if batch["input_ids"]:
            batch["input_ids"] = self._pad_sequences(
                batch["input_ids"],
                pad_value=self.tokenizer.pad_token_id or 0,
            )

        if batch["attention_mask"]:
            batch["attention_mask"] = self._pad_sequences(
                batch["attention_mask"],
                pad_value=0,
            )

        if batch["labels"]:
            batch["labels"] = self._pad_sequences(
                batch["labels"],
                pad_value=self.label_pad_token_id,
            )

        # Stack images if all samples have them
        if batch["pixel_values"] and all(has_images):
            try:
                # Check if all images have same shape
                shapes = [pv.shape for pv in batch["pixel_values"]]
                if len(set(shapes)) == 1:
                    batch["pixel_values"] = torch.stack(batch["pixel_values"])
                else:
                    # Pad to max size if shapes differ
                    batch["pixel_values"] = self._pad_images(batch["pixel_values"])
            except Exception as e:
                logger.warning(f"Failed to stack pixel_values: {e}")
                del batch["pixel_values"]
        else:
            # Remove if not all samples have images
            if "pixel_values" in batch:
                del batch["pixel_values"]

        return batch

    def _pad_sequences(
        self,
        sequences: List[torch.Tensor],
        pad_value: int,
    ) -> torch.Tensor:
        """Pad sequences to same length.

        Args:
            sequences: List of 1D tensors
            pad_value: Value to pad with

        Returns:
            Padded tensor of shape (batch, max_length)
        """
        # Find max length
        max_len = max(seq.size(0) for seq in sequences)

        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Cap at max_length
        max_len = min(max_len, self.max_length)

        # Pad sequences
        padded = []
        for seq in sequences:
            if seq.size(0) > max_len:
                # Truncate
                padded.append(seq[:max_len])
            elif seq.size(0) < max_len:
                # Pad
                padding = torch.full(
                    (max_len - seq.size(0),),
                    pad_value,
                    dtype=seq.dtype,
                )
                padded.append(torch.cat([seq, padding]))
            else:
                padded.append(seq)

        return torch.stack(padded)

    def _pad_images(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """Pad images to same size.

        Args:
            images: List of image tensors (C, H, W) or (N, C, H, W)

        Returns:
            Padded tensor
        """
        # Get max dimensions
        if images[0].dim() == 3:
            # Single images: (C, H, W)
            max_h = max(img.size(1) for img in images)
            max_w = max(img.size(2) for img in images)

            padded = []
            for img in images:
                c, h, w = img.shape
                if h < max_h or w < max_w:
                    new_img = torch.zeros(c, max_h, max_w, dtype=img.dtype)
                    new_img[:, :h, :w] = img
                    padded.append(new_img)
                else:
                    padded.append(img)

            return torch.stack(padded)

        elif images[0].dim() == 4:
            # Multiple images/frames: (N, C, H, W)
            max_n = max(img.size(0) for img in images)
            max_h = max(img.size(2) for img in images)
            max_w = max(img.size(3) for img in images)

            padded = []
            for img in images:
                n, c, h, w = img.shape
                new_img = torch.zeros(max_n, c, max_h, max_w, dtype=img.dtype)
                new_img[:n, :, :h, :w] = img
                padded.append(new_img)

            return torch.stack(padded)

        else:
            raise ValueError(f"Unexpected image dimensions: {images[0].dim()}")


@dataclass
class VideoDataCollator(SmolVLMDataCollator):
    """Data collator for video training.

    Extends SmolVLMDataCollator with video-specific handling.
    """

    max_frames: int = 32

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Collate batch with video support."""
        batch = super().__call__(features)

        # Handle video frames separately
        video_features = [f.get("video_frames") for f in features]

        if any(vf is not None for vf in video_features):
            # Pad video frames to same length
            max_frames = min(
                max(vf.size(0) for vf in video_features if vf is not None),
                self.max_frames,
            )

            padded_videos = []
            video_masks = []

            for vf in video_features:
                if vf is None:
                    # Create zero tensor for samples without video
                    padded_videos.append(torch.zeros(max_frames, 3, 384, 384))
                    video_masks.append(torch.zeros(max_frames, dtype=torch.bool))
                else:
                    n_frames = vf.size(0)
                    if n_frames > max_frames:
                        # Subsample frames uniformly
                        indices = torch.linspace(0, n_frames - 1, max_frames).long()
                        padded_videos.append(vf[indices])
                        video_masks.append(torch.ones(max_frames, dtype=torch.bool))
                    elif n_frames < max_frames:
                        # Pad with zeros
                        padding = torch.zeros(
                            max_frames - n_frames,
                            *vf.shape[1:],
                            dtype=vf.dtype,
                        )
                        padded_videos.append(torch.cat([vf, padding]))
                        mask = torch.cat(
                            [
                                torch.ones(n_frames, dtype=torch.bool),
                                torch.zeros(max_frames - n_frames, dtype=torch.bool),
                            ]
                        )
                        video_masks.append(mask)
                    else:
                        padded_videos.append(vf)
                        video_masks.append(torch.ones(max_frames, dtype=torch.bool))

            batch["video_frames"] = torch.stack(padded_videos)
            batch["video_mask"] = torch.stack(video_masks)

        return batch


def create_data_collator(
    tokenizer,
    processor,
    max_length: int = 2048,
    is_video: bool = False,
    **kwargs,
) -> Union[SmolVLMDataCollator, VideoDataCollator]:
    """Create appropriate data collator.

    Args:
        tokenizer: Tokenizer instance
        processor: Image processor
        max_length: Maximum sequence length
        is_video: Whether training includes video

    Returns:
        Data collator instance
    """
    if is_video:
        return VideoDataCollator(
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    else:
        return SmolVLMDataCollator(
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
