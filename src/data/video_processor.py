"""Video processing utilities for SmolVLM2.

Handles video frame extraction and preprocessing using decord.
Optimal clip length is ~3.5 minutes based on training paper.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import decord
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logger.warning("decord not available. Install with: pip install decord")


class VideoProcessor:
    """Process videos into frames for VLM training.

    Extracts frames uniformly from videos and preprocesses them
    for the vision encoder.
    """

    def __init__(
        self,
        image_processor,
        num_frames: int = 32,
        frame_sampling: str = "uniform",
        image_size: int = 384,
        max_video_duration: float = 210.0,  # ~3.5 minutes optimal
    ):
        """Initialize video processor.

        Args:
            image_processor: HuggingFace image processor
            num_frames: Number of frames to extract
            frame_sampling: Sampling strategy ("uniform", "random", "keyframe")
            image_size: Target frame size
            max_video_duration: Maximum video duration in seconds
        """
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.frame_sampling = frame_sampling
        self.image_size = image_size
        self.max_video_duration = max_video_duration

        if not DECORD_AVAILABLE:
            raise ImportError("decord is required for video processing")

        # Set decord to use CPU for better compatibility
        decord.bridge.set_bridge("native")

    def load_video(self, video_path: Union[str, Path]) -> VideoReader:
        """Load video file.

        Args:
            video_path: Path to video file

        Returns:
            VideoReader instance
        """
        video_path = str(video_path)
        vr = VideoReader(video_path, ctx=cpu(0))
        return vr

    def get_frame_indices(
        self,
        total_frames: int,
        fps: float,
    ) -> np.ndarray:
        """Calculate frame indices to extract.

        Args:
            total_frames: Total frames in video
            fps: Video FPS

        Returns:
            Array of frame indices
        """
        duration = total_frames / fps

        # Clip to max duration
        if duration > self.max_video_duration:
            max_frames = int(self.max_video_duration * fps)
            total_frames = min(total_frames, max_frames)

        if self.frame_sampling == "uniform":
            # Uniform sampling across video
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        elif self.frame_sampling == "random":
            # Random sampling
            indices = np.sort(
                np.random.choice(total_frames, size=self.num_frames, replace=False)
            )

        elif self.frame_sampling == "keyframe":
            # Sample more densely at beginning and end
            # Middle section sampled more sparsely
            quarter = self.num_frames // 4
            indices = np.concatenate([
                np.linspace(0, total_frames // 4, quarter, dtype=int),
                np.linspace(total_frames // 4, 3 * total_frames // 4, 2 * quarter, dtype=int),
                np.linspace(3 * total_frames // 4, total_frames - 1, quarter, dtype=int),
            ])
            indices = np.unique(indices)[:self.num_frames]

        else:
            raise ValueError(f"Unknown frame sampling: {self.frame_sampling}")

        return indices

    def extract_frames(
        self,
        video_path: Union[str, Path],
    ) -> List[Image.Image]:
        """Extract frames from video.

        Args:
            video_path: Path to video file

        Returns:
            List of PIL Images
        """
        vr = self.load_video(video_path)
        total_frames = len(vr)
        fps = vr.get_avg_fps()

        logger.debug(
            f"Video: {total_frames} frames, {fps:.1f} FPS, "
            f"{total_frames/fps:.1f}s duration"
        )

        # Get frame indices
        indices = self.get_frame_indices(total_frames, fps)

        # Extract frames
        frames = vr.get_batch(indices).asnumpy()

        # Convert to PIL
        pil_frames = [Image.fromarray(frame) for frame in frames]

        return pil_frames

    def process_video(
        self,
        video_path: Union[str, Path],
    ) -> torch.Tensor:
        """Process video into tensor for model.

        Args:
            video_path: Path to video file

        Returns:
            Tensor of shape (num_frames, C, H, W)
        """
        frames = self.extract_frames(video_path)

        # Process through image processor
        pixel_values = self.image_processor(
            images=frames,
            return_tensors="pt",
        )["pixel_values"]

        return pixel_values

    def process_frames(
        self,
        frames: List[Image.Image],
    ) -> torch.Tensor:
        """Process list of frames into tensor.

        Args:
            frames: List of PIL Images

        Returns:
            Tensor of shape (num_frames, C, H, W)
        """
        # Sample if too many frames
        if len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Process through image processor
        pixel_values = self.image_processor(
            images=frames,
            return_tensors="pt",
        )["pixel_values"]

        return pixel_values

    @staticmethod
    def get_video_info(video_path: Union[str, Path]) -> dict:
        """Get video metadata.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video info
        """
        if not DECORD_AVAILABLE:
            return {"error": "decord not available"}

        vr = VideoReader(str(video_path), ctx=cpu(0))

        return {
            "num_frames": len(vr),
            "fps": vr.get_avg_fps(),
            "duration_seconds": len(vr) / vr.get_avg_fps(),
            "width": vr[0].shape[1],
            "height": vr[0].shape[0],
        }


def create_video_processor(
    image_processor,
    num_frames: int = 32,
    frame_sampling: str = "uniform",
) -> VideoProcessor:
    """Create video processor with default settings.

    Args:
        image_processor: HuggingFace image processor
        num_frames: Number of frames to extract
        frame_sampling: Sampling strategy

    Returns:
        VideoProcessor instance
    """
    return VideoProcessor(
        image_processor=image_processor,
        num_frames=num_frames,
        frame_sampling=frame_sampling,
    )
