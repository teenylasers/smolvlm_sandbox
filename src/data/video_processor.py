"""Video processing utilities for SmolVLM2.

Handles video frame extraction and preprocessing.
Supports multiple backends: decord (Linux), av/opencv (Apple Silicon).
Optimal clip length is ~3.5 minutes based on training paper.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union
from pathlib import Path
import logging
import platform

logger = logging.getLogger(__name__)

# Detect available video backends
DECORD_AVAILABLE = False
AV_AVAILABLE = False
TORCHVISION_AVAILABLE = False

try:
    import decord
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    pass

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    pass

try:
    import torchvision.io
    TORCHVISION_AVAILABLE = True
except ImportError:
    pass

# Determine best backend for current platform
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

if IS_APPLE_SILICON:
    # Prefer torchvision or av on Apple Silicon (decord doesn't work)
    if TORCHVISION_AVAILABLE:
        DEFAULT_BACKEND = "torchvision"
    elif AV_AVAILABLE:
        DEFAULT_BACKEND = "av"
    elif DECORD_AVAILABLE:
        DEFAULT_BACKEND = "decord"
    else:
        DEFAULT_BACKEND = None
else:
    # Prefer decord on Linux/x86 (fastest)
    if DECORD_AVAILABLE:
        DEFAULT_BACKEND = "decord"
    elif TORCHVISION_AVAILABLE:
        DEFAULT_BACKEND = "torchvision"
    elif AV_AVAILABLE:
        DEFAULT_BACKEND = "av"
    else:
        DEFAULT_BACKEND = None

if DEFAULT_BACKEND:
    logger.info(f"Video backend: {DEFAULT_BACKEND}")
else:
    logger.warning(
        "No video backend available. Install one of: "
        "decord (Linux), torchvision, or av"
    )


class VideoProcessor:
    """Process videos into frames for VLM training.

    Extracts frames uniformly from videos and preprocesses them
    for the vision encoder.

    Supports multiple backends:
    - decord: Fastest, Linux only
    - av (PyAV): Works on Apple Silicon
    - opencv: Works on Apple Silicon
    """

    def __init__(
        self,
        image_processor,
        num_frames: int = 32,
        frame_sampling: str = "uniform",
        image_size: int = 384,
        max_video_duration: float = 210.0,  # ~3.5 minutes optimal
        backend: Optional[str] = None,
    ):
        """Initialize video processor.

        Args:
            image_processor: HuggingFace image processor
            num_frames: Number of frames to extract
            frame_sampling: Sampling strategy ("uniform", "random", "keyframe")
            image_size: Target frame size
            max_video_duration: Maximum video duration in seconds
            backend: Video backend ("decord", "av", "opencv", or None for auto)
        """
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.frame_sampling = frame_sampling
        self.image_size = image_size
        self.max_video_duration = max_video_duration

        # Select backend
        self.backend = backend or DEFAULT_BACKEND

        if self.backend is None:
            raise ImportError(
                "No video backend available. Install one of:\n"
                "  - Linux: pip install decord\n"
                "  - Apple Silicon: pip install av  OR  pip install opencv-python"
            )

        # Initialize backend-specific settings
        if self.backend == "decord":
            if not DECORD_AVAILABLE:
                raise ImportError("decord not available. Install with: pip install decord")
            decord.bridge.set_bridge("native")
        elif self.backend == "av":
            if not AV_AVAILABLE:
                raise ImportError("av not available. Install with: pip install av")
        elif self.backend == "torchvision":
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision not available. Install with: pip install torchvision")

        logger.info(f"VideoProcessor initialized with backend: {self.backend}")

    def _load_video_decord(self, video_path: str) -> Tuple[np.ndarray, float, int]:
        """Load video using decord backend."""
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        return vr, fps, total_frames

    def _load_video_av(self, video_path: str) -> Tuple[List[np.ndarray], float, int]:
        """Load video using PyAV backend."""
        container = av.open(video_path)
        stream = container.streams.video[0]

        fps = float(stream.average_rate) if stream.average_rate else 30.0
        total_frames = stream.frames if stream.frames > 0 else int(stream.duration * fps / stream.time_base.denominator)

        # We'll load frames lazily, return container info
        return container, fps, total_frames

    def _load_video_torchvision(self, video_path: str) -> Tuple[any, float, int]:
        """Load video using torchvision backend."""
        # Get video metadata
        from torchvision.io import read_video_timestamps
        pts, fps = read_video_timestamps(video_path)
        total_frames = len(pts)
        fps = fps or 30.0
        return video_path, fps, total_frames

    def _extract_frames_decord(self, video_path: str, indices: np.ndarray) -> List[np.ndarray]:
        """Extract frames using decord."""
        vr = VideoReader(str(video_path), ctx=cpu(0))
        frames = vr.get_batch(indices).asnumpy()
        return [frame for frame in frames]

    def _extract_frames_av(self, video_path: str, indices: np.ndarray) -> List[np.ndarray]:
        """Extract frames using PyAV."""
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        frames = []
        indices_set = set(indices.tolist())
        frame_idx = 0

        for frame in container.decode(video=0):
            if frame_idx in indices_set:
                img = frame.to_ndarray(format='rgb24')
                frames.append(img)
                if len(frames) >= len(indices):
                    break
            frame_idx += 1

        container.close()
        return frames

    def _extract_frames_torchvision(self, video_path: str, indices: np.ndarray) -> List[np.ndarray]:
        """Extract frames using torchvision."""
        from torchvision.io import read_video

        # Read full video (torchvision doesn't support seeking by frame index easily)
        video, audio, info = read_video(str(video_path), pts_unit='sec')

        # video shape: (T, H, W, C) - already in RGB format
        total_frames = video.shape[0]

        # Adjust indices if they exceed actual frames
        valid_indices = indices[indices < total_frames]

        frames = [video[idx].numpy() for idx in valid_indices]
        return frames

    def load_video(self, video_path: Union[str, Path]) -> Tuple[any, float, int]:
        """Load video file.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (video_handle, fps, total_frames)
        """
        video_path = str(video_path)

        if self.backend == "decord":
            return self._load_video_decord(video_path)
        elif self.backend == "av":
            return self._load_video_av(video_path)
        elif self.backend == "torchvision":
            return self._load_video_torchvision(video_path)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

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
        video_path = str(video_path)

        # Get video info
        _, fps, total_frames = self.load_video(video_path)

        logger.debug(
            f"Video: {total_frames} frames, {fps:.1f} FPS, "
            f"{total_frames/fps:.1f}s duration"
        )

        # Get frame indices
        indices = self.get_frame_indices(total_frames, fps)

        # Extract frames using appropriate backend
        if self.backend == "decord":
            frames = self._extract_frames_decord(video_path, indices)
        elif self.backend == "av":
            frames = self._extract_frames_av(video_path, indices)
        elif self.backend == "torchvision":
            frames = self._extract_frames_torchvision(video_path, indices)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

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
    def get_video_info(video_path: Union[str, Path], backend: Optional[str] = None) -> dict:
        """Get video metadata.

        Args:
            video_path: Path to video file
            backend: Video backend to use (auto-detected if None)

        Returns:
            Dictionary with video info
        """
        video_path = str(video_path)
        backend = backend or DEFAULT_BACKEND

        if backend is None:
            return {"error": "No video backend available"}

        try:
            if backend == "decord":
                vr = VideoReader(video_path, ctx=cpu(0))
                return {
                    "num_frames": len(vr),
                    "fps": vr.get_avg_fps(),
                    "duration_seconds": len(vr) / vr.get_avg_fps(),
                    "width": vr[0].shape[1],
                    "height": vr[0].shape[0],
                    "backend": "decord",
                }

            elif backend == "av":
                container = av.open(video_path)
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                num_frames = stream.frames if stream.frames > 0 else 0
                duration = float(stream.duration * stream.time_base) if stream.duration else 0
                info = {
                    "num_frames": num_frames,
                    "fps": fps,
                    "duration_seconds": duration,
                    "width": stream.width,
                    "height": stream.height,
                    "backend": "av",
                }
                container.close()
                return info

            elif backend == "torchvision":
                from torchvision.io import read_video_timestamps, read_video
                pts, fps = read_video_timestamps(video_path)
                fps = fps or 30.0
                # Read first frame to get dimensions
                video, _, _ = read_video(video_path, start_pts=0, end_pts=0.1, pts_unit='sec')
                return {
                    "num_frames": len(pts),
                    "fps": fps,
                    "duration_seconds": len(pts) / fps,
                    "width": video.shape[2] if len(video) > 0 else 0,
                    "height": video.shape[1] if len(video) > 0 else 0,
                    "backend": "torchvision",
                }

        except Exception as e:
            return {"error": str(e), "backend": backend}


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
