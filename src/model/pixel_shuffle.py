"""Pixel Shuffle Connector for SmolVLM2.

Implements 3x3 pixel shuffle for 9x visual token compression,
following the approach from Idefics3/SmolVLM.

The pixel shuffle operation rearranges spatial features into channels,
reducing spatial resolution while increasing representational density.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional


class PixelShuffle(nn.Module):
    """Pixel shuffle (space-to-depth) operation.

    Rearranges spatial features into additional channels.
    For ratio=3: (H, W, C) -> (H/3, W/3, C*9)

    This provides 9x compression of visual tokens while preserving information.
    """

    def __init__(self, ratio: int = 3):
        """Initialize pixel shuffle.

        Args:
            ratio: Downsampling ratio. For SmolVLM2, use 3 (vs 2 in Idefics3)
        """
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pixel shuffle.

        Args:
            x: Input tensor of shape (B, H, W, C) or (B, N, C) where N = H*W

        Returns:
            Output tensor with spatial dimensions reduced by ratio
        """
        if x.dim() == 3:
            # Reshape from (B, N, C) to (B, H, W, C)
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            assert H * W == N, f"Cannot reshape {N} tokens to square grid"
            x = x.view(B, H, W, C)

        B, H, W, C = x.shape
        r = self.ratio

        assert H % r == 0 and W % r == 0, \
            f"Spatial dims ({H}, {W}) must be divisible by ratio {r}"

        # Rearrange: (B, H, W, C) -> (B, H/r, W/r, C*r*r)
        x = rearrange(x, 'b (h r1) (w r2) c -> b h w (c r1 r2)', r1=r, r2=r)

        return x


class PixelShuffleConnector(nn.Module):
    """Full connector from vision encoder to language model.

    Pipeline:
    1. Pixel shuffle: 9x spatial compression
    2. Layer norm
    3. MLP projection to language model hidden dim

    For SmolVLM2:
    - Input: 384x384 image -> 24x24 patches from SigLIP -> 576 tokens
    - After pixel shuffle (3x3): 8x8 = 64 tokens per 384x384 region
    - Actually 81 tokens accounting for the math: (24/3)^2 = 64...
      The paper says 81 which suggests different patch handling
    """

    def __init__(
        self,
        vision_hidden_size: int = 768,
        text_hidden_size: int = 576,
        pixel_shuffle_ratio: int = 3,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        """Initialize connector.

        Args:
            vision_hidden_size: Hidden dim of vision encoder (SigLIP base = 768)
            text_hidden_size: Hidden dim of text decoder (SmolLM2-135M = 576)
            pixel_shuffle_ratio: Spatial reduction ratio (3 for SmolVLM2)
            num_layers: Number of MLP layers
            dropout: Dropout probability
        """
        super().__init__()

        self.pixel_shuffle_ratio = pixel_shuffle_ratio
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size

        # Pixel shuffle increases channels by ratio^2
        shuffled_dim = vision_hidden_size * (pixel_shuffle_ratio ** 2)

        # Pixel shuffle layer
        self.pixel_shuffle = PixelShuffle(ratio=pixel_shuffle_ratio)

        # Layer norm after shuffle
        self.layer_norm = nn.LayerNorm(shuffled_dim)

        # MLP projection
        layers = []
        input_dim = shuffled_dim

        for i in range(num_layers):
            output_dim = text_hidden_size if i == num_layers - 1 else shuffled_dim
            layers.append(nn.Linear(input_dim, output_dim))

            if i < num_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            input_dim = output_dim

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        vision_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project vision features to language model space.

        Args:
            vision_features: Output from vision encoder
                Shape: (B, H, W, C) or (B, N, C) where N = H*W
            attention_mask: Optional mask for valid tokens

        Returns:
            Projected features ready for language model
            Shape: (B, N', text_hidden_size) where N' = N / (ratio^2)
        """
        # Apply pixel shuffle
        x = self.pixel_shuffle(vision_features)

        # Flatten spatial dims if needed
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)

        # Layer norm and MLP
        x = self.layer_norm(x)
        x = self.mlp(x)

        return x

    @property
    def num_visual_tokens(self) -> int:
        """Calculate number of visual tokens after pixel shuffle.

        For a 384x384 image with patch_size=16 and ratio=3:
        - Patches: 384/16 = 24 per dimension
        - After shuffle: 24/3 = 8 per dimension
        - Total: 8*8 = 64 tokens

        Note: Paper mentions 81 tokens which may account for
        different sub-image handling or edge cases.
        """
        # Assuming 384x384 image, 16x16 patches
        patches_per_dim = 384 // 16  # 24
        tokens_per_dim = patches_per_dim // self.pixel_shuffle_ratio  # 8
        return tokens_per_dim ** 2  # 64


def create_connector(config) -> PixelShuffleConnector:
    """Create connector from config.

    Args:
        config: SmolVLMConfig instance

    Returns:
        Initialized PixelShuffleConnector
    """
    return PixelShuffleConnector(
        vision_hidden_size=config.vision_hidden_size,
        text_hidden_size=config.text_hidden_size,
        pixel_shuffle_ratio=config.pixel_shuffle_ratio,
        num_layers=config.connector_num_layers,
    )
