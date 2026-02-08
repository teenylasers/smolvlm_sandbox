"""SmolVLM2 Model Initialization.

Initializes SmolVLM2 from pretrained SigLIP and SmolLM2 components.
Extends RoPE for longer context and creates the full VLM architecture.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from .pixel_shuffle import create_connector
from .smolvlm_config import SmolVLMConfig, get_config

logger = logging.getLogger(__name__)


class SmolVLM2Model(nn.Module):
    """SmolVLM2 Vision-Language Model.

    Architecture:
    - Vision Encoder: SigLIP base-patch16-512
    - Connector: Pixel Shuffle (3x3) + MLP
    - Text Decoder: SmolLM2 (135M or 360M)
    """

    def __init__(
        self,
        config: SmolVLMConfig,
        vision_encoder: Optional[nn.Module] = None,
        text_decoder: Optional[nn.Module] = None,
        connector: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.connector = connector

        # Track which components are frozen
        self._vision_frozen = config.freeze_vision_encoder

    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self._vision_frozen = True
            logger.info("Vision encoder frozen")

    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder parameters."""
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
            self._vision_frozen = False
            logger.info("Vision encoder unfrozen")

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through vision encoder and connector.

        Args:
            pixel_values: Image tensor (B, C, H, W) or (B, N_images, C, H, W)

        Returns:
            Visual tokens (B, N_tokens, hidden_size)
        """
        # Handle batched images
        if pixel_values.dim() == 5:
            B, N_img, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(B * N_img, C, H, W)
            batch_images = True
        else:
            batch_images = False

        # Get vision features
        vision_outputs = self.vision_encoder(pixel_values)

        # Extract last hidden state
        if hasattr(vision_outputs, "last_hidden_state"):
            vision_features = vision_outputs.last_hidden_state
        else:
            vision_features = vision_outputs[0]

        # Project through connector
        visual_tokens = self.connector(vision_features)

        # Reshape if batched
        if batch_images:
            _, N_tok, D = visual_tokens.shape
            visual_tokens = visual_tokens.view(B, N_img * N_tok, D)

        return visual_tokens

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Text token IDs (B, seq_len)
            pixel_values: Image tensors (B, C, H, W)
            attention_mask: Attention mask (B, seq_len)
            labels: Target labels for loss computation

        Returns:
            Dictionary with loss and logits
        """
        # Encode images if provided
        if pixel_values is not None:
            visual_tokens = self.encode_images(pixel_values)
            # TODO: Merge visual tokens with text embeddings
            # This requires custom embedding handling

        # Get text embeddings
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)

        # TODO: Insert visual tokens at image token positions
        # For now, just pass text through

        outputs = self.text_decoder(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return outputs

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        params = self.parameters()
        if trainable_only:
            params = filter(lambda p: p.requires_grad, params)
        return sum(p.numel() for p in params)


def extend_rope_for_long_context(
    model: nn.Module,
    new_base: float = 273000.0,
    new_max_position_embeddings: int = 16384,
) -> nn.Module:
    """Extend RoPE base for longer context.

    SmolLM2 uses RoPE base of 10000 for 2k/4k context.
    For VLM with images encoded as many tokens, we extend to 16k.

    Following "Scaling Laws of RoPE-based Extrapolation":
    - Increase base from 10k to 273k for 16k context

    Args:
        model: The language model to modify
        new_base: New RoPE theta value (273000 for 16k context)
        new_max_position_embeddings: New max sequence length

    Returns:
        Modified model
    """
    # Update config
    if hasattr(model.config, "rope_theta"):
        old_base = model.config.rope_theta
        model.config.rope_theta = new_base
        logger.info(f"Extended RoPE theta from {old_base} to {new_base}")

    if hasattr(model.config, "max_position_embeddings"):
        old_max = model.config.max_position_embeddings
        model.config.max_position_embeddings = new_max_position_embeddings
        logger.info(
            f"Extended max_position_embeddings from {old_max} to {new_max_position_embeddings}"
        )

    # Reinitialize RoPE embeddings in each layer
    # This depends on the specific model architecture
    for name, module in model.named_modules():
        if "rotary" in name.lower() or "rope" in name.lower():
            if hasattr(module, "base"):
                module.base = new_base
            if hasattr(module, "inv_freq"):
                # Recalculate inverse frequencies
                dim = module.inv_freq.shape[0] * 2
                inv_freq = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))
                module.register_buffer("inv_freq", inv_freq)

    return model


def initialize_smolvlm_model(
    config: Optional[SmolVLMConfig] = None,
    model_size: str = "256m",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
    use_flash_attention: bool = True,
) -> Tuple[SmolVLM2Model, AutoProcessor, AutoTokenizer]:
    """Initialize SmolVLM2 from pretrained components.

    Args:
        config: Model configuration (if None, uses default for model_size)
        model_size: "256m" or "500m"
        torch_dtype: Model precision
        device_map: Device placement strategy
        use_flash_attention: Whether to use flash attention 2

    Returns:
        Tuple of (model, processor, tokenizer)
    """
    if config is None:
        config = get_config(model_size)

    logger.info(f"Initializing SmolVLM2 {config.model_size}")

    # Attention implementation
    attn_impl = "flash_attention_2" if use_flash_attention else "eager"

    # Load vision encoder (SigLIP)
    logger.info(f"Loading vision encoder: {config.vision_encoder_name}")
    vision_encoder = AutoModel.from_pretrained(
        config.vision_encoder_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Load text decoder (SmolLM2)
    logger.info(f"Loading text decoder: {config.text_decoder_name}")
    text_decoder = AutoModelForCausalLM.from_pretrained(
        config.text_decoder_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    # Extend context length
    text_decoder = extend_rope_for_long_context(
        text_decoder,
        new_base=config.rope_theta,
        new_max_position_embeddings=config.max_position_embeddings,
    )

    # Create connector
    logger.info("Creating pixel shuffle connector")
    connector = create_connector(config)

    # Assemble model
    model = SmolVLM2Model(
        config=config,
        vision_encoder=vision_encoder,
        text_decoder=text_decoder,
        connector=connector,
    )

    # Freeze vision encoder if configured
    if config.freeze_vision_encoder:
        model.freeze_vision_encoder()

    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(config.text_decoder_name)

    # Add special tokens for images/video
    special_tokens = {
        "additional_special_tokens": [
            config.image_token,
            config.video_token,
            config.image_start_token,
            config.image_end_token,
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    # Resize embeddings if tokens were added
    if len(tokenizer) > text_decoder.config.vocab_size:
        text_decoder.resize_token_embeddings(len(tokenizer))

    # Create processor for image preprocessing
    try:
        processor = AutoProcessor.from_pretrained(config.vision_encoder_name)
    except Exception:
        # Fallback: create basic processor
        from transformers import CLIPImageProcessor

        processor = CLIPImageProcessor.from_pretrained(config.vision_encoder_name)

    logger.info(f"Model initialized with {model.num_parameters():,} parameters")
    logger.info(f"Trainable parameters: {model.num_parameters(trainable_only=True):,}")

    return model, processor, tokenizer


def load_pretrained_smolvlm(
    model_name_or_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = "auto",
) -> Tuple[nn.Module, AutoProcessor, AutoTokenizer]:
    """Load pretrained SmolVLM2 from HuggingFace.

    Args:
        model_name_or_path: HF model ID or local path
        torch_dtype: Model precision
        device_map: Device placement

    Returns:
        Tuple of (model, processor, tokenizer)
    """
    from transformers import AutoModelForVision2Seq

    logger.info(f"Loading pretrained model: {model_name_or_path}")

    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return model, processor, tokenizer
