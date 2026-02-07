"""Custom trainer for SmolVLM2.

Extends HuggingFace Trainer with:
- Multi-modal batch handling
- Vision encoder unfreezing schedule
- Gradient checkpointing for memory efficiency
"""

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SmolVLMTrainingArguments(TrainingArguments):
    """Extended training arguments for SmolVLM2."""

    # Vision encoder settings
    freeze_vision_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze vision encoder initially"},
    )
    unfreeze_vision_after_steps: int = field(
        default=10000,
        metadata={"help": "Steps after which to unfreeze vision encoder"},
    )

    # Image/Video settings
    max_image_patches: int = field(
        default=36,
        metadata={"help": "Maximum image patches for large images"},
    )
    max_video_frames: int = field(
        default=32,
        metadata={"help": "Maximum video frames"},
    )

    # Memory optimization
    use_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing"},
    )
    vision_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Also checkpoint vision encoder"},
    )


class SmolVLMTrainer(Trainer):
    """Trainer for SmolVLM2 with multi-modal support."""

    def __init__(
        self,
        model: nn.Module,
        args: SmolVLMTrainingArguments,
        vision_encoder: Optional[nn.Module] = None,
        **kwargs,
    ):
        """Initialize trainer.

        Args:
            model: Full SmolVLM2 model or just the language model
            args: Training arguments
            vision_encoder: Optional separate vision encoder reference
            **kwargs: Additional Trainer arguments
        """
        super().__init__(model=model, args=args, **kwargs)

        self.vision_encoder = vision_encoder
        self._vision_frozen = args.freeze_vision_encoder

        # Track unfreezing
        self._vision_unfrozen_at_step = None

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Single training step with vision unfreezing check.

        Args:
            model: The model
            inputs: Batch of inputs

        Returns:
            Loss tensor
        """
        # Check if we should unfreeze vision encoder
        if (
            self._vision_frozen
            and self.state.global_step >= self.args.unfreeze_vision_after_steps
        ):
            self._unfreeze_vision_encoder()

        return super().training_step(model, inputs)

    def _unfreeze_vision_encoder(self):
        """Unfreeze vision encoder parameters."""
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'vision_encoder'):
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = True

        self._vision_frozen = False
        self._vision_unfrozen_at_step = self.state.global_step

        logger.info(
            f"Vision encoder unfrozen at step {self.state.global_step}"
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple]:
        """Compute loss for multi-modal inputs.

        Args:
            model: The model
            inputs: Batch containing input_ids, pixel_values, etc.
            return_outputs: Whether to return model outputs

        Returns:
            Loss or (loss, outputs)
        """
        # Forward pass
        outputs = model(**inputs)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Manual loss computation for custom models
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            labels = inputs.get("labels")

            if labels is not None:
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            else:
                loss = torch.tensor(0.0, device=logits.device)

        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model.

        Handles moving tensors to correct device.

        Args:
            inputs: Input batch

        Returns:
            Prepared inputs
        """
        prepared = {}

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.args.device)
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    prepared[key] = [v.to(self.args.device) for v in value]
                else:
                    prepared[key] = value
            else:
                prepared[key] = value

        return prepared

    def get_train_dataloader(self):
        """Get training dataloader.

        Overridden to handle streaming datasets.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        from torch.utils.data import DataLoader

        # Check if streaming/iterable dataset
        if hasattr(self.train_dataset, '__iter__') and not hasattr(self.train_dataset, '__len__'):
            # Iterable dataset - don't use sampler
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return super().get_train_dataloader()

    def create_optimizer(self):
        """Create optimizer with parameter groups.

        Uses different learning rates for vision vs language.
        """
        if self.optimizer is not None:
            return self.optimizer

        # Group parameters
        vision_params = []
        language_params = []
        connector_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'vision' in name.lower() or 'encoder' in name.lower():
                vision_params.append(param)
            elif 'connector' in name.lower() or 'project' in name.lower():
                connector_params.append(param)
            else:
                language_params.append(param)

        # Different LRs per group
        optimizer_grouped_parameters = [
            {
                "params": vision_params,
                "lr": self.args.learning_rate * 0.1,  # Lower LR for vision
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": connector_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": language_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
        ]

        # Filter empty groups
        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if len(g["params"]) > 0
        ]

        optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        return self.optimizer

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Log training metrics.

        Adds custom metrics like vision encoder status.
        """
        # Add custom metrics
        logs["vision_frozen"] = 1.0 if self._vision_frozen else 0.0

        if self._vision_unfrozen_at_step is not None:
            logs["vision_unfrozen_at"] = self._vision_unfrozen_at_step

        super().log(logs, start_time)


def create_training_args(
    output_dir: str,
    model_size: str = "256m",
    stage: str = "vision",
    **overrides,
) -> SmolVLMTrainingArguments:
    """Create training arguments with sensible defaults.

    Args:
        output_dir: Directory for checkpoints
        model_size: "256m" or "500m"
        stage: "vision" or "video"
        **overrides: Override specific arguments

    Returns:
        Training arguments
    """
    # Base settings
    base_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 8 if stage == "vision" else 4,
        "gradient_accumulation_steps": 4 if stage == "vision" else 8,
        "learning_rate": 1e-4 if stage == "vision" else 2e-5,
        "warmup_steps": 1000 if stage == "vision" else 500,
        "max_steps": 50000 if stage == "vision" else 30000,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_steps": 1000,
        "save_total_limit": 3,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
        "report_to": ["wandb"],

        # Vision encoder settings
        "freeze_vision_encoder": stage == "vision",
        "unfreeze_vision_after_steps": 10000,

        # Optimizer
        "optim": "adamw_torch_fused",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.1,
    }

    # Merge with overrides
    base_args.update(overrides)

    return SmolVLMTrainingArguments(**base_args)
