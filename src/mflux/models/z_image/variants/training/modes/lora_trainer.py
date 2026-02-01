from typing import TYPE_CHECKING

from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

if TYPE_CHECKING:
    from mflux.models.z_image.variants.training.z_image_base import ZImageBase


class LoRATrainer:
    """LoRA-specific training utilities.

    LoRA (Low-Rank Adaptation) training:
    - Freezes all base model weights
    - Only trains the LoRA adapter matrices (A and B)
    - Efficient for fine-tuning with limited data
    - Memory efficient as most weights don't need gradients
    """

    @staticmethod
    def setup_for_training(model: "ZImageBase") -> None:
        """Setup model for LoRA training.

        1. Freeze the entire model
        2. Unfreeze only LoRA layers (lora_A, lora_B matrices)
        """
        # Freeze everything
        model.freeze()

        # Unfreeze LoRA layers in transformer
        LoRATrainer._unfreeze_lora_layers(model.transformer)

    @staticmethod
    def _unfreeze_lora_layers(module: nn.Module) -> None:
        """Recursively find and unfreeze LoRA layers."""
        for name, child in module.named_modules():
            if isinstance(child, LoRALinear):
                child.unfreeze(keys=["lora_A", "lora_B"], strict=False)

    @staticmethod
    def count_trainable_parameters(model: "ZImageBase") -> tuple[int, int]:
        """Count trainable and total parameters.

        Uses MLX's tree_flatten to iterate over parameters.

        Returns:
            (trainable_params, total_params)
        """
        trainable = 0
        total = 0

        # Use tree_flatten to get (name, param) pairs from model parameters
        for name, param in tree_flatten(model.parameters()):
            param_count = param.size
            total += param_count
            if "lora_A" in name or "lora_B" in name:
                trainable += param_count

        return trainable, total

    @staticmethod
    def print_training_info(model: "ZImageBase") -> None:
        """Print information about the training setup."""
        trainable, total = LoRATrainer.count_trainable_parameters(model)
        percentage = (trainable / total) * 100 if total > 0 else 0

        print("\n=== LoRA Training Setup ===")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Training percentage: {percentage:.2f}%")
        print(f"Memory savings: ~{(1 - percentage / 100) * 100:.1f}%")
        print("=" * 30)
