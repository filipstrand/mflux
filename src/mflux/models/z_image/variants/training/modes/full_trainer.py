from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mflux.models.z_image.variants.training.state.training_spec import FullFinetuneSpec
    from mflux.models.z_image.variants.training.z_image_base import ZImageBase

# Memory estimation constants (in GB) for Z-Image 6B model
# Updated 2026-01 for Z-Image-Base 6B parameter model
MODEL_BASE_MEMORY_GB = 12.0  # 6B params at bf16 (2 bytes per param)
GRADIENTS_MEMORY_GB = 24.0  # Same size as model at fp32
OPTIMIZER_STATE_MEMORY_GB = 24.0  # AdamW momentum + variance
PER_IMAGE_ACTIVATIONS_GB = 18.0  # Conservative estimate including intermediate tensors
FIXED_OVERHEAD_GB = MODEL_BASE_MEMORY_GB + GRADIENTS_MEMORY_GB + OPTIMIZER_STATE_MEMORY_GB  # 60 GB


class FullTrainer:
    """Full fine-tuning utilities for Z-Image.

    Full fine-tuning:
    - Unfreezes transformer weights (6B parameters)
    - Optionally unfreezes VAE and text encoder
    - Requires more memory but achieves deeper adaptation
    - Best for domain-specific heavy customization

    With 512GB unified memory, you can:
    - Train full transformer at batch_size=8-16
    - Use gradient accumulation for larger effective batches
    """

    @staticmethod
    def setup_for_training(model: "ZImageBase", spec: "FullFinetuneSpec") -> None:
        """Setup model for full fine-tuning.

        Unfreezes components based on specification.
        """
        # Start by freezing everything
        model.freeze()

        # Unfreeze transformer if specified
        if spec.train_transformer:
            model.transformer.unfreeze()

        # Unfreeze VAE if specified (rare)
        if spec.train_vae:
            model.vae.unfreeze()

        # Unfreeze text encoder if specified (rare)
        if spec.train_text_encoder:
            model.text_encoder.unfreeze()

    @staticmethod
    def count_trainable_parameters(model: "ZImageBase") -> tuple[int, int]:
        """Count total parameters in the model.

        Note: This returns total parameter count for estimation purposes.
        MLX doesn't expose a simple way to check frozen state per-parameter,
        so this assumes all parameters are counted. The actual trainable
        count depends on which components were unfrozen in setup_for_training().

        Returns:
            (total_params, total_params) - Both values are the same for estimation
        """
        total = 0

        for name, param in model.named_parameters():
            total += param.size

        return total, total

    @staticmethod
    def print_training_info(model: "ZImageBase", spec: "FullFinetuneSpec") -> None:
        """Print information about the training setup."""
        print("\n=== Full Fine-tuning Setup ===")
        print(f"Training transformer: {spec.train_transformer}")
        print(f"Training VAE: {spec.train_vae}")
        print(f"Training text encoder: {spec.train_text_encoder}")
        print(f"Gradient checkpointing: {spec.gradient_checkpointing}")
        print()

        # Estimate memory usage using module constants
        if spec.train_transformer:
            print("Estimated memory usage (transformer only):")
            print(f"  - Base model: ~{MODEL_BASE_MEMORY_GB:.0f} GB")
            print(f"  - Gradients: ~{GRADIENTS_MEMORY_GB:.0f} GB")
            print(f"  - Optimizer state: ~{OPTIMIZER_STATE_MEMORY_GB:.0f} GB")
            print("  - Activations (batch=8): ~40-60 GB")
            print("  - Total estimate: ~100-120 GB")
            print()
            print("Your 512GB memory can handle batch_size=8-16 comfortably!")

        print("=" * 30)

    @staticmethod
    def estimate_optimal_batch_size(available_memory_gb: float = 512) -> int:
        """Estimate optimal batch size based on available memory.

        Uses module-level constants for memory estimates.

        Args:
            available_memory_gb: Available unified memory in GB

        Returns:
            Recommended batch size
        """
        available_for_batches = available_memory_gb - FIXED_OVERHEAD_GB
        optimal_batch = int(available_for_batches / PER_IMAGE_ACTIVATIONS_GB)

        # Cap at reasonable maximum
        return min(max(optimal_batch, 1), 32)
