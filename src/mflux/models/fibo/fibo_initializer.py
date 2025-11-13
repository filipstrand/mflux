"""FIBO model initializer.

Loads weights and initializes the FIBO model components.
"""

from mflux.models.fibo.model.fibo_vae.vae import VAE
from mflux.models.fibo.weights.fibo_weight_handler import FIBOWeightHandler


class FIBOInitializer:
    """Initializer for FIBO model."""

    @staticmethod
    def init(
        fibo_model,
        quantize: int | None = None,
        local_path: str | None = None,
    ) -> None:
        """Initialize FIBO model with weights.

        Args:
            fibo_model: FIBO model instance to initialize
            quantize: Quantization bits (not implemented yet)
            local_path: Local path to model weights (optional)
        """
        # 1. Load VAE weights
        weights = FIBOWeightHandler.load_regular_weights(
            repo_id="briaai/FIBO",  # Default FIBO model
            local_path=local_path,
        )

        # 2. Initialize VAE model
        fibo_model.vae = VAE()

        # 3. Apply weights to VAE
        # The weights from FIBOWeightHandler are already in the correct nested MLX structure
        # (including resample Conv2d weights via FIBOWeightMapping). MLX's model.update()
        # accepts nested dictionaries directly.
        if weights.vae:
            fibo_model.vae.update(weights.vae, strict=False)

        # 4. Store quantization level (not implemented yet)
        fibo_model.bits = quantize
        fibo_model.local_path = local_path

        # 5. Optionally quantize the model (not implemented yet)
        if quantize:
            # TODO: Implement quantization for FIBO VAE
            pass
