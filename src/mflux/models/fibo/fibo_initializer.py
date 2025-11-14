from mflux.models.fibo.model.fibo_vae.vae import VAE
from mflux.models.fibo.weights.fibo_weight_handler import FIBOWeightHandler


class FIBOInitializer:
    @staticmethod
    def init(
        fibo_model,
        quantize: int | None = None,
        local_path: str | None = None,
    ) -> None:
        # 1. Load VAE weights
        weights = FIBOWeightHandler.load_regular_weights(
            repo_id="briaai/FIBO",
            local_path=local_path,
        )

        # 2. Initialize VAE model
        fibo_model.vae = VAE()

        # 3. Apply weights to VAE
        if weights.vae:
            fibo_model.vae.update(weights.vae, strict=False)

        # 4. Store quantization level (not implemented yet)
        fibo_model.bits = quantize
        fibo_model.local_path = local_path

        # 5. Optionally quantize the model (not implemented yet)
        if quantize:
            pass
