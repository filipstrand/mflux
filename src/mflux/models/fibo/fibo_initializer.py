from mflux.models.fibo.model.fibo_transformer import FiboTransformer
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

        # 4. Initialize transformer model (mirror diffusers BriaFiboTransformer2DModel config)
        fibo_model.transformer = FiboTransformer(
            in_channels=48,
            num_layers=8,
            num_single_layers=38,
        )

        # 5. Apply weights to transformer
        if getattr(weights, "transformer", None):
            fibo_model.transformer.update(weights.transformer, strict=False)

        # 6. Store quantization level (not implemented yet)
        fibo_model.bits = quantize
        fibo_model.local_path = local_path

        # 7. Optionally quantize the model (not implemented yet)
        if quantize:
            pass
