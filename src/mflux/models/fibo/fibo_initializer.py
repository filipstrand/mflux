from mflux.config.model_config import ModelConfig
from mflux.models.fibo.model.fibo_text_encoder import SmolLM3_3B_TextEncoder
from mflux.models.fibo.model.fibo_transformer import FiboTransformer
from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.fibo.tokenizer import FiboTokenizerHandler
from mflux.models.fibo.weights.fibo_weight_handler import FIBOWeightHandler
from mflux.models.fibo.weights.fibo_weight_util import FIBOWeightUtil


class FIBOInitializer:
    @staticmethod
    def init(
        fibo_model,
        model_config: ModelConfig | None = None,
        quantize: int | None = None,
        local_path: str | None = None,
    ) -> None:
        # 1. Load VAE weights
        weights = FIBOWeightHandler.load_regular_weights(
            repo_id=model_config.model_name,
            local_path=local_path,
        )

        # 2. Initialize tokenizers
        tokenizer_handler = FiboTokenizerHandler(
            repo_id=model_config.model_name,
            local_path=local_path,
        )
        fibo_model.fibo_tokenizer = tokenizer_handler.fibo

        # 3. Initialize all models
        fibo_model.vae = Wan2_2_VAE()
        fibo_model.text_encoder = SmolLM3_3B_TextEncoder()
        fibo_model.transformer = FiboTransformer(
            in_channels=48,
            num_layers=8,
            num_single_layers=38,
        )

        # 4. Apply weights and quantize VAE, transformer, and text encoder
        fibo_model.bits = FIBOWeightUtil.set_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights,
            vae=fibo_model.vae,
            transformer=fibo_model.transformer,
            text_encoder=fibo_model.text_encoder,
        )
