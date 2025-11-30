from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.weight_applier import WeightApplier
from mflux.models.common.weights.weight_loader import WeightLoader
from mflux.models.fibo.model.fibo_text_encoder import SmolLM3_3B_TextEncoder
from mflux.models.fibo.model.fibo_transformer import FiboTransformer
from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.fibo.tokenizer import FiboTokenizerHandler
from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition


class FIBOInitializer:
    @staticmethod
    def init(
        fibo_model,
        model_config: ModelConfig | None = None,
        quantize: int | None = None,
        local_path: str | None = None,
    ) -> None:
        # 1. Load weights using generic loader
        weights = WeightLoader.load(
            weight_definition=FIBOWeightDefinition,
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

        # 4. Apply weights and quantize
        fibo_model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            models={
                "vae": fibo_model.vae,
                "transformer": fibo_model.transformer,
                "text_encoder": fibo_model.text_encoder,
            },
            quantize_arg=quantize,
            weight_definition=FIBOWeightDefinition,
        )
