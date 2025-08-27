from mflux.config.model_config import ModelConfig
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.tokenizer.qwen_tokenizer_handler import QwenTokenizerHandler
from mflux.weights.qwen_weight_handler import QwenImageWeightHandler
from mflux.weights.qwen_weight_util import QwenWeightUtil


class QwenImageInitializer:
    @staticmethod
    def init(
        qwen_model,
        model_config: ModelConfig,
        quantize: int | None,
        local_path: str | None,
    ) -> None:
        # 0. Set paths, configs, and prompt_cache for later
        qwen_model.prompt_cache = {}
        qwen_model.model_config = model_config

        # 1. Load the regular weights
        weights = QwenImageWeightHandler.load_pretrained_weights(
            repo_id=model_config.model_name,
            local_path=local_path,
        )

        # 2. Initialize tokenizers
        tokenizer_handler = QwenTokenizerHandler(
            repo_id=model_config.model_name,
            local_path=local_path,
        )
        qwen_model.qwen_tokenizer = tokenizer_handler.qwen

        # 3. Initialize all models
        qwen_model.vae = QwenVAE()
        qwen_model.transformer = QwenTransformer()
        qwen_model.text_encoder = QwenTextEncoder()

        # 4. Apply weights and quantize the models
        qwen_model.bits = QwenWeightUtil.set_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights,
            vae=qwen_model.vae,
            transformer=qwen_model.transformer,
            text_encoder=qwen_model.text_encoder,
        )
