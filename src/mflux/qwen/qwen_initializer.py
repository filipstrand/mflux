from mflux.config.model_config import ModelConfig
from mflux.models.vae.qwen_vae import QwenImageVAE
from mflux.qwen.qwen_transformer_full import QwenImageTransformerMLX
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
        qwen_model.model_config = model_config

        weights = QwenImageWeightHandler.load_pretrained_weights(
            repo_id=model_config.model_name,
            local_path=local_path,
        )

        qwen_model.vae = QwenImageVAE()
        qwen_model.transformer = QwenImageTransformerMLX()
        # TODO: Initialize text encoder when implemented

        # Apply weights and quantization (like Flux)
        qwen_model.bits = QwenWeightUtil.set_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights,
            vae=qwen_model.vae,
            transformer=qwen_model.transformer,
        )
