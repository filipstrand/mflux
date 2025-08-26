from pathlib import Path

from mflux.config.model_config import ModelConfig
from mflux.models.text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.vae.qwen_vae import QwenImageVAE
from mflux.qwen.qwen_transformer_full import QwenImageTransformerMLX
from mflux.tokenizer.qwen_tokenizer_handler import QwenTokenizerHandler
from mflux.weights.download import snapshot_download
from mflux.weights.qwen_text_encoder_loader import QwenTextEncoderLoader
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
        qwen_model.text_encoder = QwenTextEncoder()

        # Initialize tokenizer
        tokenizer_handler = QwenTokenizerHandler(
            repo_id=model_config.model_name,
            local_path=local_path,
        )
        qwen_model.qwen_tokenizer = tokenizer_handler.qwen

        # Load text encoder weights
        if local_path:
            text_encoder_path = Path(local_path) / "text_encoder"
        else:
            # Download/get cached weights for text encoder
            cached_path = Path(
                snapshot_download(
                    repo_id=model_config.model_name,
                    allow_patterns=["text_encoder/*.safetensors", "text_encoder/*.json"],
                )
            )
            text_encoder_path = cached_path / "text_encoder"

        print(f"📝 Loading text encoder weights from: {text_encoder_path}")
        QwenTextEncoderLoader.apply_weights(qwen_model.text_encoder, text_encoder_path)

        # Apply weights and quantization (like Flux)
        qwen_model.bits = QwenWeightUtil.set_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights,
            vae=qwen_model.vae,
            transformer=qwen_model.transformer,
            text_encoder=qwen_model.text_encoder,
        )

        # Initialize prompt cache
        qwen_model.prompt_cache = {}
