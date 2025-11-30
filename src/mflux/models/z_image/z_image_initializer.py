from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.tokenizer.tokenizer import Tokenizer
from mflux.models.z_image.weights.weight_handler import WeightHandler
from mflux.models.z_image.weights.weight_util import WeightUtil
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping


class ZImageInitializer:
    @staticmethod
    def init(
        z_image_model,
        model_config: ModelConfig,
        quantize: int | None,
        local_path: str | None,
        load_text_encoder: bool = True,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        # 0. Store config for later use
        z_image_model.model_config = model_config

        # 1. Load the weights
        weights = WeightHandler.load_weights(
            repo_id=model_config.model_name,
            local_path=local_path,
            load_text_encoder=load_text_encoder,
        )

        # 2. Initialize models
        z_image_model.vae = VAE()
        z_image_model.transformer = ZImageTransformer()

        # 3. Initialize text encoder and tokenizer if requested
        if load_text_encoder:
            z_image_model.text_encoder = TextEncoder()
            z_image_model.tokenizer = Tokenizer.from_pretrained(
                repo_id=model_config.model_name,
                local_path=local_path,
            )
        else:
            z_image_model.text_encoder = None
            z_image_model.tokenizer = None

        # 4. Apply weights and quantize
        z_image_model.bits = WeightUtil.set_weights_and_quantize(
            quantize_arg=quantize,
            weights=weights,
            vae=z_image_model.vae,
            transformer=z_image_model.transformer,
            text_encoder=z_image_model.text_encoder,
        )

        # 5. Apply LoRA weights if provided
        z_image_model.lora_paths, z_image_model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=ZImageLoRAMapping.get_mapping(),
            transformer=z_image_model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
