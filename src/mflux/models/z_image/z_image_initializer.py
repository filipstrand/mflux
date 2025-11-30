from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.weights.weight_applier import WeightApplier
from mflux.models.common.weights.weight_loader import WeightLoader
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.tokenizer.tokenizer import Tokenizer
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition


class ZImageInitializer:
    @staticmethod
    def init(
        z_image_model,
        model_config: ModelConfig,
        quantize: int | None,
        local_path: str | None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        # 0. Store config for later use
        z_image_model.model_config = model_config

        # 1. Load weights using generic loader
        weights = WeightLoader.load(
            weight_definition=ZImageWeightDefinition,
            repo_id=model_config.model_name,
            local_path=local_path,
        )

        # 2. Initialize models
        z_image_model.vae = VAE()
        z_image_model.transformer = ZImageTransformer()
        z_image_model.text_encoder = TextEncoder()
        z_image_model.tokenizer = Tokenizer.from_pretrained(
            repo_id=model_config.model_name,
            local_path=local_path,
        )

        # 3. Apply weights and quantize
        z_image_model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            models={
                "vae": z_image_model.vae,
                "transformer": z_image_model.transformer,
                "text_encoder": z_image_model.text_encoder,
            },
            quantize_arg=quantize,
            weight_definition=ZImageWeightDefinition,
        )

        # 4. Apply LoRA weights if provided
        z_image_model.lora_paths, z_image_model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=ZImageLoRAMapping.get_mapping(),
            transformer=z_image_model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
