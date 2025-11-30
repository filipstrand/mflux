from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.weights.weight_applier import WeightApplier
from mflux.models.common.weights.weight_loader import WeightLoader
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.tokenizer.qwen_tokenizer_handler import QwenTokenizerHandler
from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition


class QwenImageInitializer:
    @staticmethod
    def init(
        qwen_model,
        model_config: ModelConfig,
        quantize: int | None,
        local_path: str | None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        # 0. Set paths, configs, and prompt_cache for later
        qwen_model.prompt_cache = {}
        qwen_model.model_config = model_config

        # 1. Load weights using generic loader
        weights = WeightLoader.load(
            weight_definition=QwenWeightDefinition,
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

        # 4. Check if visual weights are present and create VisionTransformer if needed
        text_encoder_weights = weights.components.get("text_encoder")
        if text_encoder_weights is not None:
            has_visual_weights = "encoder" in text_encoder_weights and "visual" in text_encoder_weights["encoder"]
            if has_visual_weights and qwen_model.text_encoder.encoder.visual is None:
                qwen_model.text_encoder.encoder.visual = VisionTransformer(
                    patch_size=14,
                    temporal_patch_size=2,
                    in_channels=3,
                    embed_dim=1280,
                    depth=32,
                    num_heads=16,
                    mlp_ratio=2.671875,
                    hidden_size=qwen_model.text_encoder.encoder.hidden_size,
                    spatial_merge_size=2,
                )

        # 5. Apply weights and quantize
        qwen_model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            models={
                "vae": qwen_model.vae,
                "transformer": qwen_model.transformer,
                "text_encoder": qwen_model.text_encoder,
            },
            quantize_arg=quantize,
            weight_definition=QwenWeightDefinition,
        )

        # 6. Set LoRA weights
        qwen_model.lora_paths, qwen_model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=QwenLoRAMapping.get_mapping(),
            transformer=qwen_model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
