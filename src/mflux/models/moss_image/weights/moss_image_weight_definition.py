from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.moss_image.weights.moss_image_weight_mapping import MOSSImageWeightMapping


class MOSSImageWeightDefinition:
    # Loads OpenMOSS-Team/MOSS-Image-SFT in its published layout, or a checkpoint written by mflux-save.
    MAX_SEQUENCE_LENGTH = 4096

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=MOSSImageWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=MOSSImageWeightMapping.get_transformer_mapping,
                # The RoPE frequency bands are derived constants. The model computes them in float32.
                key_transform=lambda key: None if key == "rope.bands" else key,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                mapping_getter=MOSSImageWeightMapping.get_text_encoder_mapping,
                # Text-only use: the vision tower is not loaded.
                weight_prefix_filters=["language_model."],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3_5",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=MOSSImageWeightDefinition.MAX_SEQUENCE_LENGTH,
                padding="do_not_pad",
                use_chat_template=True,
                chat_template_kwargs={"enable_thinking": True},
                download_patterns=[
                    "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/chat_template.jinja",
                ],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/*",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")
