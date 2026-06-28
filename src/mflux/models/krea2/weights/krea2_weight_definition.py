from typing import List

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.krea2.weights.krea2_weight_mapping import Krea2WeightMapping

# The Krea 2 VAE is the Qwen-Image autoencoder, distributed separately (krea-2-turbo/vae has only a
# config). It is fetched from this repo by the initializer.
KREA2_VAE_REPO = "Qwen/Qwen-Image"


class Krea2WeightDefinition:
    """Weight definition for Krea 2 Turbo.

    The transformer ships in the original Krea naming (blocks/txtfusion/first/last/...) and is loaded
    via the index json (multi_json). The text encoder is a Qwen3-VL checkpoint; only the
    `language_model.*` keys are used (text-only conditioning). The VAE is fetched separately.
    """

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                precision=ModelConfig.precision,
                num_blocks=28,
                mapping_getter=Krea2WeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="single",
                precision=mx.bfloat16,
                skip_quantization=True,  # like qwen: quantizing the TE degrades semantics
                num_layers=36,
                weight_prefix_filters=["language_model."],
                mapping_getter=Krea2WeightMapping.get_text_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_vae_component() -> ComponentDefinition:
        return ComponentDefinition(
            name="vae",
            hf_subdir="vae",
            loading_mode="single",
            precision=ModelConfig.precision,
            mapping_getter=Krea2WeightMapping.get_vae_mapping,
        )

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen",
                hf_subdir="tokenizer",
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=LanguageTokenizer,
                max_length=512,
                padding="max_length",
                download_patterns=["tokenizer/**", "added_tokens.json", "chat_template.jinja"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/**",
            "added_tokens.json",
            "chat_template.jinja",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        # The text-fusion projector is a tiny Linear (in_features=12) whose last weight dim is not
        # divisible by the quantization group size; keep it unquantized.
        weight = getattr(module, "weight", None)
        if weight is not None and weight.ndim == 2 and weight.shape[-1] % 64 != 0:
            return False
        return True
