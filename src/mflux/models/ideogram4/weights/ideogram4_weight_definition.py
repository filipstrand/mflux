from typing import List

import mlx.core as mx

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
from mflux.models.ideogram4.tokenizer import Ideogram4Tokenizer


class Ideogram4WeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                num_blocks=4,
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="conditional_transformer",
                hf_subdir="transformer",
                loading_mode="fp8_safetensors",
                skip_quantization=True,
                weight_transform=Ideogram4WeightDefinition.prepare_tensor,
            ),
            ComponentDefinition(
                name="unconditional_transformer",
                hf_subdir="unconditional_transformer",
                loading_mode="fp8_safetensors",
                skip_quantization=True,
                weight_transform=Ideogram4WeightDefinition.prepare_tensor,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="fp8_safetensors",
                skip_quantization=True,
                weight_prefix_filters=["language_model."],
                key_transform=Ideogram4WeightDefinition.transform_text_encoder_key,
                weight_transform=Ideogram4WeightDefinition.prepare_tensor,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="ideogram4",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=Ideogram4Tokenizer,
                max_length=512,
                add_special_tokens=False,
                download_patterns=["tokenizer/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "model_index.json",
            "scheduler/*.json",
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "unconditional_transformer/*.safetensors",
            "unconditional_transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/**",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")

    @staticmethod
    def prepare_tensor(key: str, value: mx.array) -> mx.array:
        if value.dtype == mx.uint8 or key.endswith(".weight_scale"):
            return value
        if value.dtype in (mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8, mx.bool_):
            return value
        return value.astype(ModelConfig.precision)

    @staticmethod
    def transform_text_encoder_key(key: str) -> str | None:
        if not key.startswith("language_model."):
            return None
        mapped = key[len("language_model.") :]
        if mapped.startswith(("embed_tokens.", "layers.", "norm.")):
            return mapped
        return None
