from typing import List

import mlx.core as mx

from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.krea2.weights.krea2_weight_mapping import Krea2WeightMapping
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class Krea2DiffusersWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                mapping_getter=QwenWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                mapping_getter=None,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="mlx_native",
                precision=mx.bfloat16,
                skip_quantization=True,
                weight_subkey="encoder",
                weight_prefix_filters=["language_model"],
                num_blocks=36,
                mapping_getter=lambda: Krea2WeightMapping.get_text_encoder_mapping("language_model"),
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return Krea2WeightDefinition.get_tokenizers()

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return Krea2WeightDefinition.quantization_predicate(path, module)


class Krea2WeightDefinition:
    @staticmethod
    def resolve(model_config: ModelConfig):
        return Krea2DiffusersWeightDefinition

    @staticmethod
    def tokenizer_path_for(model_config: ModelConfig, model_path: str) -> str:
        return model_path

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return Krea2DiffusersWeightDefinition.get_components()

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="krea2",
                hf_subdir="tokenizer",
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=LanguageTokenizer,
                max_length=512,
                padding="max_length",
                download_patterns=[
                    "tokenizer/**",
                    "added_tokens.json",
                    "chat_template.jinja",
                    "special_tokens_map.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                ],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return Krea2DiffusersWeightDefinition.get_download_patterns()

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")
