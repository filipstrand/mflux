from typing import List, Optional

import mlx.nn as nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import (
    ComponentDefinition,
    TokenizerDefinition,
)
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

# Base model names that use Qwen3 (single encoder, no T5 / text_encoder_2)
KLEIN_MODELS = {
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
}


class FluxWeightDefinition:
    @staticmethod
    def get_components(model_config: Optional[ModelConfig] = None) -> List[ComponentDefinition]:
        is_klein = model_config is not None and model_config.is_klein()
        components = [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=FluxWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=FluxWeightMapping.get_transformer_mapping,
            ),
        ]
        if not is_klein:
            components.append(
                ComponentDefinition(
                    name="t5_encoder",
                    hf_subdir="text_encoder_2",
                    model_attr="t5_text_encoder",
                    num_blocks=24,
                    precision=ModelConfig.precision,
                    mapping_getter=FluxWeightMapping.get_t5_encoder_mapping,
                )
            )
        components.append(
            ComponentDefinition(
                name="clip_encoder",
                hf_subdir="text_encoder",
                model_attr="clip_text_encoder",
                precision=ModelConfig.precision,
                mapping_getter=FluxWeightMapping.get_clip_encoder_mapping,
            )
        )
        return components

    @staticmethod
    def get_tokenizers(model_config: Optional[ModelConfig] = None) -> List[TokenizerDefinition]:
        is_klein = model_config is not None and model_config.is_klein()
        tokenizers = [
            TokenizerDefinition(
                name="clip",
                hf_subdir="tokenizer",
                tokenizer_class="CLIPTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=77,
                download_patterns=["tokenizer/**"],
            ),
        ]
        if not is_klein:
            tokenizers.append(
                TokenizerDefinition(
                    name="t5",
                    hf_subdir="tokenizer_2",
                    tokenizer_class="T5Tokenizer",
                    encoder_class=LanguageTokenizer,
                    max_length=256,  # Will be overridden by model_config.max_sequence_length
                    download_patterns=["tokenizer_2/**"],
                )
            )
        return tokenizers

    @staticmethod
    def get_download_patterns(model_config: Optional[ModelConfig] = None) -> List[str]:
        is_klein = model_config is not None and model_config.is_klein()
        patterns = [
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "vae/*.safetensors",
            "vae/*.json",
        ]
        if not is_klein:
            patterns += [
                "text_encoder_2/*.safetensors",
                "text_encoder_2/*.json",
            ]
        return patterns

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")


class FluxControlnetWeightDefinition:
    @staticmethod
    def get_controlnet_component() -> ComponentDefinition:
        return ComponentDefinition(
            name="transformer_controlnet",
            hf_subdir="",
            loading_mode="single",
            precision=ModelConfig.precision,
            mapping_getter=FluxWeightMapping.get_controlnet_transformer_mapping,
        )

    @staticmethod
    def get_components(model_config: Optional[ModelConfig] = None) -> List[ComponentDefinition]:
        return FluxWeightDefinition.get_components(model_config) + [
            ComponentDefinition(
                name="transformer_controlnet",
                hf_subdir="transformer_controlnet",
                precision=ModelConfig.precision,
                mapping_getter=FluxWeightMapping.get_transformer_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers(model_config: Optional[ModelConfig] = None) -> List[TokenizerDefinition]:
        return FluxWeightDefinition.get_tokenizers(model_config)

    @staticmethod
    def get_download_patterns(model_config: Optional[ModelConfig] = None) -> List[str]:
        return FluxWeightDefinition.get_download_patterns(model_config)

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return FluxWeightDefinition.quantization_predicate(path, module)


class FluxReduxWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="siglip",
                hf_subdir="image_encoder",
                mapping_getter=None,
                model_attr="image_encoder",
                precision=ModelConfig.precision,
                bulk_transform=WeightTransforms.transpose_conv2d_weight,
                weight_subkey="vision_model",
            ),
            ComponentDefinition(
                name="redux_encoder",
                hf_subdir="image_embedder",
                mapping_getter=None,
                model_attr="image_embedder",
                precision=ModelConfig.precision,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "image_encoder/*.safetensors",
            "image_encoder/config.json",
            "image_embedder/*.safetensors",
            "image_embedder/config.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        if isinstance(module, nn.Conv2d):
            return False
        if hasattr(module, "weight") and hasattr(module.weight, "shape"):
            if module.weight.shape == (1152, 4304):
                return False
            if module.weight.shape[-1] % 64 != 0:
                return False
        return hasattr(module, "to_quantized")
