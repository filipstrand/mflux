from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.weight_definition import ComponentDefinition
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping


class FluxWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                mapping_getter=FluxWeightMapping.get_vae_mapping,
                precision=ModelConfig.precision,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                mapping_getter=FluxWeightMapping.get_transformer_mapping,
                precision=ModelConfig.precision,
            ),
            ComponentDefinition(
                name="t5_encoder",
                hf_subdir="text_encoder_2",
                mapping_getter=FluxWeightMapping.get_t5_encoder_mapping,
                num_blocks=24,
                precision=ModelConfig.precision,
            ),
            ComponentDefinition(
                name="clip_encoder",
                hf_subdir="text_encoder",
                mapping_getter=FluxWeightMapping.get_clip_encoder_mapping,
                precision=ModelConfig.precision,
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "text_encoder_2/*.safetensors",
            "text_encoder_2/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "vae/*.safetensors",
            "vae/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        # Quantize everything for Flux
        return hasattr(module, "to_quantized")
