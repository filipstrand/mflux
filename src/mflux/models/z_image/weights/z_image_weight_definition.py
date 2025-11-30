from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.weight_definition import ComponentDefinition
from mflux.models.z_image.weights.z_image_weight_mapping import ZImageWeightMapping


class ZImageWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                mapping_getter=ZImageWeightMapping.get_vae_mapping,
                num_blocks=4,
                precision=ModelConfig.precision,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                mapping_getter=ZImageWeightMapping.get_transformer_mapping,
                num_layers=30,
                precision=ModelConfig.precision,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                mapping_getter=ZImageWeightMapping.get_text_encoder_mapping,
                num_layers=36,
                precision=ModelConfig.precision,
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
        # Quantize everything for Z-Image
        return hasattr(module, "to_quantized")
