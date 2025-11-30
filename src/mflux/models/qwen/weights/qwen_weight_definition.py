from typing import List

import mlx.core as mx

from mflux.models.common.weights.weight_definition import ComponentDefinition
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class QwenWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                mapping_getter=QwenWeightMapping.get_vae_mapping,
                loading_mode="single",
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                mapping_getter=QwenWeightMapping.get_transformer_mapping,
                loading_mode="multi_glob",
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                mapping_getter=QwenWeightMapping.get_text_encoder_mapping,
                loading_mode="multi_json",
                precision=mx.bfloat16,  # Text encoder uses BF16 for numerical consistency
                skip_quantization=True,  # Quantization causes significant semantic degradation
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
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        # Quantize everything except text encoder (handled by skip_quantization)
        return hasattr(module, "to_quantized")
