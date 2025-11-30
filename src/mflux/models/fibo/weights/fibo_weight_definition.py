from typing import List

import mlx.nn as nn

from mflux.models.common.weights.weight_definition import ComponentDefinition
from mflux.models.fibo.weights.fibo_weight_mapping import FIBOWeightMapping


class FIBOWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                mapping_getter=FIBOWeightMapping.get_vae_mapping,
                num_blocks=4,
                loading_mode="torch_convert",  # FIBO uses torch for bfloat16 handling
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                mapping_getter=FIBOWeightMapping.get_transformer_mapping,
                num_blocks=38,
                num_layers=46,
                loading_mode="torch_convert",
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                mapping_getter=FIBOWeightMapping.get_text_encoder_mapping,
                num_blocks=36,
                loading_mode="torch_convert",
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
        # 1. Skip Conv2d layers
        if isinstance(module, nn.Conv2d):
            return False

        # 2. Skip any layer with incompatible dimensions
        if hasattr(module, "weight") and hasattr(module.weight, "shape"):
            if module.weight.shape == (1152, 4304):
                return False
            if module.weight.shape[-1] % 64 != 0:
                return False

        # Only quantize layers that have to_quantized method
        return hasattr(module, "to_quantized")
