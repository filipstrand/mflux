from typing import List

import mlx.nn as nn

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.depth_pro.weights.depth_pro_weight_mapping import DepthProWeightMapping


class DepthProWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="depth_pro",
                hf_subdir="",
                loading_mode="torch_checkpoint",
                mapping_getter=DepthProWeightMapping.get_mapping,
                download_url="https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt",
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return []

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        if isinstance(module, nn.Conv2d):
            return False
        return hasattr(module, "to_quantized")
