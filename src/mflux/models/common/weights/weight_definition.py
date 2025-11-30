from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, TypeAlias

import mlx.core as mx

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget

if TYPE_CHECKING:
    from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
    from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

    WeightDefinitionType: TypeAlias = type[
        FluxWeightDefinition | FIBOWeightDefinition | QwenWeightDefinition | ZImageWeightDefinition
    ]


@dataclass
class ComponentDefinition:
    name: str
    hf_subdir: str
    mapping_getter: Callable[[], List[WeightTarget]]
    num_blocks: int | None = None
    num_layers: int | None = None
    loading_mode: str = "mlx_native"
    precision: mx.Dtype | None = None
    skip_quantization: bool = False
