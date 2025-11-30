from typing import TYPE_CHECKING

import mlx.nn as nn

from mflux.models.common.weights.loaded_weights import LoadedWeights

if TYPE_CHECKING:
    from mflux.models.common.weights.weight_definition import WeightDefinitionType


class WeightApplier:
    @staticmethod
    def apply_and_quantize(
        weights: LoadedWeights,
        models: dict[str, nn.Module],
        quantize_arg: int | None,
        weight_definition: "WeightDefinitionType",
    ) -> int | None:
        stored_q = weights.meta_data.quantization_level
        components = {c.name: c for c in weight_definition.get_components()}

        # Case 1: No quantization
        if stored_q is None and quantize_arg is None:
            WeightApplier._set_weights(weights, models)
            return None

        # Case 2: On-the-fly quantization
        if stored_q is None and quantize_arg is not None:
            WeightApplier._set_weights(weights, models)
            WeightApplier._quantize(models, quantize_arg, components, weight_definition)
            return quantize_arg

        # Case 3: Pre-quantized
        if stored_q is not None:
            WeightApplier._quantize(models, stored_q, components, weight_definition)
            WeightApplier._set_weights(weights, models)
            return stored_q

        raise ValueError("Error setting weights: unexpected quantization state")

    @staticmethod
    def _set_weights(weights: LoadedWeights, models: dict[str, nn.Module]) -> None:
        for name, model in models.items():
            component_weights = weights.components.get(name)
            if component_weights is not None:
                model.update(component_weights, strict=False)

    @staticmethod
    def _quantize(
        models: dict[str, nn.Module],
        bits: int,
        components: dict,
        weight_definition: "WeightDefinitionType",
    ) -> None:
        for name, model in models.items():
            component = components.get(name)
            if component and component.skip_quantization:
                continue  # Skip this component (e.g., Qwen text encoder)
            nn.quantize(model, class_predicate=weight_definition.quantization_predicate, bits=bits)
