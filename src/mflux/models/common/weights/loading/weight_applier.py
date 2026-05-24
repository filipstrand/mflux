import inspect
from typing import TYPE_CHECKING

import mlx.nn as nn

from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType


class WeightApplier:
    @staticmethod
    def quantization_predicate_for_bits(quantization_predicate, bits: int | None):
        if quantization_predicate is None:
            return None

        try:
            parameters = inspect.signature(quantization_predicate).parameters
        except (TypeError, ValueError):
            return quantization_predicate

        if len(parameters) < 3:
            return quantization_predicate

        return lambda path, module: quantization_predicate(path, module, bits)

    @staticmethod
    def quantization_predicate_for_weights(
        weight_definition: "WeightDefinitionType",
        bits: int,
        weights: LoadedWeights | None = None,
    ):
        predicate_getter = getattr(weight_definition, "quantization_predicate_for_loaded_weights", None)
        if predicate_getter is not None:
            quantization_predicate = predicate_getter(weights=weights, bits=bits)
        else:
            quantization_predicate = weight_definition.quantization_predicate

        return WeightApplier.quantization_predicate_for_bits(quantization_predicate, bits)

    @staticmethod
    def apply_and_quantize_single(
        weights: LoadedWeights,
        model: nn.Module,
        component: ComponentDefinition,
        quantize_arg: int | None,
        quantization_predicate=None,
    ) -> int | None:
        stored_q = weights.meta_data.quantization_level
        component_weights = weights.components.get(component.name)

        if component_weights is None:
            raise ValueError(f"No weights found for component: {component.name}")

        if quantization_predicate is None:

            def quantization_predicate(path, module):
                return hasattr(module, "to_quantized")

        bits, warning = QuantizationResolution.resolve(stored=stored_q, requested=quantize_arg)

        if warning:
            print(f"⚠️  {warning}")

        quantization_predicate = WeightApplier.quantization_predicate_for_bits(quantization_predicate, bits)

        if bits is None:
            model.update(component_weights, strict=False)
        elif stored_q is None:
            model.update(component_weights, strict=False)
            if not component.skip_quantization:
                nn.quantize(model, class_predicate=quantization_predicate, bits=bits)
        else:
            if not component.skip_quantization:
                nn.quantize(model, class_predicate=quantization_predicate, bits=bits)
            model.update(component_weights, strict=False)

        return bits

    @staticmethod
    def apply_and_quantize(
        weights: LoadedWeights,
        models: dict[str, nn.Module],
        quantize_arg: int | None,
        weight_definition: "WeightDefinitionType",
    ) -> int | None:
        stored_q = weights.meta_data.quantization_level
        components = {c.name: c for c in weight_definition.get_components()}

        bits, warning = QuantizationResolution.resolve(stored=stored_q, requested=quantize_arg)

        if warning:
            print(f"⚠️  {warning}")

        if bits is None:
            WeightApplier._set_weights(weights, models, components)
        elif stored_q is None:
            WeightApplier._set_weights(weights, models, components)
            WeightApplier._quantize(models, bits, components, weight_definition)
        else:
            WeightApplier._quantize(models, bits, components, weight_definition, weights=weights)
            WeightApplier._set_weights(weights, models, components)

        return bits

    @staticmethod
    def _set_weights(
        weights: LoadedWeights,
        models: dict[str, nn.Module],
        components: dict | None = None,
    ) -> None:
        for name, model in models.items():
            component_weights = weights.components.get(name)
            if component_weights is not None:
                if components is not None:
                    component = components.get(name)
                    if component is not None and component.weight_subkey is not None:
                        component_weights = component_weights.get(component.weight_subkey, component_weights)
                model.update(component_weights, strict=False)

    @staticmethod
    def _quantize(
        models: dict[str, nn.Module],
        bits: int,
        components: dict,
        weight_definition: "WeightDefinitionType",
        weights: LoadedWeights | None = None,
    ) -> None:
        quantization_predicate = WeightApplier.quantization_predicate_for_weights(
            weight_definition=weight_definition,
            bits=bits,
            weights=weights,
        )
        for name, model in models.items():
            component = components.get(name)
            if component and component.skip_quantization:
                continue
            nn.quantize(model, class_predicate=quantization_predicate, bits=bits)
