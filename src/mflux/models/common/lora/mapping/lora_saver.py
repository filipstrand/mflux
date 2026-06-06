import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


class LoRASaver:
    @staticmethod
    def bake_and_strip_lora(module: nn.Module) -> nn.Module:
        def _assign(parent, attr_name, idx, new_child):
            if parent is None:
                return
            if isinstance(parent, list) and idx is not None:
                parent[idx] = new_child
            elif isinstance(parent, dict) and attr_name is not None:
                parent[attr_name] = new_child
            elif attr_name is not None:
                setattr(parent, attr_name, new_child)

        def _bake_single(lora_layer: LoRALinear) -> nn.Module:
            return LoRASaver._bake_lora_into_linear(lora_layer.linear, lora_layer)

        def _bake_lokr(lokr_layer: LoKrLinear) -> nn.Module:
            return LoRASaver._bake_lokr_into_linear(lokr_layer.linear, lokr_layer)

        def _bake_fused(fused_layer: FusedLoRALinear) -> nn.Module:
            current = fused_layer.base_linear
            for lora in fused_layer.loras:
                if isinstance(lora, LoRALinear):
                    current = LoRASaver._bake_lora_into_linear(current, lora)
                elif isinstance(lora, LoKrLinear):
                    current = LoRASaver._bake_lokr_into_linear(current, lora)
            return current

        def _walk(obj, parent=None, attr_name=None, idx=None):
            # Replace wrappers first
            if isinstance(obj, FusedLoRALinear):
                new_child = _bake_fused(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child
            elif isinstance(obj, LoKrLinear):
                new_child = _bake_lokr(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child
            elif isinstance(obj, LoRALinear):
                new_child = _bake_single(obj)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child

            # Recurse into containers/modules
            if isinstance(obj, list):
                for i, child in enumerate(list(obj)):
                    _walk(child, obj, None, i)
            elif isinstance(obj, tuple):
                temp_list = list(obj)
                for i, child in enumerate(temp_list):
                    _walk(child, temp_list, None, i)
                if parent is not None:
                    _assign(parent, attr_name, idx, type(obj)(temp_list))
            elif isinstance(obj, dict):
                for key, child in list(obj.items()):
                    _walk(child, obj, key, None)
            elif isinstance(obj, nn.Module):
                for name, child in vars(obj).items():
                    if isinstance(child, (nn.Module, list, tuple, dict)):
                        _walk(child, obj, name, None)

        _walk(module, None, None, None)
        return module

    @staticmethod
    def _dense_weight(linear: nn.Linear | nn.QuantizedLinear) -> mx.array:
        if isinstance(linear, nn.QuantizedLinear):
            return mx.dequantize(
                linear.weight,
                linear.scales,
                biases=linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )
        return linear.weight

    @staticmethod
    def _bake_lora_into_linear(base_linear: nn.Linear | nn.QuantizedLinear, lora_layer: LoRALinear) -> nn.Module:
        delta = mx.matmul(lora_layer.lora_A, lora_layer.lora_B)
        delta = mx.transpose(delta)
        delta = lora_layer.scale * delta
        return LoRASaver._bake_delta_into_linear(base_linear, delta)

    @staticmethod
    def _bake_lokr_into_linear(base_linear: nn.Linear | nn.QuantizedLinear, lokr_layer: LoKrLinear) -> nn.Module:
        dense_weight = LoRASaver._dense_weight(base_linear)
        delta = lokr_layer.scale * lokr_layer.delta_weight(base_weight=dense_weight)
        return LoRASaver._bake_delta_into_linear(base_linear, delta)

    @staticmethod
    def _bake_delta_into_linear(
        base_linear: nn.Linear | nn.QuantizedLinear,
        delta: mx.array,
    ) -> nn.Module:
        if not hasattr(base_linear, "weight"):
            return base_linear

        dense_weight = LoRASaver._dense_weight(base_linear)
        if dense_weight.shape != delta.shape:
            print(
                "⚠️  Skipping LoRA bake due to shape mismatch: "
                f"weight {dense_weight.shape} vs delta {delta.shape}"
            )
            return base_linear

        merged = dense_weight + delta.astype(dense_weight.dtype)

        try:
            if isinstance(base_linear, nn.QuantizedLinear):
                has_bias = hasattr(base_linear, "bias") and getattr(base_linear, "bias", None) is not None
                dense_linear = nn.Linear(merged.shape[1], merged.shape[0], bias=has_bias)
                dense_linear.weight = merged
                if has_bias:
                    dense_linear.bias = base_linear.bias
                return nn.QuantizedLinear.from_linear(
                    dense_linear,
                    group_size=base_linear.group_size,
                    bits=base_linear.bits,
                    mode=base_linear.mode,
                )

            base_linear.weight = merged.astype(base_linear.weight.dtype)
            return base_linear
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Failed to bake LoRA into base layer: {e}")
            return base_linear
