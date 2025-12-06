import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
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
            base_linear = lora_layer.linear
            LoRASaver._apply_lora_delta(base_linear, lora_layer)
            return base_linear

        def _bake_fused(fused_layer: FusedLoRALinear) -> nn.Module:
            base_linear = fused_layer.base_linear
            for lora in fused_layer.loras:
                if isinstance(lora, LoRALinear):
                    LoRASaver._apply_lora_delta(base_linear, lora)
            return base_linear

        def _walk(obj, parent=None, attr_name=None, idx=None):
            # Replace wrappers first
            if isinstance(obj, FusedLoRALinear):
                new_child = _bake_fused(obj)
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
    def _apply_lora_delta(base_linear: nn.Module, lora_layer: LoRALinear) -> None:
        if not hasattr(base_linear, "weight"):
            return

        weight = base_linear.weight
        delta = mx.matmul(lora_layer.lora_A, lora_layer.lora_B)  # shape: [in, out]
        delta = mx.transpose(delta)  # shape: [out, in]
        delta = lora_layer.scale * delta

        if weight.shape != delta.shape:
            print(f"⚠️  Skipping LoRA bake due to shape mismatch: weight {weight.shape} vs delta {delta.shape}")
            return

        try:
            base_linear.weight = weight + delta.astype(weight.dtype)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Failed to bake LoRA into base layer: {e}")
