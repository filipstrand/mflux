import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


def _is_fp8_linear(linear) -> bool:
    # fp8 layers (e.g. Ideogram 4's Fp8Linear) store raw uint8 codes in .weight plus a
    # per-row weight_scale. A float delta cannot be folded into the codes directly:
    # `delta.astype(uint8)` truncates the (small) LoRA delta to zero, silently destroying
    # the LoRA while leaving the base intact.
    weight = getattr(linear, "weight", None)
    return (
        weight is not None
        and weight.dtype == mx.uint8
        and hasattr(linear, "weight_scale")
        and not isinstance(linear, nn.QuantizedLinear)
    )


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
            if _is_fp8_linear(base_linear):
                return LoRASaver._fold_fp8_loras_to_q8(base_linear, [lora_layer])
            LoRASaver._apply_lora_delta(base_linear, lora_layer)
            return base_linear

        def _bake_fused(fused_layer: FusedLoRALinear) -> nn.Module:
            base_linear = fused_layer.base_linear
            loras = [lora for lora in fused_layer.loras if isinstance(lora, LoRALinear)]
            if _is_fp8_linear(base_linear):
                return LoRASaver._fold_fp8_loras_to_q8(base_linear, loras)
            for lora in loras:
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
    def _fold_fp8_loras_to_q8(base_linear: nn.Module, lora_layers: list[LoRALinear]) -> nn.Module:
        # Dequantize the fp8 base ONCE, add the LoRA delta(s) in float32, and requantize to
        # MLX q8 (group-64 affine keeps more mantissa than fp8-e4m3, so this loses no
        # quality). Besides making the bake CORRECT on fp8 bases, the folded layer uses
        # MLX's fused quantized-matmul kernel instead of materializing the full
        # higher-precision weight matrix on every forward, which is also faster.
        dense = mx.from_fp8(base_linear.weight, dtype=mx.float32) * base_linear.weight_scale[:, None]
        merged = dense
        for lora_layer in lora_layers:
            delta = mx.transpose(mx.matmul(lora_layer.lora_A, lora_layer.lora_B))
            merged = merged + lora_layer.scale * delta.astype(mx.float32)
        bias = getattr(base_linear, "bias", None)
        compute_dtype = getattr(base_linear, "compute_dtype", mx.bfloat16)
        linear = nn.Linear(merged.shape[1], merged.shape[0], bias=bias is not None)
        linear.weight = merged.astype(compute_dtype)
        if bias is not None:
            linear.bias = bias
        quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
        mx.eval(quantized.parameters())
        return quantized

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
