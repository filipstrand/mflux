import math

import mlx.core as mx
from mlx import nn


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear | nn.QuantizedLinear,
        r: int = 16,
        scale: float = 1.0,
        alpha: float | None = None,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            scale=scale,
            alpha=alpha,
        )
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        scale: float = 1.0,
        bias: bool = False,
        alpha: float | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.r = r
        # Alpha controls the scaling of LoRA contribution. Default to rank for backward compatibility.
        # When alpha == r (default), effective_scale equals scale, maintaining backward compatibility.
        alpha_value = alpha if alpha is not None else float(r)
        self.alpha = alpha_value
        # Precompute effective scale to avoid division on every forward pass
        self.effective_scale = (alpha_value / r) * scale
        init_scale = 1 / math.sqrt(input_dims)

        # Standard LoRA initialization: A uses Kaiming uniform, B initialized to zero
        # for gradual adaptation (output starts as base model output)
        # Use string keys to register as trainable parameters in MLX nn.Module
        self["lora_A"] = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self["lora_B"] = mx.zeros((r, output_dims))

    def __call__(self, x):
        base_out = self.linear(x)
        lora_out = mx.matmul(mx.matmul(x, self["lora_A"]), self["lora_B"])
        return base_out + self.effective_scale * lora_out
