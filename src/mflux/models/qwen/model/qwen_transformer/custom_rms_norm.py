"""
Custom RMSNorm implementation matching diffusers' behavior exactly.

This implementation converts to float32 for variance calculation to match
the numerical behavior of the diffusers implementation, which helps maintain
consistency when porting from PyTorch.
"""

import mlx.core as mx
from mlx import nn


class DiffusersStyleRMSNorm(nn.Module):
    """
    RMSNorm that matches diffusers' implementation behavior.

    Key differences from standard MLX RMSNorm:
    1. Converts to float32 for variance calculation (like diffusers does)
    2. Explicitly handles dtype conversion to match bfloat16 behavior
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype

        # Convert to float32 for variance calculation (matches diffusers)
        hidden_states_f32 = hidden_states.astype(mx.float32)

        # Calculate variance in float32
        variance = mx.mean(hidden_states_f32**2, axis=-1, keepdims=True)

        # Normalize
        hidden_states = hidden_states_f32 * mx.rsqrt(variance + self.eps)

        # Apply weight and convert back to original dtype
        hidden_states = hidden_states * self.weight

        return hidden_states.astype(input_dtype)
