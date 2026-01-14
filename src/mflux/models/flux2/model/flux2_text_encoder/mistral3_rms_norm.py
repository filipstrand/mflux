"""RMSNorm for Mistral3 text encoder."""

import mlx.core as mx
from mlx import nn


class Mistral3RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        hidden_size: Hidden dimension size
        eps: Epsilon for numerical stability
    """

    def __init__(self, hidden_size: int = 5120, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).astype(input_dtype)
