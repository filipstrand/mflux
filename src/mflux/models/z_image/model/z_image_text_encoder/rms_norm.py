import mlx.core as mx
from mlx import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        return self.weight * x * mx.rsqrt(variance + self.eps)
