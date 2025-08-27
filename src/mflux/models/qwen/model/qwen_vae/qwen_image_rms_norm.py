import mlx.core as mx
from mlx import nn


class QwenImageRMSNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = float(num_channels) ** 0.5
        self.weight = mx.ones((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        l2 = mx.sqrt(mx.sum(mx.square(x), axis=1, keepdims=True) + self.eps)
        x_normalized = (x / l2) * self.scale
        weight = self.weight.reshape(1, -1, 1, 1, 1)
        return x_normalized * weight
