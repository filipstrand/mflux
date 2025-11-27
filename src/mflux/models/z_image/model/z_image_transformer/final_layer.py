import mlx.core as mx
from mlx import nn


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = 1e-6
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = [nn.Linear(min(hidden_size, 256), hidden_size, bias=True)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaLN_modulation[0](nn.silu(c))
        scale = mx.expand_dims(scale, axis=1)
        x = FinalLayer._layer_norm(x, self.eps) * scale
        x = self.linear(x)
        return x

    @staticmethod
    def _layer_norm(x: mx.array, eps) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) / mx.sqrt(var + eps)
