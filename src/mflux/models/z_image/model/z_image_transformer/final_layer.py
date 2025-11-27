import mlx.core as mx
from mlx import nn


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = [nn.Linear(min(hidden_size, 256), hidden_size, bias=True)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaLN_modulation[0](nn.silu(c))
        scale = mx.expand_dims(scale, axis=1)
        x = self.norm(x) * scale
        return self.linear(x)
