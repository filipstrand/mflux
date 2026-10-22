import mlx.core as mx
from mlx import nn


class Flux3Modulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 3, bias=False)

    def __call__(self, vec: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        # (B, dim) -> (shift, scale, gate), each (B, 1, dim)
        mod = self.linear(nn.silu(vec))[:, None, :]
        shift, scale, gate = mx.split(mod, 3, axis=-1)
        return shift, scale, gate
