import mlx.core as mx
from mlx import nn


class Flux3FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 3.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.linear_in = nn.Linear(dim, hidden * 2, bias=False)
        self.linear_out = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.linear_in(x), 2, axis=-1)
        return self.linear_out(nn.silu(gate) * up)
