import mlx.core as mx
from mlx import nn


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int = 4,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_in_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.proj_in_gate(x))
        x = self.proj_in(x)
        x = gate * x
        x = self.proj_out(x)
        return x
