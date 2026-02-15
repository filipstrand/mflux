import mlx.core as mx
from mlx import nn


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int = 4,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=bias)
        self.proj_in_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.proj_in_gate(x))
        x = self.proj_in(x)
        x = gate * x
        x = self.proj_out(x)
        return x


class GELUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.proj_in = nn.Linear(dim, hidden_dim, bias=bias)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj_in(x)
        x = nn.gelu_approx(x)
        x = self.proj_out(x)
        return x
