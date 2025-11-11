import mlx.core as mx
from mlx import nn


class VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
