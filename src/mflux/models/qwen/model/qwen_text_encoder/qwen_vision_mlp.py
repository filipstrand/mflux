import mlx.core as mx
from mlx import nn


class VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # GLU-style MLP with 3 projections (SwiGLU variant)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # GLU formula: down_proj(silu(gate_proj(x)) * up_proj(x))
        # silu(x) = x * sigmoid(x), also known as Swish
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
