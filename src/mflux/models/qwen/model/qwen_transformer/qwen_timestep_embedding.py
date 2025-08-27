import mlx.core as mx
from mlx import nn


class QwenTimestepEmbedding(nn.Module):
    def __init__(self, proj_dim: int, inner_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(proj_dim, inner_dim)
        self.linear_2 = nn.Linear(inner_dim, inner_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.silu(self.linear_1(x))
        x = self.linear_2(x)
        return x
