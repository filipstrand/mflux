import mlx.core as mx
from mlx import nn


class IrisSwiGLU(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(2 * int(dim * mlp_ratio) / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class IrisMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.activation = {"gelu": nn.gelu, "silu": nn.silu}[activation]

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.activation(self.fc1(x)))
