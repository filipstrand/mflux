import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_transformer.fibo_gelu import FiboGELU


class FiboFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        activation_fn: str = "gelu-approximate",
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        self.net: list[nn.Module] = [
            FiboGELU(dim_in=dim, dim_out=inner_dim, approximate="tanh", bias=True),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=True),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
