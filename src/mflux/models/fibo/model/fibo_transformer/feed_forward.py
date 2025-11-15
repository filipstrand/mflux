import mlx.core as mx
from mlx import nn


class FiboGELU(nn.Module):
    """
    Minimal MLX port of diffusers.models.activations.GELU with approximate='tanh'.

    We rely on `nn.gelu_approx` to match PyTorch's `F.gelu(..., approximate="tanh")` behavior.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "tanh", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.proj(hidden_states)
        # MLX's gelu_approx corresponds to PyTorch GELU with tanh approximation.
        hidden_states = nn.gelu_approx(hidden_states)
        return hidden_states


class FiboFeedForward(nn.Module):
    """
    MLX port of diffusers.models.attention.FeedForward with activation_fn=\"gelu-approximate\".

    The parameter structure is:
      - net[0].proj.{weight,bias}
      - net[2].{weight,bias}
    matching the PyTorch state_dict keys like:
      - ff.net.0.proj.weight
      - ff.net.2.weight
    """

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

        if activation_fn != "gelu-approximate":
            raise ValueError("FiboFeedForward currently only supports activation_fn='gelu-approximate'.")

        self.net: list[nn.Module] = [
            FiboGELU(dim_in=dim, dim_out=inner_dim, approximate="tanh", bias=True),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=True),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
