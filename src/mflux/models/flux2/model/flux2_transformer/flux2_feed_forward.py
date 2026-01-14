"""Feed forward network for FLUX.2 joint blocks.

FLUX.2 uses a different naming convention than FLUX.1:
- linear_in (instead of net.0.proj)
- linear_out (instead of net.2)

The structure is the same: hidden_dim -> mlp_hidden -> hidden_dim
"""

import mlx.core as mx
from mlx import nn


class Flux2FeedForward(nn.Module):
    """Feed forward network for FLUX.2 joint transformer blocks.

    Args:
        hidden_dim: Input/output dimension
        mlp_ratio: Multiplier for intermediate dimension (default 3.0)
        activation_function: Activation function to use
    """

    def __init__(
        self,
        hidden_dim: int = 6144,
        mlp_ratio: float = 3.0,
        activation_function=nn.gelu,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        self.linear_in = nn.Linear(hidden_dim, mlp_hidden_dim, bias=False)
        self.linear_out = nn.Linear(mlp_hidden_dim, hidden_dim, bias=False)
        self.activation_function = activation_function

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.linear_in(hidden_states)
        hidden_states = self.activation_function(hidden_states)
        hidden_states = self.linear_out(hidden_states)
        return hidden_states


class Flux2FeedForwardContext(nn.Module):
    """Feed forward network for context/text stream in FLUX.2 joint blocks.

    Uses the same structure but may have different input/output dimensions
    if context has different embedding size.
    """

    def __init__(
        self,
        hidden_dim: int = 6144,
        mlp_ratio: float = 3.0,
        activation_function=nn.gelu_approx,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        self.linear_in = nn.Linear(hidden_dim, mlp_hidden_dim, bias=False)
        self.linear_out = nn.Linear(mlp_hidden_dim, hidden_dim, bias=False)
        self.activation_function = activation_function

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.linear_in(hidden_states)
        hidden_states = self.activation_function(hidden_states)
        hidden_states = self.linear_out(hidden_states)
        return hidden_states
