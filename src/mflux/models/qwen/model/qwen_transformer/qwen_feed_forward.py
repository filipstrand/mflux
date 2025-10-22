import mlx.core as mx
from mlx import nn


class QwenFeedForward(nn.Module):
    """
    Matches PyTorch FeedForward class exactly (diffusers/models/attention.py:1668-1731).

    PyTorch initialization:
        FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        # Creates: GELU(dim, inner_dim=4*dim, approximate="tanh", bias=True) -> Dropout(0.0) -> Linear(4*dim, dim, bias=True)
    """

    def __init__(self, dim: int = 3072):
        super().__init__()
        # PyTorch: self.net[0] = GELU(dim, inner_dim=4*dim, approximate="tanh", bias=True)
        # GELU internally does: Linear(dim, 4*dim) -> gelu(approximate="tanh")
        self.mlp_in = nn.Linear(dim, 4 * dim, bias=True)

        # PyTorch: self.net[1] = Dropout(0.0) - no-op, so we skip it
        # PyTorch: self.net[2] = Linear(inner_dim, dim_out, bias=True)
        self.mlp_out = nn.Linear(4 * dim, dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Matches PyTorch FeedForward.forward exactly.

        PyTorch:
            for module in self.net:
                hidden_states = module(hidden_states)
            # Which is: GELU -> Dropout(no-op) -> Linear
        """
        # PyTorch: hidden_states = GELU(hidden_states)
        # GELU does: proj(hidden_states) -> gelu(approximate="tanh")
        # Match PyTorch: NO explicit dtype casting, keep original dtype
        hidden_states = self.mlp_in(hidden_states)

        # PyTorch: F.gelu(gate, approximate="tanh")
        hidden_states = nn.gelu_approx(hidden_states)

        # PyTorch: Dropout(0.0) - no-op, skip

        # PyTorch: Linear(inner_dim, dim_out, bias=True)
        hidden_states = self.mlp_out(hidden_states)

        return hidden_states
