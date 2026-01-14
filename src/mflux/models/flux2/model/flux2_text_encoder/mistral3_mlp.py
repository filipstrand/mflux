"""MLP/FFN for Mistral3 text encoder."""

import mlx.core as mx
from mlx import nn


class Mistral3MLP(nn.Module):
    """Mistral3 MLP with gated activation (SwiGLU).

    Uses gate_proj and up_proj with SiLU activation.

    Args:
        hidden_size: Hidden dimension (5120)
        intermediate_size: Intermediate dimension (32768)
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 32768,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Forward pass through MLP.

        Args:
            hidden_states: Input tensor [B, seq_len, hidden_size]

        Returns:
            Output tensor [B, seq_len, hidden_size]
        """
        # SwiGLU: silu(gate) * up
        gate = nn.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = gate * up
        hidden_states = self.down_proj(hidden_states)
        return hidden_states
