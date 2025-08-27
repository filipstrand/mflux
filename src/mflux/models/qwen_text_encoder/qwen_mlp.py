import mlx.core as mx
from mlx import nn


class QwenMLP(nn.Module):
    """
    Feed-forward network for Qwen text encoder.

    Uses SiLU (Swish) activation function and follows the
    standard flux_transformer MLP architecture.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        # Gate and up projections (similar to LLaMA-style MLP)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # Down projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass of Qwen MLP.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Apply gate and up projections
        gate_output = nn.silu(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)

        # Element-wise multiplication (gating mechanism)
        intermediate_output = gate_output * up_output

        # Apply down projection
        output = self.down_proj(intermediate_output)

        return output
