import mlx.core as mx
import mlx.nn as nn


class Qwen3MLP(nn.Module):
    """Qwen3 MLP with SwiGLU activation.

    gate_proj and up_proj have same output dimension.
    SwiGLU: swish(gate) * up, then down projection.
    """

    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 9728  # From Qwen3-4B config

    def __init__(self):
        super().__init__()

        self.gate_proj = nn.Linear(self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(self.INTERMEDIATE_SIZE, self.HIDDEN_SIZE, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor [B, S, HIDDEN_SIZE]

        Returns:
            Output tensor [B, S, HIDDEN_SIZE]
        """
        # SwiGLU: swish(gate) * up
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
