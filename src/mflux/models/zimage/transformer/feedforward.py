import mlx.core as mx
import mlx.nn as nn


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feedforward network.

    SwiGLU: out = (Swish(W1 * x) * W3 * x) @ W2

    Key difference from GEGLU:
    - Uses Swish (SiLU) instead of GELU
    - Three-weight design: gate (W1), up (W3), down (W2)

    Used by both ContextRefiner and main transformer blocks.
    """

    DIM = 3840
    HIDDEN_DIM = 10240  # From HuggingFace weights

    def __init__(self):
        super().__init__()
        hidden_dim = self.HIDDEN_DIM

        # W1: gate projection
        self.w1 = nn.Linear(self.DIM, hidden_dim, bias=False)
        # W2: down projection
        self.w2 = nn.Linear(hidden_dim, self.DIM, bias=False)
        # W3: up projection
        self.w3 = nn.Linear(self.DIM, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor [B, S, DIM]

        Returns:
            Output tensor [B, S, DIM]
        """
        # SwiGLU: swish(gate) * up, then down
        gate = nn.silu(self.w1(x))  # Swish activation
        up = self.w3(x)
        return self.w2(gate * up)
