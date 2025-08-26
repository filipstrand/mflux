import mlx.core as mx
from mlx import nn


class QwenRMSNorm(nn.Module):
    """
    RMS Normalization implementation for Qwen models.

    This follows the same pattern as used in the Qwen transformer blocks
    but adapted for text encoding.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps  # Store as eps (like reference) not variance_epsilon

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)

        # Compute RMS
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)

        # Apply weight and convert back to original dtype
        hidden_states = self.weight * hidden_states
        return hidden_states.astype(input_dtype)
