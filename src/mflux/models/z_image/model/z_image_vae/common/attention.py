import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.config.config import Config


class Attention(nn.Module):
    """
    Self-attention block for the VAE.

    Matches diffusers Attention with:
    - group_norm before attention
    - QKV projections
    - residual connection

    Used in UNetMidBlock2D.
    """

    def __init__(self, channels: int = 512, num_groups: int = 32):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(
            num_groups=num_groups,
            dims=channels,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, input_array: mx.array) -> mx.array:
        # Input is BCHW, convert to BHWC
        input_array = mx.transpose(input_array, (0, 2, 3, 1))

        B, H, W, C = input_array.shape

        # Group norm
        hidden_states = self.group_norm(input_array.astype(mx.float32)).astype(Config.precision)

        # QKV projections - reshape for attention
        queries = self.to_q(hidden_states).reshape(B, H * W, 1, C)
        keys = self.to_k(hidden_states).reshape(B, H * W, 1, C)
        values = self.to_v(hidden_states).reshape(B, H * W, 1, C)

        # Transpose to (B, num_heads, seq_len, head_dim)
        queries = mx.transpose(queries, (0, 2, 1, 3))
        keys = mx.transpose(keys, (0, 2, 1, 3))
        values = mx.transpose(values, (0, 2, 1, 3))

        # Scaled dot product attention
        scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=queries.dtype))
        hidden_states = scaled_dot_product_attention(queries, keys, values, scale=scale)

        # Reshape back to spatial
        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3)).reshape(B, H, W, C)

        # Output projection
        hidden_states = self.to_out[0](hidden_states)

        # Residual connection
        output = input_array + hidden_states

        # Convert back to BCHW
        return mx.transpose(output, (0, 3, 1, 2))
