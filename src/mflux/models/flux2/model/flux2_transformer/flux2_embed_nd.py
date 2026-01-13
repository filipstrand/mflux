"""Rotary position embeddings for FLUX.2.

Same RoPE structure as FLUX.1 but with 6144 dim (48 heads * 128 head_dim).
The axes_dim remains [16, 56, 56] as head_dim is still 128.
"""

import mlx.core as mx
from mlx import nn


class Flux2EmbedND(nn.Module):
    """N-dimensional rotary position embeddings for FLUX.2."""

    def __init__(self):
        super().__init__()
        self.dim = 6144  # 48 heads * 128 head_dim
        self.theta = 10000
        # axes_dim sums to 128 (head_dim) - same as FLUX.1
        self.axes_dim = [16, 56, 56]

    def __call__(self, ids: mx.array) -> mx.array:
        """Compute rotary embeddings for position IDs.

        Args:
            ids: Position IDs [batch, seq_len, 3] (time, height, width)

        Returns:
            Rotary embeddings [batch, 1, seq_len, head_dim//2, 2, 2]
        """
        emb = mx.concatenate(
            [Flux2EmbedND.rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)],
            axis=-3,
        )
        return mx.expand_dims(emb, axis=1)

    @staticmethod
    def rope(pos: mx.array, dim: int, theta: float) -> mx.array:
        """Compute rotary position embeddings.

        Args:
            pos: Position indices [batch, seq_len]
            dim: Dimension for this axis
            theta: Base for frequency computation

        Returns:
            Rotary embedding matrices [batch, seq_len, dim//2, 2, 2]
        """
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (theta**scale)
        batch_size, seq_length = pos.shape
        pos_expanded = mx.expand_dims(pos, axis=-1)
        omega_expanded = mx.expand_dims(omega, axis=0)
        out = pos_expanded * omega_expanded
        cos_out = mx.cos(out)
        sin_out = mx.sin(out)
        stacked_out = mx.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
        out = mx.reshape(stacked_out, (batch_size, -1, dim // 2, 2, 2))
        return out
