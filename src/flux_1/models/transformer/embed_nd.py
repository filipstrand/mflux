import mlx.core as mx
from mlx import nn


class EmbedND(nn.Module):

    def __init__(self):
        super().__init__()
        self.dim = 3072
        self.theta = 10000
        self.axes_dim = [16, 56, 56]

    def forward(self, ids: mx.array) -> mx.array:
        emb = mx.concatenate(
            [EmbedND.rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)],
            axis=-3,
        )
        return mx.expand_dims(emb, axis=1)

    @staticmethod
    def rope(pos: mx.array, dim: int, theta: float) -> mx.array:
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (theta ** scale)
        batch_size, seq_length = pos.shape
        pos_expanded = mx.expand_dims(pos, axis=-1)
        omega_expanded = mx.expand_dims(omega, axis=0)
        out = pos_expanded * omega_expanded
        cos_out = mx.cos(out)
        sin_out = mx.sin(out)
        stacked_out = mx.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
        out = mx.reshape(stacked_out, (batch_size, -1, dim // 2, 2, 2))
        return out
