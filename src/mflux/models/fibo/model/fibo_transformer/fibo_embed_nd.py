import mlx.core as mx
from mlx import nn


class FiboEmbedND(nn.Module):
    def __init__(self, theta: int = 10000, axes_dim: list[int] | None = None):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim or [16, 56, 56]

    def __call__(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        if ids.ndim == 3 and ids.shape[0] == 1:
            ids = ids[0]

        if ids.ndim != 2 or ids.shape[-1] != 3:
            raise ValueError(f"Expected ids with shape (S, 3) or (1, S, 3), got {ids.shape}")

        n_axes = ids.shape[-1]
        pos = ids.astype(mx.float32)  # (S, 3)

        cos_out: list[mx.array] = []
        sin_out: list[mx.array] = []

        for i in range(n_axes):
            axis_dim = self.axes_dim[i]
            cos_axis, sin_axis = self._get_1d_rotary_pos_embed_mlx(
                dim=axis_dim,
                pos=pos[:, i],
                theta=self.theta,
            )
            cos_out.append(cos_axis)
            sin_out.append(sin_axis)

        freqs_cos = mx.concatenate(cos_out, axis=-1)
        freqs_sin = mx.concatenate(sin_out, axis=-1)
        return freqs_cos, freqs_sin

    @staticmethod
    def _get_1d_rotary_pos_embed_mlx(
        dim: int,
        pos: mx.array,
        theta: float = 10000.0,
    ) -> tuple[mx.array, mx.array]:
        assert dim % 2 == 0, "RoPE dim must be even"
        if pos.ndim != 1:
            pos = mx.reshape(pos, (-1,))
        pos = pos.astype(mx.float32)
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        angles = pos[:, None] * freqs[None, :]
        cos_base = mx.cos(angles)
        sin_base = mx.sin(angles)
        cos = mx.reshape(mx.stack([cos_base, cos_base], axis=-1), (pos.shape[0], -1))
        sin = mx.reshape(mx.stack([sin_base, sin_base], axis=-1), (pos.shape[0], -1))
        return cos, sin
