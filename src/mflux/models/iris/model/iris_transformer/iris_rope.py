import mlx.core as mx


class IrisRope:
    @staticmethod
    def rope_2d(
        head_dim: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0
    ) -> tuple[mx.array, mx.array]:
        n_pairs = head_dim // 4
        freqs = 1.0 / theta ** (mx.arange(0, head_dim, 4, dtype=mx.float32)[:n_pairs] / head_dim)
        step = scale / max(max(height, width) - 1, 1)
        x_pos = mx.arange(width, dtype=mx.float32) * step
        y_pos = mx.arange(height, dtype=mx.float32) * step
        x_ang = mx.broadcast_to((x_pos[:, None] * freqs[None, :])[None, :, :], (height, width, n_pairs))
        y_ang = mx.broadcast_to((y_pos[:, None] * freqs[None, :])[:, None, :], (height, width, n_pairs))
        ang = mx.stack([x_ang, y_ang], axis=-1).reshape(height * width, 2 * n_pairs)
        return mx.cos(ang), mx.sin(ang)

    @staticmethod
    def rope_1d(head_dim: int, length: int, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
        freqs = 1.0 / theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        ang = mx.arange(length, dtype=mx.float32)[:, None] * freqs[None, :]
        return mx.cos(ang), mx.sin(ang)

    @staticmethod
    def apply(x: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
        # x: [B, N, H, head_dim], adjacent pairs (x[2i], x[2i+1]) rotate as complex numbers
        cos, sin = rope
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        rotated = mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1)
        return rotated.reshape(x.shape).astype(x.dtype)
