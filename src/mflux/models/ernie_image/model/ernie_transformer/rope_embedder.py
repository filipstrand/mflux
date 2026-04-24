import mlx.core as mx
from mlx import nn


def _rope(pos: mx.array, dim: int, theta: int) -> mx.array:
    # pos: [...], dim: int, theta: int
    # Returns: [..., dim//2] — raw angle values (not cos/sin yet)
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim  # [dim//2]
    omega = 1.0 / (theta**scale)  # [dim//2]
    # pos[..., None] * omega[None, ...] – broadcast outer product
    return pos[..., None].astype(mx.float32) * omega  # [..., dim//2]


class ErnieRopeEmbedder(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array) -> mx.array:
        # ids: [B, S, 3]
        # Returns: [B, S, 1, head_dim] – raw angles (apply cos/sin in attention)
        parts = [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)]
        emb = mx.concatenate(parts, axis=-1)  # [B, S, sum(axes_dim)//2=64]
        emb = emb[:, :, None, :]  # [B, S, 1, 64]
        # Duplicate each angle to produce [θ₀, θ₀, θ₁, θ₁, ...] pattern → head_dim=128
        emb = mx.stack([emb, emb], axis=-1)  # [B, S, 1, 64, 2]
        return emb.reshape(*emb.shape[:-2], -1)  # [B, S, 1, 128]


def apply_rotary_emb(q: mx.array, k: mx.array, freqs_cis: mx.array) -> tuple[mx.array, mx.array]:
    # q, k: [B, S, heads, head_dim]
    # freqs_cis: [B, S, 1, head_dim] – raw angles
    cos = mx.cos(freqs_cis).astype(q.dtype)  # [B, S, 1, head_dim]
    sin = mx.sin(freqs_cis).astype(q.dtype)

    rot_dim = freqs_cis.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    half = rot_dim // 2
    q1, q2 = q_rot[..., :half], q_rot[..., half:]
    k1, k2 = k_rot[..., :half], k_rot[..., half:]

    q_rotated = mx.concatenate([-q2, q1], axis=-1)
    k_rotated = mx.concatenate([-k2, k1], axis=-1)

    q_out = mx.concatenate([q_rot * cos + q_rotated * sin, q_pass], axis=-1)
    k_out = mx.concatenate([k_rot * cos + k_rotated * sin, k_pass], axis=-1)
    return q_out, k_out
