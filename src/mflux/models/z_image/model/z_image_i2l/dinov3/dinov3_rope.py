import math

import mlx.core as mx


def compute_dinov3_rope_embeddings(
    num_patches_h: int,
    num_patches_w: int,
    head_dim: int = 128,
    theta: float = 100.0,
    rescale: float = 2.0,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[mx.array, mx.array]:
    """Compute 2D RoPE (cos, sin) for DINOv3 patch tokens.

    DINOv3 uses 2D coordinates of patch centers normalized to [-1, +1],
    then computes rotary embeddings from those coordinates.

    Args:
        num_patches_h: Number of patches along height.
        num_patches_w: Number of patches along width.
        head_dim: Dimension per attention head (4096/32 = 128).
        theta: RoPE theta parameter (default 100.0).
        rescale: Position coordinate rescale factor (default 2.0, applied as log-uniform mean).
        dtype: Output dtype.

    Returns:
        (cos, sin) each of shape (num_patches_h * num_patches_w, head_dim).
    """
    # inv_freq: (head_dim / 4,) — 32 frequencies
    inv_freq = 1.0 / (theta ** mx.arange(0, 1, 4 / head_dim))

    # Patch center coordinates normalized to [-1, +1]
    coords_h = (mx.arange(0.5, num_patches_h) / num_patches_h) * 2.0 - 1.0
    coords_w = (mx.arange(0.5, num_patches_w) / num_patches_w) * 2.0 - 1.0

    # Meshgrid: (H, W, 2)
    grid_h = mx.broadcast_to(coords_h[:, None], (num_patches_h, num_patches_w))
    grid_w = mx.broadcast_to(coords_w[None, :], (num_patches_h, num_patches_w))
    coords = mx.stack([grid_h, grid_w], axis=-1)  # (H, W, 2)
    coords = coords.reshape(-1, 2)  # (H*W, 2)

    # angles: (H*W, 2, head_dim/4) -> (H*W, head_dim/2) -> tile to (H*W, head_dim)
    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.reshape(-1, angles.shape[1] * angles.shape[2])  # (H*W, head_dim/2)
    angles = mx.concatenate([angles, angles], axis=-1)  # (H*W, head_dim)

    cos = mx.cos(angles).astype(dtype)
    sin = mx.sin(angles).astype(dtype)
    return cos, sin


def apply_dinov3_rope(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    num_prefix_tokens: int = 5,
) -> tuple[mx.array, mx.array]:
    """Apply RoPE to query and key tensors, skipping prefix tokens (CLS + registers).

    Args:
        q: (B, num_heads, N, head_dim)
        k: (B, num_heads, N, head_dim)
        cos: (num_patches, head_dim)
        sin: (num_patches, head_dim)
        num_prefix_tokens: Number of prefix tokens to skip (1 CLS + 4 registers = 5).

    Returns:
        (q_rotated, k_rotated) with same shapes.
    """
    q_prefix = q[:, :, :num_prefix_tokens, :]
    q_patches = q[:, :, num_prefix_tokens:, :]
    k_prefix = k[:, :, :num_prefix_tokens, :]
    k_patches = k[:, :, num_prefix_tokens:, :]

    # rotate_half: split last dim in half, swap and negate
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    q_patches = q_patches * cos + rotate_half(q_patches) * sin
    k_patches = k_patches * cos + rotate_half(k_patches) * sin

    q = mx.concatenate([q_prefix, q_patches], axis=2)
    k = mx.concatenate([k_prefix, k_patches], axis=2)
    return q, k
