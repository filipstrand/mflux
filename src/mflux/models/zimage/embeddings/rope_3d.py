import mlx.core as mx
import mlx.nn as nn


def apply_rope_complex(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Apply rotary embeddings using complex number representation.

    This matches the diffusers implementation which uses interleaved pairing
    of adjacent elements as complex numbers.

    Args:
        x: Input tensor [B, S, n_heads, head_dim]
        freqs_cis: Complex frequencies [S, head_dim//2, 2] where last dim is [cos, sin]

    Returns:
        Rotated tensor with same shape as input
    """
    # Reshape x to pair adjacent elements: [B, S, n_heads, head_dim//2, 2]
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)

    # Extract real and imaginary parts (adjacent pairs)
    x_real = x_reshaped[..., 0]  # [B, S, n_heads, head_dim//2]
    x_imag = x_reshaped[..., 1]  # [B, S, n_heads, head_dim//2]

    # Extract cos and sin from freqs_cis: [S, head_dim//2]
    # Broadcast to [1, S, 1, head_dim//2]
    freqs_cos = freqs_cis[..., 0][None, :, None, :]
    freqs_sin = freqs_cis[..., 1][None, :, None, :]

    # Complex multiplication: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos

    # Interleave back: [B, S, n_heads, head_dim//2, 2] -> [B, S, n_heads, head_dim]
    out = mx.stack([out_real, out_imag], axis=-1)
    return out.reshape(*x.shape)


def apply_rope(q: mx.array, k: mx.array, freqs_cos: mx.array, freqs_sin: mx.array) -> tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to Q and K.

    Args:
        q: Query tensor [B, S, n_heads, head_dim]
        k: Key tensor [B, S, n_kv_heads, head_dim]
        freqs_cos: Cosine frequencies [S, head_dim//2]
        freqs_sin: Sine frequencies [S, head_dim//2]

    Returns:
        Rotated Q and K tensors
    """
    # Stack cos and sin into [S, head_dim//2, 2] format for complex multiplication
    freqs_cis = mx.stack([freqs_cos, freqs_sin], axis=-1)

    q_rotated = apply_rope_complex(q, freqs_cis)
    k_rotated = apply_rope_complex(k, freqs_cis)

    return q_rotated, k_rotated


class RoPE3D(nn.Module):
    """3D Rotary Position Embeddings for (time, height, width).

    Z-Image uses 3-axis RoPE with specific frequency configurations:
    - axes_dims: (32, 48, 48) - frequency dimensions per axis
    - axes_lens: (1024, 512, 512) - maximum sequence lengths
    - theta: 256.0 - base frequency

    Position encoding scheme:
    - Caption tokens: time=1,2,3,...,cap_len, h=0, w=0
    - Image tokens: time=cap_len+1 (fixed), h=0..H-1, w=0..W-1
    """

    AXES_DIMS = (32, 48, 48)  # Dims for time, height, width (total 128 = head_dim)
    AXES_LENS = (1024, 512, 512)  # Max lengths per axis
    THETA = 256.0  # Base frequency
    HEAD_DIM = 128  # Matches S3DiT head dimension

    def __init__(self):
        super().__init__()
        # Precompute frequencies for each axis (will be looked up by position)
        self._freqs_cache = {}
        self._combined_cache = {}  # Cache for combined image + caption frequencies
        self._precompute_freqs()

    def _precompute_freqs(self):
        """Precompute frequency tables for each axis."""
        for axis_idx, (dim, max_len) in enumerate(zip(self.AXES_DIMS, self.AXES_LENS)):
            # Compute base frequencies: 1 / theta^(2i/d) for i=0..dim//2-1
            half_dim = dim // 2
            base_freqs = 1.0 / (self.THETA ** (mx.arange(0, half_dim, dtype=mx.float32) / half_dim))

            # Compute angles for all positions: [max_len, dim//2]
            positions = mx.arange(max_len, dtype=mx.float32)[:, None]
            angles = positions * base_freqs[None, :]

            # Store as cos and sin
            self._freqs_cache[axis_idx] = (mx.cos(angles), mx.sin(angles))

    def get_freqs_for_positions(self, pos_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Look up frequencies for given position IDs.

        Args:
            pos_ids: Position indices [N, 3] where columns are (time, height, width)

        Returns:
            Tuple of (freqs_cos, freqs_sin) each [N, head_dim//2]
        """
        # Vectorized version: Extract all axis positions at once
        axis_positions = [pos_ids[:, axis_idx].astype(mx.int32) for axis_idx in range(3)]

        # Look up frequencies for all axes in parallel
        axis_cos_list = [self._freqs_cache[axis_idx][0][axis_positions[axis_idx]] for axis_idx in range(3)]
        axis_sin_list = [self._freqs_cache[axis_idx][1][axis_positions[axis_idx]] for axis_idx in range(3)]

        # Concatenate along last dimension: [N, head_dim//2]
        freqs_cos = mx.concatenate(axis_cos_list, axis=-1)
        freqs_sin = mx.concatenate(axis_sin_list, axis=-1)

        return freqs_cos, freqs_sin

    def get_image_freqs(self, h_patches: int, w_patches: int, time_offset: int = 1) -> tuple[mx.array, mx.array]:
        """Generate RoPE frequencies for image patches.

        Args:
            h_patches: Number of patches in height
            w_patches: Number of patches in width
            time_offset: Time position for all image tokens (typically cap_len + 1)

        Returns:
            Tuple of (freqs_cos, freqs_sin) each [N_patches, head_dim//2]
        """
        n_patches = h_patches * w_patches

        # Create position IDs: [N_patches, 3]
        h_pos, w_pos = self._get_2d_positions(h_patches, w_patches)
        t_pos = mx.full((n_patches,), time_offset, dtype=mx.int32)

        pos_ids = mx.stack([t_pos, h_pos, w_pos], axis=-1)
        return self.get_freqs_for_positions(pos_ids)

    def get_caption_freqs(self, cap_len: int) -> tuple[mx.array, mx.array]:
        """Generate RoPE frequencies for caption tokens.

        Args:
            cap_len: Number of caption tokens

        Returns:
            Tuple of (freqs_cos, freqs_sin) each [cap_len, head_dim//2]
        """
        # Caption positions: time=1,2,3,...,cap_len, h=0, w=0
        t_pos = mx.arange(1, cap_len + 1, dtype=mx.int32)
        h_pos = mx.zeros((cap_len,), dtype=mx.int32)
        w_pos = mx.zeros((cap_len,), dtype=mx.int32)

        pos_ids = mx.stack([t_pos, h_pos, w_pos], axis=-1)
        return self.get_freqs_for_positions(pos_ids)

    def get_combined_freqs(
        self, h_patches: int, w_patches: int, cap_len: int, padded_cap_len: int
    ) -> tuple[mx.array, mx.array]:
        """Get cached combined frequencies for image + caption.

        This is the main optimization: since image dimensions and caption length
        are typically fixed for a generation run, we can precompute and cache
        the combined frequencies instead of computing them separately each time.

        Args:
            h_patches: Number of patches in height
            w_patches: Number of patches in width
            cap_len: Number of caption tokens
            padded_cap_len: Padded caption length (for time offset calculation)

        Returns:
            Tuple of (combined_freqs_cos, combined_freqs_sin) each [N_img + cap_len, head_dim//2]
        """
        key = (h_patches, w_patches, cap_len, padded_cap_len)
        if key not in self._combined_cache:
            # Compute image frequencies with proper time offset
            img_freqs_cos, img_freqs_sin = self.get_image_freqs(h_patches, w_patches, time_offset=padded_cap_len + 1)

            # Compute caption frequencies
            cap_freqs_cos, cap_freqs_sin = self.get_caption_freqs(cap_len)

            # Concatenate: image first, then caption (matches S3DiT order)
            combined_cos = mx.concatenate([img_freqs_cos, cap_freqs_cos], axis=0)
            combined_sin = mx.concatenate([img_freqs_sin, cap_freqs_sin], axis=0)

            self._combined_cache[key] = (combined_cos, combined_sin)

        return self._combined_cache[key]

    def __call__(self, height: int, width: int) -> tuple[mx.array, mx.array]:
        """Generate RoPE frequencies for image dimensions (legacy interface).

        Args:
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            Tuple of (freqs_cos, freqs_sin) each [N_patches, head_dim//2]
        """
        h_patches = height // 16
        w_patches = width // 16
        return self.get_image_freqs(h_patches, w_patches, time_offset=1)

    def _get_2d_positions(self, h_patches: int, w_patches: int) -> tuple[mx.array, mx.array]:
        """Generate 2D position indices for image patches.

        Returns:
            h_pos, w_pos: Position arrays flattened to [N]
        """
        h_pos = mx.arange(h_patches, dtype=mx.int32)
        w_pos = mx.arange(w_patches, dtype=mx.int32)

        # Create meshgrid and flatten
        h_grid = mx.repeat(h_pos[:, None], w_patches, axis=1)  # [H, W]
        w_grid = mx.repeat(w_pos[None, :], h_patches, axis=0)  # [H, W]

        return h_grid.flatten(), w_grid.flatten()
