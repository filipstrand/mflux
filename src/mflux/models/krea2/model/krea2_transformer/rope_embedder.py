import mlx.core as mx
import numpy as np


class Krea2RotaryPosEmbed:
    """3-axis (t, h, w) rotary position embedding, Flux-style interleaved.

    Ported from diffusers Krea2RotaryPosEmbed + get_1d_rotary_pos_embed(repeat_interleave_real=True).
    For each axis i, freqs = outer(pos[:, i], inv_freq_i); cos/sin are repeat-interleaved by 2 and
    concatenated across the 3 axes to width = sum(axes_dim) = attention_head_dim.

    The cos/sin tables depend only on position_ids (fixed per resolution), so they are computed once
    in float64 (matching diffusers) and cached as bf16/fp32 mx arrays.
    """

    def __init__(self, theta: int, axes_dim: list[int]):
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def compute(self, position_ids: mx.array, dtype: mx.Dtype = mx.float32) -> tuple[mx.array, mx.array]:
        # position_ids: (seq, 3) int. Compute in float64 numpy for parity with diffusers (freqs_dtype=float64).
        pos = np.asarray(position_ids.astype(mx.float32), dtype=np.float64)  # (seq, 3)
        cos_parts = []
        sin_parts = []
        for i, dim in enumerate(self.axes_dim):
            inv_freq = 1.0 / (self.theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))  # (dim/2,)
            freqs = np.outer(pos[:, i], inv_freq)  # (seq, dim/2)
            cos = np.repeat(np.cos(freqs), 2, axis=1)  # (seq, dim)
            sin = np.repeat(np.sin(freqs), 2, axis=1)
            cos_parts.append(cos)
            sin_parts.append(sin)
        cos = np.concatenate(cos_parts, axis=-1)  # (seq, head_dim)
        sin = np.concatenate(sin_parts, axis=-1)
        return mx.array(cos.astype(np.float32)).astype(dtype), mx.array(sin.astype(np.float32)).astype(dtype)


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply interleaved rotary embedding (use_real_unbind_dim=-1, sequence_dim=1).

    x: (B, S, H, D); cos/sin: (S, D). Computation done in float32 then cast back.
    """
    # cos/sin broadcast over batch and heads: (1, S, 1, D)
    cos_b = cos[None, :, None, :]
    sin_b = sin[None, :, None, :]
    x_f = x.astype(mx.float32)
    # x_real, x_imag = x.reshape(..., -1, 2).unbind(-1); x_rotated = stack([-x_imag, x_real]).flatten
    shape = x_f.shape
    x_pairs = x_f.reshape(*shape[:-1], shape[-1] // 2, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(shape)
    out = x_f * cos_b.astype(mx.float32) + x_rotated * sin_b.astype(mx.float32)
    return out.astype(x.dtype)
