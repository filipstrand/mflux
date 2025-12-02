import mlx.core as mx
import mlx.nn as nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.zimage.embeddings.rope_3d import apply_rope


class Attention(nn.Module):
    """Self-attention with QK normalization and RoPE support.

    Key differences from standard attention:
    - RMSNorm applied to Q and K before attention (qk_norm=True)
    - RoPE applied after normalization
    - No GQA in Z-Image (n_heads == n_kv_heads == 30)
    """

    # Hardcoded for S3-DiT
    DIM = 3840
    N_HEADS = 30
    N_KV_HEADS = 30  # No GQA
    HEAD_DIM = 128
    SCALE = HEAD_DIM**-0.5

    def __init__(self):
        super().__init__()

        # Projections
        self.to_q = nn.Linear(self.DIM, self.N_HEADS * self.HEAD_DIM, bias=False)
        self.to_k = nn.Linear(self.DIM, self.N_KV_HEADS * self.HEAD_DIM, bias=False)
        self.to_v = nn.Linear(self.DIM, self.N_KV_HEADS * self.HEAD_DIM, bias=False)
        self.to_out = nn.Linear(self.N_HEADS * self.HEAD_DIM, self.DIM, bias=False)

        # QK Normalization (critical for training stability)
        self.norm_q = nn.RMSNorm(self.HEAD_DIM)
        self.norm_k = nn.RMSNorm(self.HEAD_DIM)

    def __call__(self, x: mx.array, rope: tuple[mx.array, mx.array] | None = None) -> mx.array:
        """Forward pass with optional RoPE.

        Args:
            x: Input tensor [B, S, DIM]
            rope: Optional tuple of (freqs_cos, freqs_sin) from RoPE3D

        Returns:
            Output tensor [B, S, DIM]
        """
        B, S, _ = x.shape

        # Project
        q = self.to_q(x).reshape(B, S, self.N_HEADS, self.HEAD_DIM)
        k = self.to_k(x).reshape(B, S, self.N_KV_HEADS, self.HEAD_DIM)
        v = self.to_v(x).reshape(B, S, self.N_KV_HEADS, self.HEAD_DIM)

        # QK Normalization
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Apply RoPE
        if rope is not None:
            freqs_cos, freqs_sin = rope
            q, k = apply_rope(q, k, freqs_cos, freqs_sin)

        # Transpose for attention: [B, heads, S, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention (fused kernel)
        out = scaled_dot_product_attention(q, k, v, scale=self.SCALE, mask=None)

        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.to_out(out)
