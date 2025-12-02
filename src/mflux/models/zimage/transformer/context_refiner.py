import mlx.core as mx
import mlx.nn as nn

from mflux.models.zimage.transformer.attention import Attention
from mflux.models.zimage.transformer.feedforward import SwiGLUFeedForward


class ContextRefinerBlock(nn.Module):
    """Context refiner block for caption embeddings.

    Simpler than S3DiTBlock - no AdaLN conditioning.
    Uses pre/post RMSNorm pattern with SwiGLU FFN.
    """

    DIM = 3840
    N_HEADS = 30
    HEAD_DIM = 128
    NORM_EPS = 1e-5

    def __init__(self):
        super().__init__()

        # Attention norms
        self.attention_norm1 = nn.RMSNorm(self.DIM, eps=self.NORM_EPS)
        self.attention_norm2 = nn.RMSNorm(self.DIM, eps=self.NORM_EPS)

        # Attention
        self.attn = Attention()

        # FFN norms
        self.ffn_norm1 = nn.RMSNorm(self.DIM, eps=self.NORM_EPS)
        self.ffn_norm2 = nn.RMSNorm(self.DIM, eps=self.NORM_EPS)

        # SwiGLU FFN (context refiner uses SwiGLU, not GEGLU)
        self.ff = SwiGLUFeedForward()

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """Forward pass to refine caption embeddings.

        Args:
            x: Caption embeddings [B, seq_len, dim]
            rope: Optional tuple of (freqs_cos, freqs_sin) from RoPE3D

        Returns:
            Refined caption embeddings [B, seq_len, dim]
        """
        # Pre-norm attention with optional RoPE
        x_norm = self.attention_norm1(x)
        attn_out = self.attn(x_norm, rope=rope)
        x = x + self.attention_norm2(attn_out)

        # Pre-norm FFN
        x_norm = self.ffn_norm1(x)
        ff_out = self.ff(x_norm)
        x = x + self.ffn_norm2(ff_out)

        return x
