import mlx.core as mx
import mlx.nn as nn

from mflux.models.zimage.transformer.attention import Attention
from mflux.models.zimage.transformer.feedforward import SwiGLUFeedForward


class S3DiTBlock(nn.Module):
    """Single S3-DiT transformer block with AdaLN conditioning.

    Structure matches HuggingFace Z-Image-Turbo weights:
    1. AdaLN modulation from 256-dim timestep embedding
    2. Attention with pre/post RMSNorm and gated residual
    3. FFN with pre/post RMSNorm and gated residual
    """

    DIM = 3840
    TEMB_DIM = 256  # Timestep embedding dimension
    N_HEADS = 30
    N_KV_HEADS = 30
    HEAD_DIM = 128
    NORM_EPS = 1e-5

    def __init__(self):
        super().__init__()

        # AdaLN modulation: 256-dim timestep -> 4*dim modulation params
        # Output: [scale_attn, shift_attn, scale_ffn, shift_ffn] (4 vectors of dim 3840)
        self.adaLN = nn.Linear(self.TEMB_DIM, 4 * self.DIM, bias=True)

        # Attention norms (pre/post)
        self.attention_norm1 = nn.RMSNorm(self.DIM, eps=self.NORM_EPS)
        self.attention_norm2 = nn.RMSNorm(self.DIM, eps=self.NORM_EPS)

        # Self-attention
        self.attn = Attention()

        # FFN norms (pre/post)
        self.ffn_norm1 = nn.RMSNorm(self.DIM, eps=self.NORM_EPS)
        self.ffn_norm2 = nn.RMSNorm(self.DIM, eps=self.NORM_EPS)

        # FFN (SwiGLU with hidden_dim=10240)
        self.ff = SwiGLUFeedForward()

    def __call__(
        self,
        x: mx.array,
        temb: mx.array,
        rope: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """Forward pass with timestep conditioning.

        Args:
            x: Hidden states [B, S, dim]
            temb: Timestep embedding [B, temb_dim=256]
            rope: Optional tuple of (freqs_cos, freqs_sin) from RoPE3D

        Returns:
            Updated hidden states [B, S, dim]
        """
        # Get AdaLN modulation parameters (4 vectors)
        # Order from HuggingFace: scale_msa, gate_msa, scale_mlp, gate_mlp
        modulation = self.adaLN(temb)  # [B, 4*dim]
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(modulation[:, None, :], 4, axis=-1)  # Each [B, 1, dim]

        # Apply tanh to gates, add 1.0 to scales (matches diffusers)
        gate_msa = mx.tanh(gate_msa)
        gate_mlp = mx.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        # Attention block: scale input, gate output
        attn_out = self.attn(
            self.attention_norm1(x) * scale_msa,
            rope=rope,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)

        # FFN block: scale input, gate output
        ffn_out = self.ff(
            self.ffn_norm1(x) * scale_mlp,
        )
        x = x + gate_mlp * self.ffn_norm2(ffn_out)

        return x
