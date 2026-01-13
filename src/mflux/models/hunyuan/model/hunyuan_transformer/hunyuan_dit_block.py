"""Hunyuan-DiT Transformer Block.

Each DiT block contains:
1. Self-attention over image tokens (with RoPE)
2. Cross-attention for text conditioning
3. Feed-forward network (MLP)

The architecture uses:
- AdaLN modulation for norm1 only
- Simple LayerNorm for norm2 and norm3
- Residual connections without gating
"""

import mlx.core as mx
from mlx import nn

from mflux.models.hunyuan.model.hunyuan_transformer.hunyuan_attention import (
    HunyuanCrossAttention,
    HunyuanSelfAttention,
)


class HunyuanAdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization with learnable shift and scale.

    Computes: output = (input - mean) / std * (1 + scale) + shift
    where scale and shift are predicted from conditioning.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6, affine=True)
        # Linear to produce (shift, scale) from conditioning
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            temb: Timestep embedding [batch, hidden_dim]

        Returns:
            Tuple of (normalized_output, shift, scale)
        """
        # Predict shift and scale from timestep embedding
        emb = self.linear(temb)  # [batch, 2 * hidden_dim]
        shift, scale = mx.split(emb, 2, axis=-1)

        # Apply layer norm
        normed = self.norm(hidden_states)

        # Apply adaptive shift/scale
        output = normed * (1 + scale[:, None, :]) + shift[:, None, :]

        return output, shift, scale


class HunyuanFeedForward(nn.Module):
    """Feed-forward network for Hunyuan-DiT.

    Uses GELU activation with approximate computation.
    Hidden dim is typically 4x the input dim.
    """

    def __init__(
        self,
        hidden_dim: int = 1408,
        intermediate_dim: int | None = None,
    ):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 4

        self.net_0_proj = nn.Linear(hidden_dim, intermediate_dim, bias=True)
        self.net_2 = nn.Linear(intermediate_dim, hidden_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Forward pass."""
        hidden_states = self.net_0_proj(hidden_states)
        hidden_states = nn.gelu_approx(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class HunyuanDiTBlock(nn.Module):
    """Single DiT transformer block for Hunyuan.

    Architecture (matches HuggingFace Diffusers):
    1. Self-attention (with AdaLN conditioning via norm1)
    2. Cross-attention (with simple LayerNorm via norm2)
    3. Feed-forward (with simple LayerNorm via norm3)

    Args:
        hidden_dim: Hidden dimension (1408 for Hunyuan-DiT)
        num_heads: Number of attention heads (16 for Hunyuan-DiT)
        head_dim: Dimension per head (88 for Hunyuan-DiT)
        text_dim: Text encoder dimension (hidden_dim after projection)
        intermediate_dim: FFN intermediate dimension
    """

    def __init__(
        self,
        hidden_dim: int = 1408,
        num_heads: int = 16,
        head_dim: int = 88,
        text_dim: int = 1408,
        intermediate_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-attention components with AdaLN
        self.norm1 = HunyuanAdaLayerNorm(hidden_dim)
        self.attn1 = HunyuanSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Cross-attention components with simple LayerNorm
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn2 = HunyuanCrossAttention(
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Feed-forward components with simple LayerNorm
        self.norm3 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ff = HunyuanFeedForward(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        rotary_emb: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass of DiT block.

        Args:
            hidden_states: Image hidden states [batch, img_seq_len, hidden_dim]
            encoder_hidden_states: Text encoder output [batch, text_seq_len, text_dim]
            temb: Timestep embedding [batch, hidden_dim]
            rotary_emb: Rotary position embeddings for self-attention
            attention_mask: Optional attention mask for cross-attention

        Returns:
            Updated hidden states [batch, img_seq_len, hidden_dim]
        """
        # 1. Self-attention with AdaLN conditioning
        norm_hidden_states, _, _ = self.norm1(hidden_states, temb)
        attn_output = self.attn1(norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = hidden_states + attn_output

        # 2. Cross-attention with simple LayerNorm
        norm_hidden_states = self.norm2(hidden_states)
        cross_attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + cross_attn_output

        # 3. Feed-forward with simple LayerNorm
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output

        return hidden_states
