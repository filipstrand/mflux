"""NextDiT block for NewBie-image.

NextDiT (based on Lumina architecture) uses:
- Grouped Query Attention (GQA)
- AdaLN-Single modulation (efficient variant)
- SwiGLU feed-forward network
- Pre-normalization with RMSNorm
"""

import mlx.core as mx
import mlx.nn as nn

from mflux.models.newbie.model.newbie_transformer.gqa_attention import (
    GQAttention,
    GQCrossAttention,
)


class AdaLNModulation(nn.Module):
    """Adaptive Layer Normalization modulation.

    Projects conditioning signal to scale and shift parameters
    for layer normalization. Uses single projection for efficiency.

    Args:
        hidden_dim: Hidden dimension
        num_modulations: Number of modulation outputs (typically 6 for scale+shift x 3 norms)
    """

    def __init__(self, hidden_dim: int, num_modulations: int = 6):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_modulations * hidden_dim, bias=True)
        self.num_modulations = num_modulations
        self.hidden_dim = hidden_dim

    def __call__(self, conditioning: mx.array) -> list[mx.array]:
        """
        Generate modulation parameters.

        Args:
            conditioning: Conditioning signal [batch, hidden_dim]

        Returns:
            List of modulation tensors, each [batch, 1, hidden_dim]
        """
        # Project to all modulation parameters at once
        modulations = self.linear(conditioning)  # [batch, num_mod * hidden_dim]

        # Split into individual modulations
        modulations = modulations.reshape(-1, self.num_modulations, self.hidden_dim)

        # Return as list with sequence dimension added
        return [modulations[:, i : i + 1, :] for i in range(self.num_modulations)]


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Uses gated linear unit with SiLU activation:
    FFN(x) = (SiLU(W1 * x) * W3 * x) @ W2

    Args:
        hidden_dim: Input/output dimension
        mlp_dim: Intermediate dimension (typically 2.7x hidden_dim for SwiGLU)
    """

    def __init__(self, hidden_dim: int = 2560, mlp_dim: int = 6912):
        super().__init__()

        # Gate projection (with SiLU)
        self.w1 = nn.Linear(hidden_dim, mlp_dim, bias=False)
        # Up projection (multiplied with gate)
        self.w3 = nn.Linear(hidden_dim, mlp_dim, bias=False)
        # Down projection
        self.w2 = nn.Linear(mlp_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq, hidden_dim]

        Returns:
            Output tensor [batch, seq, hidden_dim]
        """
        # SwiGLU: SiLU(W1(x)) * W3(x), then project down
        gate = nn.silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)


class NextDiTBlock(nn.Module):
    """NextDiT transformer block.

    Architecture:
    1. AdaLN-modulated self-attention (GQA)
    2. Optional AdaLN-modulated cross-attention (GQA)
    3. AdaLN-modulated SwiGLU FFN

    All with residual connections and pre-normalization.

    Args:
        hidden_dim: Hidden dimension (2560)
        num_query_heads: Number of query attention heads (24)
        num_kv_heads: Number of key-value heads for GQA (8)
        head_dim: Per-head dimension (64)
        mlp_dim: FFN intermediate dimension (6912)
        text_dim: Text embedding dimension for cross-attention
        has_cross_attention: Whether to include cross-attention
        qk_norm: Whether to apply QK normalization
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        num_query_heads: int = 24,
        num_kv_heads: int = 8,
        head_dim: int | None = None,
        mlp_dim: int = 6912,
        text_dim: int = 2560,
        has_cross_attention: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.has_cross_attention = has_cross_attention

        # Pre-normalization layers
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6) if has_cross_attention else None
        self.norm3 = nn.RMSNorm(hidden_dim, eps=1e-6)

        # Self-attention with GQA
        self.attn1 = GQAttention(
            hidden_dim=hidden_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
        )

        # Cross-attention with GQA (optional)
        if has_cross_attention:
            self.attn2 = GQCrossAttention(
                hidden_dim=hidden_dim,
                text_dim=text_dim,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                qk_norm=qk_norm,
            )
        else:
            self.attn2 = None

        # SwiGLU FFN
        self.ffn = SwiGLUFFN(hidden_dim=hidden_dim, mlp_dim=mlp_dim)

        # AdaLN modulation
        # 6 modulations: scale1, shift1, scale2, shift2, scale3, shift3
        # (or 4 if no cross-attention: scale1, shift1, scale3, shift3)
        num_modulations = 6 if has_cross_attention else 4
        self.adaLN_modulation = AdaLNModulation(hidden_dim, num_modulations)

    def __call__(
        self,
        hidden_states: mx.array,
        conditioning: mx.array,
        encoder_hidden_states: mx.array | None = None,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
        attention_mask: mx.array | None = None,
        cross_attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass for NextDiT block.

        Args:
            hidden_states: Input tensor [batch, seq, hidden_dim]
            conditioning: Timestep/guidance conditioning [batch, hidden_dim]
            encoder_hidden_states: Text embeddings for cross-attention [batch, text_seq, text_dim]
            rope_cos: RoPE cosine [seq, head_dim]
            rope_sin: RoPE sine [seq, head_dim]
            attention_mask: Self-attention mask
            cross_attention_mask: Cross-attention mask

        Returns:
            Output tensor [batch, seq, hidden_dim]
        """
        # Get AdaLN modulation parameters
        modulations = self.adaLN_modulation(conditioning)

        if self.has_cross_attention:
            scale1, shift1, scale2, shift2, scale3, shift3 = modulations
        else:
            scale1, shift1, scale3, shift3 = modulations

        # Self-attention with AdaLN
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states * (1 + scale1) + shift1
        hidden_states = self.attn1(
            hidden_states,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Cross-attention with AdaLN (if enabled)
        if self.has_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            if self.norm2 is not None:
                hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + scale2) + shift2
            hidden_states = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
            )
            hidden_states = residual + hidden_states

        # FFN with AdaLN
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = hidden_states * (1 + scale3) + shift3
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
