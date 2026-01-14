"""Encoder layer for Mistral3 text encoder."""

import mlx.core as mx
from mlx import nn

from mflux.models.flux2.model.flux2_text_encoder.mistral3_attention import Mistral3Attention
from mflux.models.flux2.model.flux2_text_encoder.mistral3_mlp import Mistral3MLP
from mflux.models.flux2.model.flux2_text_encoder.mistral3_rms_norm import Mistral3RMSNorm


class Mistral3EncoderLayer(nn.Module):
    """Single transformer layer for Mistral3.

    Pre-norm architecture with RMSNorm, GQA attention, and SwiGLU MLP.

    Args:
        hidden_size: Hidden dimension (5120)
        num_heads: Number of query attention heads (32)
        num_kv_heads: Number of key/value heads (8)
        intermediate_size: MLP intermediate dimension (32768)
        rms_norm_eps: RMSNorm epsilon (1e-5)
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        intermediate_size: int = 32768,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Pre-norm before attention
        self.input_layernorm = Mistral3RMSNorm(hidden_size, eps=rms_norm_eps)

        # Self-attention
        self.self_attn = Mistral3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=128,  # Mistral3 uses 128 head_dim
        )

        # Pre-norm before MLP
        self.post_attention_layernorm = Mistral3RMSNorm(hidden_size, eps=rms_norm_eps)

        # MLP
        self.mlp = Mistral3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """Forward pass through encoder layer.

        Args:
            hidden_states: Input tensor [B, seq_len, hidden_size]
            attention_mask: Attention mask [B, 1, seq_len, seq_len]
            position_embeddings: Tuple of (cos, sin) for RoPE

        Returns:
            Output tensor [B, seq_len, hidden_size]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
