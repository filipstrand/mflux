from typing import Tuple

import mlx.core as mx
from mlx import nn

from .smol_lm3_3b_attention import SmolLM3_3B_SelfAttention
from .smol_lm3_3b_mlp import SmolLM3_3B_MLP
from .smol_lm3_3b_rms_norm import SmolLM3_3B_RMSNorm


class SmolLM3_3B_EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        intermediate_size: int = 11_008,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 65_536,
        rope_theta: float = 5_000_000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.input_layernorm = SmolLM3_3B_RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = SmolLM3_3B_SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
        )
        self.post_attention_layernorm = SmolLM3_3B_RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SmolLM3_3B_MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None,
        cos_sin: Tuple[mx.array, mx.array],
        layer_idx: int | None = None,
    ) -> mx.array:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cos_sin=cos_sin,
        )
        hidden_states = residual + hidden_states

        # Feed-forward block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
