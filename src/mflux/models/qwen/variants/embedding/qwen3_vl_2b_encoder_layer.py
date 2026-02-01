"""Qwen3-VL 2B variant encoder layer.

This encoder layer is configured for the 2B model with:
- hidden_size: 2048 (vs 3584 in 7B)
- num_attention_heads: 16 (vs 28 in 7B)
- num_key_value_heads: 8 (vs 4 in 7B)
- intermediate_size: 8192 (vs 18944 in 7B)
- No attention bias (key difference from 7B)
- QK normalization (key difference from 7B)
"""

import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_mlp import QwenMLP
from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm

from .qwen3_vl_2b_attention import Qwen3VL2BAttention


class Qwen3VL2BEncoderLayer(nn.Module):
    """Encoder layer for Qwen3-VL 2B model.

    Architecture matches Qwen3-VL-Embedding-2B and Qwen3-VL-Reranker-2B.
    Key differences from 7B: no attention bias, QK norms, num_kv_heads=8.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,  # Changed from 4 to 8
        intermediate_size: int = 8192,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.input_layernorm = QwenRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3VL2BAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling={"mrope_section": [16, 24, 24]},
        )
        self.post_attention_layernorm = QwenRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = QwenMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
