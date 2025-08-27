import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_attention import QwenAttention
from mflux.models.qwen.model.qwen_text_encoder.qwen_mlp import QwenMLP
from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm


class QwenEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        intermediate_size: int = 18944,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.input_layernorm = QwenRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = QwenAttention(
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
