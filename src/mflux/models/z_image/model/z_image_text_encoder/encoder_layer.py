import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_text_encoder.attention import Attention
from mflux.models.z_image.model.z_image_text_encoder.mlp import MLP


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2560,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(hidden_size, num_attention_heads, num_key_value_heads, head_dim)
        self.mlp = MLP(hidden_size, intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), attention_mask, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states
