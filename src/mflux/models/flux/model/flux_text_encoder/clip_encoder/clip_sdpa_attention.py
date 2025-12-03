import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common.config import ModelConfig


class CLIPSdpaAttention(nn.Module):
    head_dimension = 64
    batch_size = 1
    num_heads = 12

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(input_dims=768, output_dims=768)
        self.k_proj = nn.Linear(input_dims=768, output_dims=768)
        self.v_proj = nn.Linear(input_dims=768, output_dims=768)
        self.out_proj = nn.Linear(input_dims=768, output_dims=768)

    def __call__(self, hidden_states: mx.array, causal_attention_mask: mx.array) -> mx.array:
        causal_attention_mask = causal_attention_mask.astype(ModelConfig.precision)

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = CLIPSdpaAttention.reshape_and_transpose(query, self.batch_size, self.num_heads, self.head_dimension)
        key = CLIPSdpaAttention.reshape_and_transpose(key, self.batch_size, self.num_heads, self.head_dimension)
        value = CLIPSdpaAttention.reshape_and_transpose(value, self.batch_size, self.num_heads, self.head_dimension)

        scale = 1 / mx.sqrt(query.shape[-1])
        hidden_states = scaled_dot_product_attention(query, key, value, scale=scale, mask=causal_attention_mask)

        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = mx.reshape(hidden_states, (self.batch_size, -1, self.num_heads * self.head_dimension))

        hidden_states = self.out_proj(hidden_states)
        return hidden_states

    @staticmethod
    def reshape_and_transpose(x, batch_size, num_heads, head_dim):
        return mx.transpose(mx.reshape(x, (batch_size, -1, num_heads, head_dim)), (0, 2, 1, 3))
