import mlx.core as mx
from mlx import nn

from mflux.models.siglip_vision_transformer.siglip_mlp import SiglipMLP


class SiglipMultiHeadAttentionPoolingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = nn.Parameter(mx.randn(1, 1, 1152))
        self.attention = torch.nn.MultiheadAttention(1152, 16, batch_first=True)
        self.layernorm = nn.LayerNorm(1152, eps=1e-6)
        self.mlp = SiglipMLP()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size = hidden_states.shape[0]

        # Create query vectors for each item in the batch
        query = mx.broadcast_to(self.query, (batch_size, 1, hidden_states.shape[-1]))

        # Apply attention to get weighted mean of the sequence
        pooled_output = self.attention(query, hidden_states, hidden_states)

        # Squeeze the sequence dimension
        pooled_output = pooled_output[:, 0]

        # Project to the final dimension
        pooled_output = self.projection(pooled_output)

        # Normalize the output
        pooled_output = pooled_output / mx.linalg.norm(pooled_output, axis=1, keepdims=True)

        return pooled_output
