import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.siglip_vision_transformer.siglip_encoder import SiglipEncoder
from mflux.models.flux.model.siglip_vision_transformer.siglip_multi_head_attention_pooling_head import (
    SiglipMultiHeadAttentionPoolingHead,
)
from mflux.models.flux.model.siglip_vision_transformer.siglip_vision_embeddings import SiglipVisionEmbeddings


class SiglipVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings()
        self.encoder = SiglipEncoder()
        self.post_layernorm = nn.LayerNorm(dims=1152, eps=1e-6)
        self.head = SiglipMultiHeadAttentionPoolingHead()

    def __call__(self, pixel_values: mx.array) -> mx.array:
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(hidden_states)
        hidden_state = self.post_layernorm(encoder_outputs)
        pooler_output = self.head(hidden_state)
        return hidden_state, pooler_output
