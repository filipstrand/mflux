import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_encoder import Siglip2Encoder
from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_pooling_head import Siglip2PoolingHead
from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_vision_embeddings import Siglip2VisionEmbeddings


class Siglip2VisionTransformer(nn.Module):
    """SigLIP2-G384 vision transformer for i2L image encoding.

    Config: hidden_size=1536, intermediate_size=6144, num_hidden_layers=40,
            num_attention_heads=16, image_size=384, patch_size=16.

    Input:  pixel_values (B, 3, 384, 384) normalized to [-1, 1]
    Output: pooler_output (B, 1536) — pooled representation per image
    """

    def __init__(self):
        super().__init__()
        self.embeddings = Siglip2VisionEmbeddings()
        self.encoder = Siglip2Encoder()
        self.post_layernorm = nn.LayerNorm(dims=1536, eps=1e-6)
        self.head = Siglip2PoolingHead()

    def __call__(self, pixel_values: mx.array) -> mx.array:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        pooler_output = self.head(hidden_states)
        return pooler_output
