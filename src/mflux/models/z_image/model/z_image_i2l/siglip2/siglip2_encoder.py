import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_i2l.siglip2.siglip2_encoder_layer import Siglip2EncoderLayer


class Siglip2Encoder(nn.Module):
    """SigLIP2-G384 encoder: 40 transformer layers."""

    def __init__(self):
        super().__init__()
        self.layers = [Siglip2EncoderLayer() for _ in range(40)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
