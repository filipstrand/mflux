import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.siglip_vision_transformer.siglip_encoder_layer import SiglipEncoderLayer


class SiglipEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [SiglipEncoderLayer() for _ in range(27)]

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
