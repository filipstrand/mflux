import mlx.core as mx
from mlx import nn

from mflux.models.text_encoder.clip_encoder.clip_encoder_layer import CLIPEncoderLayer


class EncoderCLIP(nn.Module):

    def __init__(self, num_encoder_layers: int):
        super().__init__()
        self.layers = [CLIPEncoderLayer(i) for i in range(num_encoder_layers)]

    def forward(self, tokens: mx.array, causal_attention_mask: mx.array) -> mx.array:
        hidden_states = tokens
        for encoder_layer in self.layers:
            hidden_states = encoder_layer.forward(
                hidden_states,
                causal_attention_mask
            )
        return hidden_states
